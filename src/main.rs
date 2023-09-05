use std::ops::Range;
use std::path::PathBuf;
use std::fs::File;
use std::ops::{MulAssign, SubAssign};
use ndarray::prelude::*;
use ndarray_npy::ReadNpyExt;
use ndarray_linalg::LeastSquaresSvd;
use plotters::prelude::*;
use plotters::coord::ranged3d::ProjectionMatrix;
use clap::Parser;



#[derive(Clone, Copy)]
struct Dimensions(usize, usize);

impl std::str::FromStr for Dimensions {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('x').collect();
        
        if parts.len() == 2 {
            let width: usize = parts[0].parse().map_err(|_| "Invalid integer for width")?;
            let height: usize = parts[1].parse().map_err(|_| "Invalid integer for height")?;

            Ok(Self(width, height))
        }
        else {
            Err("Dimensions must be of the form WIDTHxHEIGHT".to_string())
        }
    }
}

impl std::fmt::Display for Dimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.0, self.1)
    }
}

#[derive(Parser)]
struct Args {
    /// Input unwrapped phase
    unwrapped: PathBuf,

    /// Input quality
    quality: PathBuf,

    /// Output image
    output: PathBuf,

    #[arg(short, long, default_value_t = Dimensions(640, 480))]
    /// Dimensions of the output image
    dimensions: Dimensions,

    #[arg(short, long, value_parser = parse_range, allow_hyphen_values = true)]
    /// Range of values for the z-axis
    zlim: Option<Range<f64>>,

    #[arg(short, long, default_value_t = 0.)]
    /// Quality threshold, below which points are not plotted
    threshold: f64,

    #[arg(short, long, action)]
    /// Mirror along the x-axis in 3D space 
    mirror: bool
}

fn parse_range(s: &str) -> Result<Range<f64>, String> {
    let parts: Vec<&str> = s.split("..").collect();

    if parts.len() == 2 {
        let start: f64 = parts[0].parse().map_err(|_| "Invalid float for range start")?;
        let end: f64 = parts[1].parse().map_err(|_| "Invalid float for range end")?;

        Ok(start..end)
    }
    else {
        Err("Range must be of the form START..END".to_string())
    }
}



fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let dim = (args.dimensions.0 as u32, args.dimensions.1 as u32);
    let uphase = Array2::<f64>::read_npy(File::open(&args.unwrapped)?)?;
    let quality = Array2::<f64>::read_npy(File::open(&args.quality)?)?;
    let (h, w) = uphase.dim();

    let mut data = vec![];

    azip!((index (i, j), &u in &uphase, &q in &quality) {
        if q > args.threshold {
            data.extend([j as f64, i as f64, u, q]);
        }
    });
    
    let n_points = data.len()/4;
    let mut data = Array2::from_shape_vec((n_points, 4), data).unwrap();
    
    let b = data.slice(s![.., 2]);
    let mut A = Array2::<f64>::ones((n_points, 5));

    A.slice_mut(s![.., ..2]).assign(&data.slice(s![.., ..2]));
    A.slice_mut(s![.., ..2]).mul_assign(-1.);
    A.slice_mut(s![.., 3..]).assign(&data.slice(s![.., ..2]));
    A.slice_mut(s![.., 3]).mul_assign(&data.slice(s![.., 2]));
    A.slice_mut(s![.., 4]).mul_assign(&data.slice(s![.., 2]));

    let coeffs: Array1<f64> = A.least_squares(&b)?.solution;
    let fit = A.dot(&coeffs);
    
    data.slice_mut(s![.., 2]).sub_assign(&fit);

    let draw = BitMapBackend::new(&args.output, dim).into_drawing_area();

    let cmap = colorous::VIRIDIS;
    let min = data.rows().into_iter().map(|p| p[2]).reduce(f64::min).unwrap();
    let max = data.rows().into_iter().map(|p| p[2]).reduce(f64::max).unwrap();
    let xlim = if args.mirror { (w as f64)..0.0 } else { 0.0..(w as f64) };
    let ylim = (h as f64)..0.0;
    let zlim = args.zlim.unwrap_or(min..max);
    let mut mat = ProjectionMatrix::default();
    let mut chart = ChartBuilder::on(&draw)
        .margin(20)
        .build_cartesian_3d(xlim, ylim, zlim.clone())?;

    chart.with_projection(|p| {
        mat = p.into_matrix();
        mat
    });

    let mut markers: Vec<_> = data.rows()
        .into_iter()
        .map(|p| {
            let color = cmap.eval_continuous((p[2]-zlim.start)/(zlim.end-zlim.start));
            let style = ShapeStyle {
                color: RGBAColor(color.r, color.g, color.b, 1.),
                filled: true,
                stroke_width: 0
            };
            let (x, y, z) = (p[0], p[1], p[2]);
            let depth = mat.projected_depth((x as i32, y as i32, z as i32));

            (Circle::new((x, y, z), 1, style), depth)
        })
        .collect();

    markers.sort_unstable_by_key(|m| m.1);

    draw.fill(&WHITE)?;
    chart.configure_axes().draw()?;
    chart.draw_series(markers.into_iter().map(|m| m.0))?;

    Ok(())
}
