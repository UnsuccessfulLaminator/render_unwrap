use std::ops::{MulAssign, Range};
use std::path::PathBuf;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::process::Command;
use ndarray::prelude::*;
use ndarray_npy::ReadNpyExt;
use ndarray_linalg::LeastSquaresSvd;
use clap::Parser;
use tempfile::NamedTempFile;



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

    #[arg(short, long, default_value_t = 1., value_name = "PERIOD")]
    /// Period over which the color cycle repeats in the z-direction
    color_period: f64,

    #[arg(long, default_value_t = ("jpeg").to_string())]
    /// Gnuplot backend to use
    backend: String
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
    let (mut min_u, mut max_u) = (f64::MAX, f64::MIN);

    azip!((index (i, j), &u in &uphase, &q in &quality) {
        if q > args.threshold {
            data.extend([j as f64, i as f64, u, q]);

            if u > max_u { max_u = u; }
            if u < min_u { min_u = u; }
        }
    });
    
    let n_points = data.len()/4;
    let mut data = Array2::from_shape_vec((n_points, 4), data).unwrap();
    
    subtract_fit(data.view_mut());

    let cmap = colorous::RAINBOW;
    let xlim = 0.0..(w as f64);
    let ylim = (h as f64)..0.0;
    let zlim = args.zlim.unwrap_or(min_u..max_u);
    
    let data_file = NamedTempFile::new_in("")?;
    let mut plot_file = NamedTempFile::new_in("")?;
    let mut writer = BufWriter::new(&data_file);

    for p in data.rows() {
        let (x, y, z, q) = (p[0], p[1], p[2], p[3]);
        let mut color = cmap.eval_continuous((z/args.color_period).rem_euclid(1.));

        writeln!(writer, "{x} {z} {y} 0x{color:X}")?;
    }
    
    drop(writer);

    let data_path = data_file.into_temp_path();
    
    // General plot configuration
    writeln!(plot_file, "set term {} size {},{}", args.backend, dim.0, dim.1)?;
    writeln!(plot_file, "set output '{}'", args.output.display())?;
    writeln!(plot_file, "set xrange [{}:{}]", xlim.start, xlim.end)?;
    writeln!(plot_file, "set zrange [{}:{}]", ylim.start, ylim.end)?;
    writeln!(plot_file, "set yrange [{}:{}]", zlim.start, zlim.end)?;
    writeln!(plot_file, "set xlabel 'x'; set ylabel 'depth'; set zlabel 'y'")?;
    writeln!(plot_file, "set view 75, 20")?;
    writeln!(plot_file, "set xyplane 0")?;
    writeln!(plot_file, "set multiplot")?;
    writeln!(plot_file, "set nokey")?;
    
    // Plot x-y axes with a grid for the model to sit on
    writeln!(plot_file, "unset border")?;
    writeln!(plot_file, "set isosamples 2")?;
    writeln!(plot_file, "set grid xtics ytics ztics")?;
    writeln!(plot_file, "set tics offset screen 0,-0.01")?;
    writeln!(plot_file, "splot {} lc 'black'", ylim.start)?;
    
    // Plot the point cloud model with no additional axes
    writeln!(plot_file, "set hidden3d")?;
    writeln!(plot_file, "unset xtics; unset ytics; unset grid; unset parametric")?;
    writeln!(plot_file, "splot '{}' lc rgb variable pt 7 ps 0.2", data_path.display())?;

    let plot_path = plot_file.into_temp_path();

    Command::new("gnuplot")
        .args([&plot_path])
        .status()?;

    Ok(())
}

// Fit the equation z = (ax2+bx+c)/(dx2+ex+1) to the given array of points,
// which is a good approximation to the general equation for the phase image
// produced by a plane target. Subtract this fit from the array.
// Each row of `points` is [x, y, z].
fn subtract_fit(mut points: ArrayViewMut2<f64>) {
    let xy = points.slice(s![.., ..2]);
    let z = points.slice(s![.., 2]);
    let mut matrix = Array2::<f64>::ones((points.nrows(), 5)); // [[-x, -y, 1, xz, yz], ...]
    
    matrix.slice_mut(s![.., ..2]).assign(&xy);
    matrix.slice_mut(s![.., ..2]).mul_assign(-1.);
    matrix.slice_mut(s![.., 3..]).assign(&xy);
    matrix.slice_mut(s![.., 3]).mul_assign(&z);
    matrix.slice_mut(s![.., 4]).mul_assign(&z);

    let coeffs: Array1<f64> = matrix.least_squares(&z)
        .expect("Could not find least squares fit for given points")
        .solution;

    let fit = matrix.dot(&coeffs);
    let mut z = points.slice_mut(s![.., 2]);
    
    z -= &fit;
}
