//! Raycaster for the mazeworld exercises. Mirrors mazeworld/render_numpy.py step by step,
//! so both backends draw the same pictures; this one is parallel over poses and ~2 orders
//! of magnitude faster.
use numpy::{IntoPyArray, PyArray4, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const FOV: f64 = 1.2;
const CEIL: [f64; 3] = [0.18, 0.16, 0.20];
const FLOOR: [f64; 3] = [0.42, 0.38, 0.34];
const FOG: f64 = 0.18;

struct Hit {
    dist: f64,
    side: u8,
    cell: u8,
}

fn cast(grid: &[u8], gx: usize, gy: usize, px: f64, py: f64, mut rdx: f64, mut rdy: f64) -> Hit {
    if rdx.abs() < 1e-9 {
        rdx = 1e-9;
    }
    if rdy.abs() < 1e-9 {
        rdy = 1e-9;
    }
    let mut mx = px.floor() as i64;
    let mut my = py.floor() as i64;
    let ddx = (1.0 / rdx).abs();
    let ddy = (1.0 / rdy).abs();
    let (sx, mut side_x) = if rdx < 0.0 { (-1, (px - mx as f64) * ddx) } else { (1, (mx as f64 + 1.0 - px) * ddx) };
    let (sy, mut side_y) = if rdy < 0.0 { (-1, (py - my as f64) * ddy) } else { (1, (my as f64 + 1.0 - py) * ddy) };
    let mut side = 0u8;
    let clampx = |v: i64| v.clamp(0, gx as i64 - 1) as usize;
    let clampy = |v: i64| v.clamp(0, gy as i64 - 1) as usize;
    for _ in 0..(gx + gy + 2) {
        if side_x < side_y {
            side_x += ddx;
            mx += sx;
            side = 0;
        } else {
            side_y += ddy;
            my += sy;
            side = 1;
        }
        if grid[clampy(my) * gx + clampx(mx)] > 0 {
            break;
        }
    }
    let dist = if side == 0 { side_x - ddx } else { side_y - ddy };
    Hit { dist: dist.max(1e-3), side, cell: grid[clampy(my) * gx + clampx(mx)] }
}

fn render_one(grid: &[u8], gx: usize, gy: usize, pal: &[[f64; 3]], pose: [f64; 3], size: usize, ss: usize, out: &mut [u8]) {
    let r = size * ss;
    let (px, py, a) = (pose[0], pose[1], pose[2]);
    let (dx, dy) = (a.cos(), a.sin());
    let tanh = (FOV / 2.0).tan();
    let mut acc = vec![0.0f64; size * size * 3];
    for col in 0..r {
        let cam = (2.0 * (col as f64 + 0.5) / r as f64 - 1.0) * tanh;
        let hit = cast(grid, gx, gy, px, py, dx - dy * cam, dy + dx * cam);
        let shade = if hit.side == 0 { 1.0 } else { 0.7 } * (-FOG * hit.dist).exp();
        let half = 0.5 / hit.dist;
        let c = pal[hit.cell as usize];
        for row in 0..r {
            let rr = (row as f64 + 0.5) / r as f64 - 0.5;
            let rgb = if rr.abs() < half {
                [c[0] * shade, c[1] * shade, c[2] * shade]
            } else {
                let fc = (-FOG * (0.5 / rr.abs().max(1e-3))).exp();
                let base = if rr < 0.0 { CEIL } else { FLOOR };
                [base[0] * fc, base[1] * fc, base[2] * fc]
            };
            let o = ((row / ss) * size + col / ss) * 3;
            acc[o] += rgb[0];
            acc[o + 1] += rgb[1];
            acc[o + 2] += rgb[2];
        }
    }
    let norm = (ss * ss) as f64;
    for (o, v) in out.iter_mut().zip(acc.iter()) {
        *o = (v / norm * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

/// render(grid (Gy, Gx) uint8, palette (K, 3) float64, poses (N, 3) float64, size, ss)
/// -> (N, size, size, 3) uint8
#[pyfunction]
#[pyo3(signature = (grid, palette, poses, size=16, ss=4))]
fn render<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'py, u8>,
    palette: PyReadonlyArray2<'py, f64>,
    poses: PyReadonlyArray2<'py, f64>,
    size: usize,
    ss: usize,
) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let g = grid.as_array();
    let (gy, gx) = (g.shape()[0], g.shape()[1]);
    let gridv: Vec<u8> = g.iter().copied().collect();
    let pal: Vec<[f64; 3]> = palette.as_array().outer_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let ps: Vec<[f64; 3]> = poses.as_array().outer_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let n = ps.len();
    let frame = size * size * 3;
    let mut out = vec![0u8; n * frame];
    py.detach(|| {
        out.par_chunks_mut(frame)
            .zip(ps.par_iter())
            .for_each(|(o, p)| render_one(&gridv, gx, gy, &pal, *p, size, ss, o));
    });
    let arr = numpy::ndarray::Array4::from_shape_vec((n, size, size, 3), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

#[pymodule]
fn mazeworld_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render, m)?)?;
    Ok(())
}
