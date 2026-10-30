"""The Rust and NumPy raycasters must draw identical frames. Run after changing either."""
import time

import numpy as np

import mazeworld as mw
from mazeworld import render_numpy

import mazeworld_rs  # pip install ./rust

rng = np.random.default_rng(0)
worst = 0
for s in range(20):
    g = mw.make_maze(np.random.default_rng(s))
    poses = np.stack([mw.random_start(g, rng) for _ in range(200)])
    poses[:, :2] += rng.uniform(-0.3, 0.3, (200, 2))
    poses[:, 2] = rng.uniform(-np.pi, np.pi, 200)
    a = render_numpy.render(g, mw.PALETTE, poses, 16, 4).astype(int)
    b = mazeworld_rs.render(g, mw.PALETTE, poses, 16, 4).astype(int)
    worst = max(worst, np.abs(a - b).max())
print("max abs pixel difference:", worst)
g = mw.make_maze(rng)
poses = np.stack([mw.random_start(g, rng) for _ in range(20000)])
t = time.time()
mazeworld_rs.render(g, mw.PALETTE, poses, 16, 4)
print(f"rust: {20000 / (time.time() - t):,.0f} frames/s")
t = time.time()
render_numpy.render(g, mw.PALETTE, poses[:2000], 16, 4)
print(f"numpy: {2000 / (time.time() - t):,.0f} frames/s")
