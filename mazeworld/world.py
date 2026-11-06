"""A tiny maze world: random block mazes, an explorer policy, and episode generation.

Maze generation, dynamics and the policy are cheap and live here in NumPy. The pictures come
from `museum.render`: the mazes drawn as a small museum (brick walls with paintings, ceiling
lamps). `render_numpy` and the Rust crate hold the older flat-colored raycaster.
"""
import numpy as np

from . import museum

FORWARD, LEFT, RIGHT = 0, 1, 2
ACTIONS = ["forward", "turn left", "turn right"]
STEP = 0.15  # cells per forward step
TURN = np.pi / 16  # radians per turn step
RADIUS = 0.2  # the agent keeps this distance from walls

# Index 0 is free space. The museum renderer uses the block index only to seed which paintings
# hang where; the colors are those of the old flat-colored renderer (render_numpy, Rust).
PALETTE = np.array([
    [0.00, 0.00, 0.00],
    [0.90, 0.20, 0.15],  # red
    [0.20, 0.75, 0.25],  # green
    [0.20, 0.40, 0.95],  # blue
    [0.95, 0.80, 0.15],  # yellow
    [0.85, 0.85, 0.85],  # white
])


def backend():
    return "numpy"


SUPERSAMPLE = {16: 3, 32: 2}


def render(grid, poses, size=16, ss=None):
    """Render first-person views. grid (Gy, Gx) uint8, poses (N, 3) -> (N, size, size, 3) uint8."""
    return museum.render(grid, poses, size, ss or SUPERSAMPLE.get(size))


def make_maze(rng, n=4, loops=0.2, n_colors=len(PALETTE) - 1):
    """Recursive-backtracker maze on n x n cells -> (2n+1, 2n+1) grid. Each wall block gets a
    random color 1..n_colors. `loops` is the fraction of interior walls knocked out afterwards."""
    g = 2 * n + 1
    free = np.zeros((g, g), bool)
    seen = np.zeros((n, n), bool)
    stack = [(int(rng.integers(n)), int(rng.integers(n)))]
    seen[stack[0]] = True
    free[2 * stack[0][0] + 1, 2 * stack[0][1] + 1] = True
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= i + di < n and 0 <= j + dj < n and not seen[i + di, j + dj]]
        if not nbrs:
            stack.pop()
            continue
        a, b = nbrs[rng.integers(len(nbrs))]
        seen[a, b] = True
        free[2 * a + 1, 2 * b + 1] = True
        free[i + a + 1, j + b + 1] = True  # the wall between the two cells
        stack.append((a, b))
    # knock out some interior walls that sit between two free cells, for loops and open areas
    for y in range(1, g - 1):
        for x in range(1, g - 1):
            if free[y, x] or rng.random() > loops:
                continue
            if (free[y - 1, x] and free[y + 1, x]) or (free[y, x - 1] and free[y, x + 1]):
                free[y, x] = True
    colors = rng.integers(1, n_colors + 1, size=(g, g)).astype(np.uint8)
    return np.where(free, 0, colors).astype(np.uint8)


def _blocked(grid, x, y):
    gy, gx = grid.shape
    for ox in (-RADIUS, RADIUS):
        for oy in (-RADIUS, RADIUS):
            cx, cy = int(np.floor(x + ox)), int(np.floor(y + oy))
            if not (0 <= cx < gx and 0 <= cy < gy) or grid[cy, cx] > 0:
                return True
    return False


def step(grid, pose, action):
    """Dynamics: forward moves STEP cells unless that would hit a wall; turns rotate by TURN."""
    x, y, a = pose
    if action == FORWARD:
        nx, ny = x + STEP * np.cos(a), y + STEP * np.sin(a)
        if not _blocked(grid, nx, ny):
            x, y = nx, ny
    elif action == LEFT:
        a = a - TURN
    elif action == RIGHT:
        a = a + TURN
    return np.array([x, y, (a + np.pi) % (2 * np.pi) - np.pi])


def random_start(grid, rng):
    ys, xs = np.nonzero(grid == 0)
    k = rng.integers(len(xs))
    return np.array([xs[k] + 0.5, ys[k] + 0.5, rng.integers(32) * TURN - np.pi])  # on the lattice of turn angles


def explorer_policy(grid, pose, prev, rng):
    """Mostly walk forward; turn for a few steps at walls or at random. Persistent actions make
    the videos smooth enough to learn from."""
    ahead = pose[:2] + 1.0 * np.array([np.cos(pose[2]), np.sin(pose[2])])
    if _blocked(grid, *ahead):
        return prev if prev in (LEFT, RIGHT) else int(rng.choice([LEFT, RIGHT]))
    if prev is not None and rng.random() < (0.85 if prev == FORWARD else 0.4):
        return prev
    return int(rng.choice([FORWARD, LEFT, RIGHT], p=[0.7, 0.15, 0.15]))


def episode(seed, length=32, size=16, maze_n=4):
    """One random maze and one explorer trajectory through it.
    Returns frames (length, size, size, 3) uint8, actions (length,) int64, where actions[t] leads
    from frames[t] to frames[t+1] (the last one is unused), poses (length, 3), and the grid."""
    rng = np.random.default_rng(seed)
    grid = make_maze(rng, maze_n)
    pose = random_start(grid, rng)
    poses, actions, a = [], [], None
    for _ in range(length):
        poses.append(pose)
        a = explorer_policy(grid, pose, a, rng)
        actions.append(a)
        pose = step(grid, pose, a)
    poses = np.stack(poses)
    return dict(frames=render(grid, poses, size), actions=np.array(actions), poses=poses, grid=grid)


def dataset(n, length=32, size=16, seed=0, maze_n=4):
    """n episodes stacked: frames (n, length, size, size, 3) uint8, actions (n, length)."""
    eps = [episode(seed * 1_000_003 + i, length, size, maze_n) for i in range(n)]
    return dict(frames=np.stack([e["frames"] for e in eps]),
                actions=np.stack([e["actions"] for e in eps]),
                poses=np.stack([e["poses"] for e in eps]),
                grids=np.stack([e["grid"] for e in eps]))


def topdown(grid, poses=None, cell=8):
    """Bird's-eye picture of the maze (uint8 RGB), with the trajectory drawn in white and the
    last pose marked."""
    img = np.where((grid > 0)[..., None], np.array([214, 208, 196], np.uint8), np.uint8(40))
    img = img.repeat(cell, 0).repeat(cell, 1)
    if poses is not None:
        for x, y, _ in poses:
            px, py = int(x * cell), int(y * cell)
            img[max(py - 1, 0):py + 1, max(px - 1, 0):px + 1] = 255
        x, y, a = poses[-1]
        for s in np.linspace(0, 0.6, 6):
            px, py = int((x + s * np.cos(a)) * cell), int((y + s * np.sin(a)) * cell)
            img[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2] = (255, 120, 0)
    return img
