"""The world of notebook 4: the same block mazes and dynamics as mazeworld (notebook 3), with
6x6-cell mazes rendered with textured walls, a tiled floor, sky, distance haze and a goal
beacon (a pink orb with a light beam), plus the shortest-path expert and the dataset generator.

Image coordinates have square pixels: x_img = lateral / (depth T), y_img = -height / (depth T),
both in [-1, 1], with T = tan(FOV / 2). The camera sits at height 0.5, walls are 1 high.
"""
import numpy as np
from collections import deque

from . import world as mw

FOV = 1.3
T = np.tan(FOV / 2)
EYE = 0.5

WALLS = np.array([
    [0.00, 0.00, 0.00],
    [0.80, 0.42, 0.31],  # terracotta
    [0.46, 0.63, 0.43],  # sage
    [0.34, 0.49, 0.72],  # slate blue
    [0.88, 0.70, 0.32],  # ochre
    [0.84, 0.80, 0.72],  # limestone
])
SKY = np.array([0.36, 0.54, 0.82])
HAZE = np.array([0.82, 0.85, 0.89])
TILE_A = np.array([0.62, 0.58, 0.52])
TILE_B = np.array([0.55, 0.51, 0.46])
GOAL = np.array([1.00, 0.28, 0.62])
GOAL_LIGHT = np.array([1.00, 0.70, 0.86])
SUN = np.array([0.55, 0.83])  # direction the light comes from, in map coordinates
FOG = 0.10
ORB_R, ORB_H = 0.17, 0.0  # goal orb radius and height relative to the eye
BEAM_W = 0.035


def _hash(*a):
    s = sum(x * c for x, c in zip(a, (12.9898, 78.233, 37.719, 4.581)))
    return np.modf(np.abs(np.sin(s) * 43758.5453))[0]


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def cast(grid, poses, R):
    """DDA raycast of R columns. Returns dict of (n, R) arrays: perpendicular distance, side
    (0 = x face), hit cell, u along the face, and the ray directions."""
    poses = np.asarray(poses, np.float64).reshape(-1, 3)
    n = len(poses)
    gy, gx = grid.shape
    cam = (2 * (np.arange(R) + 0.5) / R - 1) * T
    px, py, a = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
    dx, dy = np.cos(a), np.sin(a)
    rdx, rdy = dx - dy * cam, dy + dx * cam
    rdx = np.where(np.abs(rdx) < 1e-9, 1e-9, rdx)
    rdy = np.where(np.abs(rdy) < 1e-9, 1e-9, rdy)
    mx = np.floor(px).astype(np.int64) * np.ones((1, R), np.int64)
    my = np.floor(py).astype(np.int64) * np.ones((1, R), np.int64)
    ddx, ddy = np.abs(1 / rdx), np.abs(1 / rdy)
    sx, sy = np.where(rdx < 0, -1, 1), np.where(rdy < 0, -1, 1)
    side_x = np.where(rdx < 0, (px - mx) * ddx, (mx + 1 - px) * ddx)
    side_y = np.where(rdy < 0, (py - my) * ddy, (my + 1 - py) * ddy)
    hit = np.zeros((n, R), bool)
    side = np.zeros((n, R), np.int64)
    for _ in range(gx + gy + 2):
        go_x = (side_x < side_y) & ~hit
        go_y = ~go_x & ~hit
        side_x = np.where(go_x, side_x + ddx, side_x)
        mx = np.where(go_x, mx + sx, mx)
        side_y = np.where(go_y, side_y + ddy, side_y)
        my = np.where(go_y, my + sy, my)
        side = np.where(go_x, 0, np.where(go_y, 1, side))
        hit |= grid[np.clip(my, 0, gy - 1), np.clip(mx, 0, gx - 1)] > 0
        if hit.all():
            break
    dist = np.maximum(np.where(side == 0, side_x - ddx, side_y - ddy), 1e-3)
    hx, hy = px + dist * rdx, py + dist * rdy
    u = np.where(side == 0, hy, hx)
    u = u - np.floor(u)
    # normal of the face that was hit, for sun shading
    nx = np.where(side == 0, -sx, 0)
    ny = np.where(side == 1, -sy, 0)
    return dict(dist=dist, side=side, mx=np.clip(mx, 0, gx - 1), my=np.clip(my, 0, gy - 1), u=u,
                rdx=rdx, rdy=rdy, hx=hx, hy=hy, nx=nx, ny=ny)


def wall_texture(u, v, mx, my):
    """Stone courses: 3 courses of 2 blocks, staggered, mortar lines, per-block jitter, dark
    seams at the block edges, a capstone highlight and a contact shadow at the floor."""
    course = np.floor(v * 3)
    uu = u + 0.25 * (course % 2)
    brick = np.floor(uu * 2)
    fu, fv = uu * 2 - brick, v * 3 - course
    jitter = 0.9 + 0.18 * _hash(mx, my, course, brick)
    mortar = 1 - 0.28 * np.maximum(1 - smoothstep(0.0, 0.06, np.minimum(fu, 1 - fu)) * 1.0,
                                   1 - smoothstep(0.0, 0.08, np.minimum(fv, 1 - fv)))
    seam = 1 - 0.45 * (1 - smoothstep(0.0, 0.03, np.minimum(u, 1 - u)))
    cap = 1 + 0.12 * (v < 0.05)
    contact = 1 - 0.35 * smoothstep(0.82, 1.0, v)
    return jitter * mortar * seam * cap * contact


SUPERSAMPLE = {16: 4, 32: 2, 64: 3}


def render(grid, poses, goal=None, size=64, ss=None, t=0.0):
    """grid (Gy, Gx), poses (N, 3), goal (x, y) or None -> (N, size, size, 3) uint8.
    ss x ss rays per pixel (antialiasing)."""
    ss = ss or SUPERSAMPLE.get(size, 2)
    grid = np.asarray(grid)
    poses = np.asarray(poses, np.float64).reshape(-1, 3)
    out = []
    for i in range(0, len(poses), 16):
        out.append(_render(grid, poses[i:i + 16], goal, size, ss, t))
    return np.concatenate(out)


def _render(grid, poses, goal, size, ss, t):
    n, R = len(poses), size * ss
    c = cast(grid, poses, R)
    dist = c["dist"]
    y = (2 * (np.arange(R) + 0.5) / R - 1)  # image y, -1 top .. 1 bottom
    Y = y[None, :, None]  # (1, R, 1) rows; columns are the last axis
    half = (EYE / (dist * T))[:, None, :]  # wall spans |y| < half
    is_wall = np.abs(Y) < half

    # walls
    v = (Y + half) / (2 * half)  # 0 at the top of the wall, 1 at the floor
    tex = wall_texture(c["u"][:, None, :], v, c["mx"][:, None, :], c["my"][:, None, :])
    sun = SUN / np.linalg.norm(SUN)
    lam = 0.62 + 0.38 * np.maximum(0, c["nx"] * sun[0] + c["ny"] * sun[1])
    base = WALLS[grid[c["my"], c["mx"]]]  # (n, R, 3)
    wall = base[:, None] * (tex * lam[:, None, :])[..., None]
    fw = np.exp(-FOG * dist)[:, None, :, None]
    wall = wall * fw + HAZE * (1 - fw)

    # floor: the point seen at image row y > 0 is at depth EYE / (y T)
    yf = np.maximum(Y, 1e-4)
    d = EYE / (yf * T)  # (1, R, 1)
    fx = poses[:, 0, None, None] + d * c["rdx"][:, None, :]
    fy = poses[:, 1, None, None] + d * c["rdy"][:, None, :]
    ix, iy = np.floor(fx * 2), np.floor(fy * 2)
    chk = ((ix + iy) % 2)[..., None]
    floor = TILE_A * (1 - chk) + TILE_B * chk
    gu, gv = fx * 2 - ix, fy * 2 - iy
    grout = 1 - 0.18 * (1 - smoothstep(0, 0.05, np.minimum(np.minimum(gu, 1 - gu), np.minimum(gv, 1 - gv))))
    # ambient occlusion next to walls
    cx, cy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    ux, uy = fx - cx, fy - cy
    gyn, gxn = grid.shape
    def wall_at(ox, oy):
        return grid[np.clip(cy + oy, 0, gyn - 1), np.clip(cx + ox, 0, gxn - 1)] > 0
    ao = np.ones_like(fx)
    for ox, oy, dd in ((-1, 0, ux), (1, 0, 1 - ux), (0, -1, uy), (0, 1, 1 - uy)):
        ao *= 1 - 0.45 * wall_at(ox, oy) * np.exp(-dd / 0.10)
    floor = floor * (grout * ao)[..., None]
    if goal is not None:
        r = np.hypot(fx - goal[0], fy - goal[1])
        ring = (smoothstep(0.30, 0.26, r) * 0.55 + 0.45 * smoothstep(0.30, 0.27, r) * smoothstep(0.20, 0.24, r))
        floor = floor * (1 - ring[..., None]) + GOAL * ring[..., None]
    ff = np.exp(-FOG * d)[..., None]
    floor = floor * ff + HAZE * (1 - ff)

    # sky
    s = smoothstep(0, 0.9, -Y)[..., None] ** 0.8
    sky = (HAZE * (1 - s) + SKY * s) * np.ones((n, 1, R, 1))

    img = np.where(is_wall[..., None], wall, np.where((Y > 0)[..., None], floor, sky))

    if goal is not None:
        img = _goal(img, poses, goal, c, Y, half, R, t)
    img = img.reshape(n, R // ss, ss, R // ss, ss, 3).mean((2, 4))
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


def _goal(img, poses, goal, c, Y, half, R, t):
    """Floating orb at the goal plus a light beam that rises above the walls."""
    X = (2 * (np.arange(R) + 0.5) / R - 1)[None, None, :]
    for b in range(len(poses)):
        px, py, a = poses[b]
        D = np.array([np.cos(a), np.sin(a)])
        P = np.array([-D[1], D[0]])
        rel = np.asarray(goal) - (px, py)
        depth, lat = rel @ D, rel @ P
        if depth < 0.08:
            continue
        xc = lat / (depth * T)
        wall_d = c["dist"][b][None, :]
        in_front = (wall_d > depth) | (np.abs(Y[0]) >= half[b])  # (R, R): beam not hidden by a wall
        # beam from the floor up into the sky
        bw = BEAM_W / (depth * T)
        prof = np.exp(-((X[0] - xc) / bw) ** 2)
        y_floor = EYE / (depth * T)
        beam = prof * (Y[0] < y_floor) * (0.55 + 0.25 * smoothstep(0, -1, Y[0]))
        beam = beam * in_front
        # orb
        yc = -(ORB_H + 0.03 * np.sin(2 * t)) / (depth * T)
        rr = ORB_R / (depth * T)
        q = np.sqrt((X[0] - xc) ** 2 + (Y[0] - yc) ** 2) / rr
        orb_vis = in_front
        nz = np.sqrt(np.clip(1 - q ** 2, 0, 1))
        shade = 0.5 + 0.5 * nz + 0.5 * np.exp(-((X[0] - xc + 0.35 * rr) ** 2 + (Y[0] - yc + 0.35 * rr) ** 2) / (0.06 * rr ** 2))
        inside = (q < 1) & orb_vis
        glow = np.exp(-np.maximum(q - 1, 0) * 3) * (q >= 1) * 0.45 * orb_vis
        im = img[b]
        im = im * (1 - beam[..., None]) + GOAL_LIGHT * beam[..., None]
        im = im * (1 - glow[..., None]) + GOAL_LIGHT * glow[..., None]
        im = np.where(inside[..., None], GOAL * shade[..., None], im)
        img[b] = im
    return img


# ---------------------------------------------------------------- expert

def bfs(grid, start, goal):
    """Shortest path over free cells (4-connected). start, goal: (col, row). Returns a list of cells."""
    gy, gx = grid.shape
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        x, y = cur
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < gx and 0 <= ny < gy and grid[ny, nx] == 0 and (nx, ny) not in prev:
                prev[(nx, ny)] = cur
                q.append((nx, ny))
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def distances(grid, start):
    gy, gx = grid.shape
    D = np.full(grid.shape, -1)
    D[start[1], start[0]] = 0
    q = deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < gx and 0 <= ny < gy and grid[ny, nx] == 0 and D[ny, nx] < 0:
                D[ny, nx] = D[y, x] + 1
                q.append((nx, ny))
    return D


def visibility(grid, poses, n_rays=48):
    """Hit points of n_rays across the field of view, for the map's view cone: (N, n_rays, 2)."""
    c = cast(grid, poses, n_rays)
    return np.stack([c["hx"], c["hy"]], -1)


# ---------------------------------------------------------------- expert and data

STOP = 0.75  # the goal counts as reached at this distance


def free_neighbors(grid, x, y):
    gy, gx = grid.shape
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < gx and 0 <= ny < gy and grid[ny, nx] == 0:
            yield nx, ny


def expert_action(grid, pose, D, goal):
    """Shortest-path controller that works from any pose: aim at the center of the neighbor cell
    that is one step closer to the goal (or at the goal itself in its cell); turn until roughly
    facing it, then go forward. Far off the line between the two cell centers (after a random
    action), it first heads back to the center of its own cell."""
    cx, cy = int(np.floor(pose[0])), int(np.floor(pose[1]))
    c = np.array([cx + 0.5, cy + 0.5])
    d = D[cy, cx]
    target = np.asarray(goal, float)
    if d > 0:
        for nx, ny in free_neighbors(grid, cx, cy):
            if D[ny, nx] == d - 1:
                target = np.array([nx + 0.5, ny + 0.5])
                n = target - c
                if abs((pose[0] - c[0]) * n[1] - (pose[1] - c[1]) * n[0]) > 0.15:
                    target = c
                break
    diff = (np.arctan2(target[1] - pose[1], target[0] - pose[0]) - pose[2] + np.pi) % (2 * np.pi) - np.pi
    if abs(diff) <= mw.TURN / 2 + 1e-9:
        return mw.FORWARD
    return mw.RIGHT if diff > 0 else mw.LEFT


def new_goal(grid, pose, rng, min_dist=6):
    cx, cy = int(np.floor(pose[0])), int(np.floor(pose[1]))
    D = distances(grid, (cx, cy))
    cand = np.argwhere(D >= min_dist)
    if len(cand) == 0:
        cand = np.argwhere(D >= 1)
    gy, gx = cand[rng.integers(len(cand))]
    goal = np.array([gx + 0.5, gy + 0.5])
    return goal, distances(grid, (int(gx), int(gy)))


def episode(seed, expert, L=96, size=64, maze_n=6, eps=0.06, burst=0.02):
    """One training episode of length L. expert=False: mazeworld's explorer (a goal is drawn and
    rendered but not pursued). expert=True: shortest path to the goal with a little action noise
    (eps random actions, occasional bursts), so the data also covers states off the optimal
    path; on arrival a new goal is drawn."""
    rng = np.random.default_rng(seed)
    grid = mw.make_maze(rng, maze_n, loops=0.25)
    pose = mw.random_start(grid, rng)
    goal, D = new_goal(grid, pose, rng)
    poses, actions, goals, dists = [], [], [], []
    prev, burst_left, burst_a = None, 0, 0
    for t in range(L):
        if expert and np.hypot(*(pose[:2] - goal)) < STOP:
            goal, D = new_goal(grid, pose, rng)
        poses.append(pose)
        goals.append(goal)
        dists.append(np.where(grid > 0, 255, np.minimum(D, 254)).astype(np.uint8))
        if expert:
            if burst_left > 0:
                a, burst_left = burst_a, burst_left - 1
            elif rng.random() < burst:
                burst_a, burst_left = int(rng.integers(3)), int(rng.integers(2, 7))
                a = burst_a
            elif rng.random() < eps:
                a = int(rng.integers(3))
            else:
                a = expert_action(grid, pose, D, goal)
        else:
            a = mw.explorer_policy(grid, pose, prev, rng)
        prev = a
        actions.append(a)
        pose = mw.step(grid, pose, a)
    poses, goals = np.array(poses), np.array(goals)
    frames = np.empty((L, size, size, 3), np.uint8)
    # render in runs of constant goal
    starts = [0] + [t for t in range(1, L) if not np.allclose(goals[t], goals[t - 1])] + [L]
    for s, e in zip(starts[:-1], starts[1:]):
        frames[s:e] = render(grid, poses[s:e], goals[s], size=size)
    return dict(frames=frames, actions=np.array(actions, np.int8), poses=poses.astype(np.float32),
                goals=goals.astype(np.float32), dist=np.stack(dists), grid=grid, expert=expert)

def _job(args):
    return episode(*args)


def dataset(n_explore, n_expert, L=96, size=16, seed=0, procs=4, progress=None):
    """n_explore + n_expert episodes as a dict of arrays: frames (N, L, size, size, 3) uint8,
    actions (N, L) int64 (actions[t] leads from frame t to t+1), poses (N, L, 3), goals (N, L, 2),
    dist (N, L, 13, 13) uint8 (BFS distance of every cell to the current goal, 255 = wall),
    grids (N, 13, 13), expert (N,) bool."""
    N = n_explore + n_expert
    jobs = [(seed * 10_000_019 + i, i >= n_explore, L, size) for i in range(N)]
    eps = []
    if procs > 1:
        import multiprocessing as mp
        try:
            pool = mp.Pool(procs)
        except (OSError, ImportError, NotImplementedError):  # no multiprocessing here: render serially
            pool = None
        if pool is not None:
            with pool:
                for e in pool.imap(_job, jobs, chunksize=16):
                    eps.append(e)
                    if progress:
                        progress(len(eps), N)
    for j in jobs[len(eps):]:
        eps.append(episode(*j))
        if progress:
            progress(len(eps), N)
    out = {k: np.stack([e[k] for e in eps]) for k in ("frames", "actions", "poses", "goals", "dist")}
    out["actions"] = out["actions"].astype(np.int64)
    out["grids"] = np.stack([e["grid"] for e in eps])
    out["expert"] = np.array([e["expert"] for e in eps])
    return out


def test_env(seed, min_dist=8):
    """A navigation task: a random maze, a start pose and a goal at least min_dist cells away
    (along the shortest path). Also returns how many steps the expert needs."""
    rng = np.random.default_rng(seed)
    grid = mw.make_maze(rng, 6, loops=0.25)
    pose = mw.random_start(grid, rng)
    goal, D = new_goal(grid, pose, rng, min_dist=min_dist)
    q, n_exp = pose.copy(), 0
    while np.hypot(*(q[:2] - goal)) >= STOP and n_exp < 1000:
        q = mw.step(grid, q, expert_action(grid, q, D, goal))
        n_exp += 1
    return dict(seed=seed, grid=grid, start=pose, goal=goal, D=D,
                dist=np.where(grid > 0, 255, np.minimum(D, 254)).astype(np.uint8), n_expert=n_exp)


def expert_rollout(env, max_steps=400):
    """The expert's poses and actions from env's start to its goal."""
    pose, poses, actions = env["start"].copy(), [env["start"]], []
    while np.hypot(*(pose[:2] - env["goal"])) >= STOP and len(actions) < max_steps:
        a = expert_action(env["grid"], pose, env["D"], env["goal"])
        pose = mw.step(env["grid"], pose, a)
        poses.append(pose)
        actions.append(a)
    return np.array(poses), np.array(actions)
