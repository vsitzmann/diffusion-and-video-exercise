"""Pure-NumPy raycaster. Same algorithm as the Rust backend (rust/src/lib.rs); used when that is not installed."""
import numpy as np

FOV = 1.2  # horizontal field of view in radians (~69 degrees)
CEIL = np.array([0.18, 0.16, 0.20])
FLOOR = np.array([0.42, 0.38, 0.34])
FOG = 0.18  # brightness falls off like exp(-FOG * distance)


def render(grid, palette, poses, size=16, ss=4):
    """grid: (Gy, Gx) uint8, 0 = free, k > 0 = wall with color palette[k].
    palette: (K, 3) float in [0, 1]. poses: (N, 3) float (x, y, angle).
    Renders at size*ss and box-filters down to (N, size, size, 3) uint8."""
    grid = np.asarray(grid)
    palette = np.asarray(palette, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    n, R = len(poses), size * ss
    gy, gx = grid.shape
    # one ray per high-res column, camera plane spans [-tan(FOV/2), tan(FOV/2)]
    cam = (2 * (np.arange(R) + 0.5) / R - 1) * np.tan(FOV / 2)
    px, py, a = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
    dx, dy = np.cos(a), np.sin(a)
    rdx = dx - dy * cam  # +cam = right of the view direction (y axis points "down" in the map)
    rdy = dy + dx * cam
    rdx = np.where(np.abs(rdx) < 1e-9, 1e-9, rdx)
    rdy = np.where(np.abs(rdy) < 1e-9, 1e-9, rdy)
    mx = np.floor(px).astype(np.int64) * np.ones((1, R), np.int64)
    my = np.floor(py).astype(np.int64) * np.ones((1, R), np.int64)
    ddx, ddy = np.abs(1 / rdx), np.abs(1 / rdy)
    sx = np.where(rdx < 0, -1, 1)
    sy = np.where(rdy < 0, -1, 1)
    side_x = np.where(rdx < 0, (px - mx) * ddx, (mx + 1 - px) * ddx)
    side_y = np.where(rdy < 0, (py - my) * ddy, (my + 1 - py) * ddy)
    hit = np.zeros((n, R), bool)
    side = np.zeros((n, R), np.int64)
    for _ in range(gx + gy + 2):  # DDA: step to the next grid line until a wall is hit
        go_x = side_x < side_y
        go_x &= ~hit
        go_y = ~go_x & ~hit
        side_x = np.where(go_x, side_x + ddx, side_x)
        mx = np.where(go_x, mx + sx, mx)
        side_y = np.where(go_y, side_y + ddy, side_y)
        my = np.where(go_y, my + sy, my)
        side = np.where(go_x, 0, np.where(go_y, 1, side))
        cx, cy = np.clip(mx, 0, gx - 1), np.clip(my, 0, gy - 1)
        hit |= grid[cy, cx] > 0
        if hit.all():
            break
    dist = np.where(side == 0, side_x - ddx, side_y - ddy)
    dist = np.maximum(dist, 1e-3)
    cell = grid[np.clip(my, 0, gy - 1), np.clip(mx, 0, gx - 1)]
    shade = np.where(side == 0, 1.0, 0.7) * np.exp(-FOG * dist)
    color = palette[cell] * shade[..., None]  # (n, R, 3)
    half = 0.5 / dist  # wall half-height in units of image height
    rows = (np.arange(R) + 0.5) / R - 0.5  # -0.5 (top) .. 0.5 (bottom)
    rr = rows[None, :, None]
    is_wall = np.abs(rr) < half[:, None, :]
    # floor/ceiling darken towards the horizon (distance of a floor point ~ 0.5/|row|)
    fc_dist = 0.5 / np.maximum(np.abs(rows), 1e-3)
    fc_shade = np.exp(-FOG * fc_dist)[None, :, None, None]
    fc = np.where((rr < 0)[..., None], CEIL, FLOOR) * fc_shade  # (1, R, 1, 3)
    img = np.where(is_wall[..., None], color[:, None, :, :], fc)  # (n, R, R, 3)
    img = img.reshape(n, size, ss, size, ss, 3).mean((2, 4))
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
