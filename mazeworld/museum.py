"""The look of notebook 3: the block mazes of `world.py` rendered as a small museum. Cream brick
walls with framed paintings, a slate ceiling with round lamps, and a grey floor with pools of
lamp light. Pure NumPy; the ray casting is `world4.cast`.

Image coordinates have square pixels: x_img = lateral / (depth T), y_img = -height / (depth T),
both in [-1, 1], with T = tan(FOV / 2). The camera sits at height EYE, walls are 1 high.
"""
import numpy as np

from .world4 import T, EYE, cast, smoothstep, _hash

BRICK = np.array([0.90, 0.89, 0.86])
MORTAR = np.array([0.77, 0.76, 0.73])
CEIL = np.array([0.30, 0.31, 0.35])
FLOOR = np.array([0.50, 0.50, 0.52])
LAMP = np.array([1.00, 0.93, 0.74])
HAZE = np.array([0.40, 0.40, 0.43])
FOG = 0.16
FRAMES = np.array([
    [0.55, 0.40, 0.18],  # gilt
    [0.36, 0.22, 0.14],  # walnut
    [0.12, 0.12, 0.13],  # black
    [0.62, 0.62, 0.60],  # pewter
    [0.20, 0.33, 0.33],  # teal
])
# muted painting palettes: (background, middle, accent)
PAINT = np.array([
    [[0.55, 0.66, 0.74], [0.38, 0.48, 0.32], [0.86, 0.74, 0.48]],  # landscape
    [[0.20, 0.17, 0.15], [0.62, 0.48, 0.38], [0.80, 0.70, 0.60]],  # portrait
    [[0.70, 0.52, 0.36], [0.30, 0.36, 0.48], [0.88, 0.82, 0.70]],  # dusk
    [[0.26, 0.34, 0.40], [0.66, 0.68, 0.62], [0.74, 0.36, 0.28]],  # sea
    [[0.78, 0.74, 0.62], [0.52, 0.30, 0.26], [0.22, 0.28, 0.24]],  # still life
    [[0.36, 0.42, 0.30], [0.72, 0.62, 0.40], [0.46, 0.22, 0.20]],  # forest
])
P_PAINTING = 0.45  # fraction of wall faces that carry a painting
LAMP_R = 0.11  # lamp radius in cells


def _bricks(u, v, seed):
    """Running-bond bricks, 5 courses of 3 per face, with mortar lines and per-brick jitter.
    Returns (shade, mortar weight)."""
    course = np.floor(v * 5)
    uu = u * 3 + 0.5 * (course % 2)
    brick = np.floor(uu)
    fu, fv = uu - brick, v * 5 - course
    jitter = 0.94 + 0.08 * _hash(seed, course, brick)
    m = np.maximum(1 - smoothstep(0.0, 0.05, np.minimum(fu, 1 - fu)),
                   1 - smoothstep(0.0, 0.07, np.minimum(fv, 1 - fv)))
    return jitter, m


def _painting(u, v, kind, h1, h2):
    """A small procedural painting on the canvas coordinates (u, v) in [0, 1]^2.
    kind picks the palette; h1, h2 in [0, 1) vary the composition."""
    pal = PAINT[kind]  # (..., 3, 3)
    bg, mid, acc = pal[..., 0, :], pal[..., 1, :], pal[..., 2, :]
    horizon = 0.45 + 0.25 * h1
    # background, with a lower band for the "ground" (soft edge)
    band = smoothstep(horizon - 0.04, horizon + 0.04, v)[..., None]
    img = bg * (1 - band) + mid * band
    # a soft blob (sun, face, vase) at a hashed position
    cu, cv = 0.3 + 0.4 * h2, 0.25 + 0.3 * h1
    r = np.hypot((u - cu) / 0.8, v - cv)
    blob = smoothstep(0.20, 0.12, r)[..., None]
    img = img * (1 - blob) + acc * blob
    # gentle vertical light falloff
    return img * (1.05 - 0.15 * v)[..., None]


def render(grid, poses, size=32, ss=None):
    """grid (Gy, Gx) uint8 (0 = free), poses (N, 3) -> (N, size, size, 3) uint8.
    ss x ss rays per pixel (antialiasing); by default about 128 rays across the image."""
    ss = ss or max(2, 128 // size)
    grid = np.asarray(grid)
    poses = np.asarray(poses, np.float64).reshape(-1, 3)
    out = [_render(grid, poses[i:i + 16], size, ss) for i in range(0, len(poses), 16)]
    return np.concatenate(out) if out else np.zeros((0, size, size, 3), np.uint8)


def _render(grid, poses, size, ss):
    n, R = len(poses), size * ss
    c = cast(grid, poses, R)
    dist = c["dist"]
    y = 2 * (np.arange(R) + 0.5) / R - 1  # image y, -1 top .. 1 bottom
    Y = y[None, :, None]
    top = ((1 - EYE) / (dist * T))[:, None, :]  # wall spans -top < y < bot
    bot = (EYE / (dist * T))[:, None, :]
    is_wall = (Y > -top) & (Y < bot)
    gy, gx = grid.shape

    # ---- walls
    u = c["u"][:, None, :]
    v = (Y + top) / (top + bot)  # 0 at the top of the wall, 1 at the floor
    mx, my = c["mx"][:, None, :], c["my"][:, None, :]
    # each face is identified by its cell and its normal; the maze's block colors seed the hash
    face = (c["nx"] + 2 * c["ny"])[:, None, :]
    seed = grid[c["my"], c["mx"]][:, None, :].astype(np.float64)
    jit, m = _bricks(u, v, mx * 7.0 + my * 13.0 + face)
    wall = (BRICK * jit[..., None]) * (1 - m[..., None]) + MORTAR * m[..., None]
    # the same face can be seen from either side of a corner: flip u so paintings are not mirrored
    flip = ((c["nx"] > 0) | (c["ny"] < 0))[:, None, :]
    uf = np.where(flip, 1 - u, u)
    hp = _hash(mx + 0.37 * face, my - 0.61 * face, seed, 3.3)
    has = hp < P_PAINTING
    kind = np.floor(_hash(mx, my, face, seed + 1.7) * len(PAINT)).astype(np.int64) % len(PAINT)
    fk = np.floor(_hash(mx, my, face, seed + 5.1) * len(FRAMES)).astype(np.int64) % len(FRAMES)
    wide = _hash(mx, my, face, seed + 9.3) < 0.5
    hw = np.where(wide, 0.34, 0.24)  # canvas half-width along the face
    u0, v0, hh = 0.5, 0.42, np.where(wide, 0.17, 0.21)
    du, dv = np.abs(uf - u0), np.abs(v - v0)
    fw = 0.045
    in_frame = has & (du < hw + fw) & (dv < hh + fw)
    in_canvas = has & (du < hw) & (dv < hh)
    cu = (uf - (u0 - hw)) / (2 * hw)
    cv = (v - (v0 - hh)) / (2 * hh)
    canvas = _painting(cu, cv, kind, _hash(mx, my, face, seed + 2.2), _hash(mx, my, face, seed + 4.4))
    bevel = 1 + 0.25 * (1 - smoothstep(0, fw, np.minimum(hw + fw - du, hh + fw - dv)))
    frame = FRAMES[fk] * bevel[..., None]
    wall = np.where(in_canvas[..., None], canvas, np.where(in_frame[..., None], frame, wall))
    # a soft shadow under the frame, light from the ceiling lamps, a skirting shadow at the floor
    shadow = has & ~in_frame & (du < hw + fw) & (v > v0 + hh + fw) & (v < v0 + hh + fw + 0.04)
    light = 0.82 + 0.18 * (1 - v)
    light = light * np.where(c["side"] == 0, 1.0, 0.86)[:, None, :]
    light = light * (1 - 0.18 * shadow) * (1 - 0.30 * smoothstep(0.90, 1.0, v))
    # dark seams where two blocks meet (corners read better)
    light = light * (1 - 0.30 * (1 - smoothstep(0.0, 0.025, np.minimum(u, 1 - u))))
    wall = wall * light[..., None]
    fwall = np.exp(-FOG * dist)[:, None, :, None]
    wall = wall * fwall + HAZE * (1 - fwall)

    # ---- floor and ceiling: the point seen at row y is at depth h / (|y| T)
    up = Y < 0
    hgt = np.where(up, 1 - EYE, EYE)
    d = hgt / (np.maximum(np.abs(Y), 1e-4) * T)  # (1, R, 1)
    fx = poses[:, 0, None, None] + d * c["rdx"][:, None, :]
    fy = poses[:, 1, None, None] + d * c["rdy"][:, None, :]
    cx, cy = np.floor(fx), np.floor(fy)
    ux, uy = fx - cx, fy - cy
    r_lamp = np.hypot(ux - 0.5, uy - 0.5)  # distance to the center of the cell = the lamp
    # ceiling: slate with a round lamp and its halo in every cell
    lamp = smoothstep(LAMP_R, LAMP_R * 0.8, r_lamp)
    halo = np.exp(-(r_lamp / 0.32) ** 2) * 0.35
    ceil = CEIL * (1 + halo)[..., None]
    ceil = ceil * (1 - lamp[..., None]) + LAMP * lamp[..., None]
    rim = smoothstep(LAMP_R * 1.35, LAMP_R * 1.1, r_lamp) * (1 - lamp)
    ceil = ceil * (1 - 0.35 * rim)[..., None]
    # floor: grey with large faint tiles, a pool of light under each lamp, darker next to walls
    tx, ty = np.floor(fx * 1.0), np.floor(fy * 1.0)
    tile = 0.97 + 0.05 * _hash(tx, ty, 11.0, 0.0)
    pool = 1 + 0.22 * np.exp(-(r_lamp / 0.38) ** 2)
    ix, iy = cx.astype(np.int64), cy.astype(np.int64)

    def wall_at(ox, oy):
        return grid[np.clip(iy + oy, 0, gy - 1), np.clip(ix + ox, 0, gx - 1)] > 0

    ao = np.ones_like(fx)
    for ox, oy, dd in ((-1, 0, ux), (1, 0, 1 - ux), (0, -1, uy), (0, 1, 1 - uy)):
        ao *= 1 - 0.35 * wall_at(ox, oy) * np.exp(-dd / 0.12)
    floor = FLOOR * (tile * pool * ao)[..., None]
    fc = np.where(up[..., None], ceil, floor)
    ff = np.exp(-FOG * d)[..., None]
    fc = fc * ff + HAZE * (1 - ff)

    img = np.where(is_wall[..., None], wall, fc)
    img = img.reshape(n, R // ss, ss, R // ss, ss, 3).mean((2, 4))
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
