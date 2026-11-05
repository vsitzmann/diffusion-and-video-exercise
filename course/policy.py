"""Notebook 4: one next-frame video model that is a world model (given actions) and a
goal-conditioned policy (given a map, no actions), guidance, an inverse dynamics model and
closed-loop navigation.

The denoiser follows DIAMOND / GameNGen: a 2D U-Net denoises frame t from noise, given the
previous C frames stacked in its input channels. Every context frame has its own noise level
(diffusion forcing; level 0 = the frame is missing). Noise levels k as in notebook 3:
x^k = k x + (1 - k) eps, k = 1 clean. The network predicts the clean frame (x-prediction) and is
sampled with deterministic Euler steps.
"""
import base64
import copy
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mazeworld import world as mw
from mazeworld import world4 as w4

ROOT = Path(__file__).resolve().parent.parent
N_ACT = 4  # forward, left, right, "no action given"
NULL_ACT = 3
MAP_CH, MAP_N = 4, 17
EGO_R = 4.0  # the egocentric map covers +-4 cells around the agent
C = 4  # context frames
ARROWS = "↑↶↷"


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------- network

def gn(c):
    return nn.GroupNorm(next(g for g in (32, 16, 8, 4, 2, 1) if c % g == 0 and c // g >= 4 or g == 1), c)


def fourier(k, dim, max_freq=1000.0):
    f = torch.exp(torch.linspace(0, math.log(max_freq), dim // 2, device=k.device))
    a = k[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.ada = nn.Linear(emb, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.ada(e)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + s) + b
        return self.skip(x) + self.c2(F.silu(h))


class Attn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n, self.qkv, self.out, self.heads = gn(ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1), heads
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, Ch, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, Ch // self.heads, H * W).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v)
        return x + self.out(o.transpose(-1, -2).reshape(B, Ch, H, W))


class MapEncoder(nn.Module):
    def __init__(self, emb, ch=64, size=MAP_N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(MAP_CH, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(2 * ch, 2 * ch, 3, padding=1), nn.SiLU())
        s = (size + 1) // 2
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(2 * ch * s * s, emb), nn.SiLU(), nn.Linear(emb, emb))

    def forward(self, m):
        return self.fc(self.net(m))


class Denoiser(nn.Module):
    """Predicts the clean frame from (noisy frame at level k, C context frames at levels k_ctx,
    C actions, egocentric map or "no map", frames since the map was taken). All conditioning
    except the context frames enters through adaptive group norm."""

    def __init__(self, res=16, chs=(32, 64, 128), nb=1, attn_res=(4,), C=C, emb=128, map_ch=32, max_offset=16):
        super().__init__()
        self.res, self.C = res, C
        self.cfg = dict(res=res, chs=tuple(chs), nb=nb, attn_res=tuple(attn_res), C=C, emb=emb,
                        map_ch=map_ch, max_offset=max_offset)
        self.k_mlp = nn.Sequential(nn.Linear(64, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.kc_mlp = nn.Sequential(nn.Linear(C * 32, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.act = nn.Embedding(N_ACT, 64)
        self.act_mlp = nn.Sequential(nn.Linear(C * 64, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.map = MapEncoder(emb, map_ch)
        self.null_map = nn.Parameter(torch.zeros(emb))
        self.offset = nn.Embedding(max_offset + 1, emb)
        self.inp = nn.Conv2d(3 + 4 * C, chs[0], 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        skips, ch, r = [chs[0]], chs[0], res
        for i, c in enumerate(chs):
            for _ in range(nb):
                blk = nn.ModuleList([ResBlock(ch, c, emb)])
                if r in attn_res:
                    blk.append(Attn(c))
                self.down.append(blk)
                ch = c
                skips.append(ch)
            if i < len(chs) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                skips.append(ch)
                r //= 2
        self.mid = nn.ModuleList([ResBlock(ch, ch, emb), Attn(ch), ResBlock(ch, ch, emb)])
        for i, c in reversed(list(enumerate(chs))):
            for _ in range(nb + 1):
                blk = nn.ModuleList([ResBlock(ch + skips.pop(), c, emb)])
                if r in attn_res:
                    blk.append(Attn(c))
                self.up.append(blk)
                ch = c
            if i > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1)]))
                r *= 2
        self.out_norm = gn(ch)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def cond(self, k, k_ctx, actions, map_in, offset):
        B = k.shape[0]
        e = self.k_mlp(fourier(k, 64)) + self.kc_mlp(fourier(k_ctx, 32).reshape(B, -1))
        e = e + self.act_mlp(self.act(actions).reshape(B, -1))
        if map_in is None:
            m = self.null_map.expand(B, -1)
        else:  # rows of map_in that are NaN mean "no map"
            has = ~torch.isnan(map_in).flatten(1).any(1)
            m = self.map(torch.nan_to_num(map_in, nan=0.0))
            m = torch.where(has[:, None], m, self.null_map.expand(B, -1))
        return e + m + self.offset(offset)

    def forward(self, x_noisy, k, ctx, k_ctx, actions, map_in=None, offset=None):
        """x_noisy (B, 3, r, r) at level k (B,); ctx (B, C, 3, r, r) already noised to k_ctx (B, C);
        actions (B, C) long; map_in (B, 4, 17, 17) or None; offset (B,) long."""
        B = k.shape[0]
        if offset is None:
            offset = torch.zeros(B, dtype=torch.long, device=k.device)
        e = F.silu(self.cond(k, k_ctx, actions, map_in, offset))
        lv = k_ctx[:, :, None, None].expand(-1, -1, *x_noisy.shape[-2:])
        h = self.inp(torch.cat([x_noisy, ctx.flatten(1, 2), lv], 1))
        hs = [h]
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, e)
                if len(blk) > 1:
                    h = blk[1](h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = blk[0](torch.cat([h, hs.pop()], 1), e)
                if len(blk) > 1:
                    h = blk[1](h)
            else:
                h = blk[1](blk[0](h))
        return self.out(F.silu(self.out_norm(h)))


CONFIGS = {
    16: dict(res=16, chs=(32, 64, 128), nb=1, attn_res=(4,), emb=128, map_ch=32),
    64: dict(res=64, chs=(48, 96, 160, 256), nb=1, attn_res=(16, 8), emb=256, map_ch=64),
}


class IDM(nn.Module):
    """Inverse dynamics: (frame t, frame t+1) -> logits over the 3 actions."""

    def __init__(self, res=16, ch=32):
        super().__init__()
        layers, c, r = [nn.Conv2d(6, ch, 3, padding=1), nn.SiLU()], ch, res
        while r > 4:
            c2 = 2 * c if c < 128 else c
            layers += [nn.Conv2d(c, c2, 3, stride=2, padding=1), nn.GroupNorm(8, c2), nn.SiLU()]
            c, r = c2, r // 2
        self.net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(c * r * r, 128), nn.SiLU(), nn.Linear(128, 3))
        self.res = res

    def forward(self, a, b):
        return self.net(torch.cat([a, b], 1))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load_video(res, dev=None):
    ck = torch.load(ROOT / "checkpoints" / f"nb4_video{res}.pt", map_location="cpu")
    m = Denoiser(**ck["cfg"])
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return m.to(dev or device()).eval()


def load_idm(res, dev=None):
    ck = torch.load(ROOT / "checkpoints" / f"nb4_idm{res}.pt", map_location="cpu")
    m = IDM(res)
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return m.to(dev or device()).eval()


# ------------------------------------------------------------------------------- frames and maps

def to_float(frames_u8):
    """(..., r, r, 3) uint8 (numpy or torch) -> (..., 3, r, r) float in [-1, 1]."""
    if isinstance(frames_u8, np.ndarray):
        frames_u8 = torch.from_numpy(np.ascontiguousarray(frames_u8))
    return frames_u8.float().div(127.5).sub(1).movedim(-1, -3)


def to_u8(x):
    """(..., 3, r, r) float -> (..., r, r, 3) uint8 numpy."""
    return x.detach().clamp(-1, 1).add(1).mul(127.5).round().byte().movedim(-3, -1).cpu().numpy()


def quantize(x):
    """Round a generated frame to 8 bits, like a real image."""
    return x.clamp(-1, 1).add(1).mul(127.5).round().div(127.5).sub(1)


def make_map(grid, pose, goal, dist):
    """Egocentric map (CueNav style): a 17x17 crop of +-4 cells around the agent, rotated so that
    "ahead" is up. grid (B,13,13) uint8, pose (B,3), goal (B,2), dist (B,13,13) uint8 (BFS distance
    to the goal, 255 = wall) -> (B, 4, 17, 17): walls; distance to the goal relative to the agent's
    own cell (clipped to +-4, /4); distance / 40; a blob at the goal."""
    dev = pose.device
    B = pose.shape[0]
    gy, gx = grid.shape[-2:]
    s = torch.linspace(-EGO_R, EGO_R, MAP_N, device=dev)
    fwd, right = -s[None, :, None], s[None, None, :]
    px, py, a = pose[:, 0, None, None], pose[:, 1, None, None], pose[:, 2, None, None]
    ca, sa = torch.cos(a), torch.sin(a)
    wx, wy = px + fwd * ca - right * sa, py + fwd * sa + right * ca
    cx, cy = torch.floor(wx).long(), torch.floor(wy).long()
    inside = (cx >= 0) & (cx < gx) & (cy >= 0) & (cy < gy)
    b = torch.arange(B, device=dev)[:, None, None]
    cxc, cyc = cx.clamp(0, gx - 1), cy.clamp(0, gy - 1)
    wall = ~inside | (grid[b, cyc, cxc] > 0)
    dc = dist[b, cyc, cxc].float()
    d0 = dist[torch.arange(B, device=dev), py[:, 0, 0].floor().long().clamp(0, gy - 1),
              px[:, 0, 0].floor().long().clamp(0, gx - 1)].float()
    dd = ((dc - d0[:, None, None]).clamp(-4, 4) / 4).masked_fill(wall, 0)
    dn = (dc / 40).masked_fill(wall, 0)
    gb = torch.exp(-((wx - goal[:, 0, None, None]) ** 2 + (wy - goal[:, 1, None, None]) ** 2) / (2 * 0.5 ** 2))
    return torch.stack([wall.float(), dd, dn, gb], 1)


def env_map(envs, poses, dev):
    """Egocentric maps for a list of test environments at the given poses."""
    t = lambda xs, dt=torch.float32: torch.as_tensor(np.stack(xs), dtype=dt, device=dev)
    return make_map(t([e["grid"] for e in envs], torch.uint8), t(poses), t([e["goal"] for e in envs]),
                    t([e["dist"] for e in envs], torch.uint8))


def map_rgb(m):
    """(4, 17, 17) map -> (17, 17, 3) float RGB for display: walls dark; floor from light (closer to
    the goal than the agent's own cell) to dark blue (farther away); the goal in pink."""
    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    wall, dd, gb = m[0] > 0.5, m[1], np.clip(m[3], 0, 1)[..., None]
    near, far = np.array([0.97, 0.91, 0.78]), np.array([0.25, 0.32, 0.45])
    floor = near + (far - near) * ((dd + 1) / 2)[..., None]
    img = np.where(wall[..., None], np.array([0.12, 0.11, 0.10]), floor)
    return np.clip(img * (1 - gb) + np.array([1.0, 0.28, 0.62]) * gb, 0, 1)


# ------------------------------------------------------------------------------ guided sampling

def renoise(x, k, g=None):
    """x (B, C, 3, r, r), k (B, C): x^k = k x + (1 - k) eps."""
    k = k[:, :, None, None, None]
    return k * x + (1 - k) * torch.randn(x.shape, generator=g, device=x.device)


@torch.no_grad()
def predict(model, x, k, ctx, k_ctx, acts, map_in, offset, variants=(), guide=None):
    """Clean-frame prediction with guidance. variants: list of (w, overrides of ctx / k_ctx /
    acts / map_in / offset) that define an "unconditional" branch each. With one variant this is
    guide(x_cond, x_uncond, w); with several, the guidance terms add up. All branches run as one
    batch."""
    B = x.shape[0]
    base = dict(ctx=ctx, k_ctx=k_ctx, acts=acts, map_in=map_in, offset=offset)
    rows = [base] + [{**base, **ov} for _, ov in variants]
    n = len(rows)
    maps = None
    if any(r["map_in"] is not None for r in rows):
        nan = torch.full((B, MAP_CH, MAP_N, MAP_N), float("nan"), device=x.device)
        maps = torch.cat([r["map_in"] if r["map_in"] is not None else nan for r in rows])
    cat = lambda key: torch.cat([r[key] for r in rows])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        out = model(x.repeat(n, 1, 1, 1), k.repeat(n), cat("ctx"), cat("k_ctx"), cat("acts"), maps, cat("offset"))
    out = out.float().chunk(n)
    res = out[0]
    for (w, _), o in zip(variants, out[1:]):
        res = res + guide(out[0], o, w) - out[0]
    return res


@torch.no_grad()
def sample_frame(model, ctx, k_ctx, acts, map_in=None, offset=None, n_steps=10, g=None, variants=(), guide=None):
    """Euler sampling of one frame from pure noise (k = 0) to clean (k = 1), with x-prediction:
    the velocity is (x_hat - x) / (1 - k)."""
    B, _, ch, H, W = ctx.shape
    dev = ctx.device
    if offset is None:
        offset = torch.zeros(B, dtype=torch.long, device=dev)
    x = torch.randn(B, ch, H, W, generator=g, device=dev)
    ks = torch.linspace(0, 1, n_steps + 1, device=dev)
    for j in range(n_steps):
        xh = predict(model, x, ks[j].expand(B), ctx, k_ctx, acts, map_in, offset, variants, guide)
        x = x + (ks[j + 1] - ks[j]) * (xh - x) / (1 - ks[j])
    return x


@torch.no_grad()
def generate(model, ctx, T, k_ctx=None, actions=None, map_in=None, w_map=1.0, w_hist=1.0, hist_k=0.0,
             n_steps=10, k_gen=0.96, guide=None, seed=0):
    """Autoregressive generation of T frames after the C context frames ctx (B, C, 3, r, r).
    k_ctx (B, C): noise levels at which the context is shown (default 1 = clean; 0 = missing).
    actions (B, C + T - 1) long or None ("no actions"); actions[:, j] leads from frame j to j + 1.
    map_in (B, 4, 17, 17): goal map, taken at the last context frame. Guidance: w_map (the
    unconditional branch has no map), w_hist (the unconditional branch has the history re-noised
    to level hist_k; 0 = dropped). Generated frames are rounded to 8 bits and fed back at level
    k_gen. Returns (B, T, 3, r, r)."""
    dev = ctx.device
    B = ctx.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    frames = list(ctx.unbind(1))
    levels = list((torch.ones(B, C, device=dev) if k_ctx is None else k_ctx.to(dev)).unbind(1))
    zero = torch.zeros(B, dtype=torch.long, device=dev)
    for t in range(C, C + T):
        c = torch.stack(frames[-C:], 1)
        kc = torch.stack(levels[-C:], 1)
        acts = actions[:, t - C:t].long() if actions is not None else torch.full((B, C), NULL_ACT, device=dev)
        variants = []
        if w_hist != 1.0:
            kh = torch.minimum(kc, torch.full_like(kc, hist_k))
            variants.append((w_hist, dict(ctx=renoise(c, kh, g), k_ctx=kh)))
        off = None
        if map_in is not None:
            off = torch.full((B,), min(t - C + 1, 16), dtype=torch.long, device=dev)
            if w_map != 1.0:
                variants.append((w_map, dict(map_in=None, offset=zero)))
        x = sample_frame(model, renoise(c, kc, g), kc, acts, map_in, off, n_steps, g, variants, guide)
        frames.append(quantize(x))
        levels.append(torch.full((B,), k_gen, device=dev))
    return torch.stack(frames[C:], 1)


# ------------------------------------------------------------------------------------ training

def to_device(data, dev):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in data.items()}


def video_batch(D, B, drop, g, H=8, p_policy=0.7):
    """A training batch. The conditioning is dropped with your `drop` function (Exercise 1):
    actions (expert episodes: p_policy, explorer: 0.15), the map (expert: 0.15; explorer episodes
    never get one, so "no map" means "explore") and the whole history (0.1)."""
    dev = D["frames"].device
    N, L = D["actions"].shape
    r = lambda *s: torch.rand(*s, generator=g, device=dev)
    i = torch.randint(N, (B,), generator=g, device=dev)
    t = torch.randint(C, L, (B,), generator=g, device=dev)
    idx = t[:, None] + torch.arange(-C, 1, device=dev)
    fr = to_float(D["frames"][i[:, None], idx])
    x, ctx = fr[:, -1], fr[:, :-1]
    is_exp = D["expert"][i]
    acts = drop(D["actions"][i[:, None], idx[:, :-1]].long(), torch.tensor(NULL_ACT, device=dev),
                torch.where(is_exp, p_policy, 0.15), g)
    # the map is taken o = 1..H frames before the target frame
    o = torch.randint(1, H + 1, (B,), generator=g, device=dev)
    t0 = (t - o).clamp(min=0)
    m = make_map(D["grids"][i], D["poses"][i, t0], D["goals"][i, t0], D["dist"][i, t0])
    m = drop(m, torch.tensor(float("nan"), device=dev), torch.where(is_exp, 0.15, 1.0), g)
    has_map = ~torch.isnan(m).flatten(1).any(1)
    o = torch.where(has_map, t - t0, torch.zeros_like(o))
    # noise levels: the target frame k biased toward noise (logit-normal), the context
    # mostly almost clean, sometimes anything; single frames and the whole history dropped
    k = torch.sigmoid(torch.randn(B, generator=g, device=dev) - 0.8)
    kc = torch.where(r(B, 1) < 0.6, 0.7 + 0.3 * r(B, C), r(B, C))
    kc = torch.where(r(B, C) < 0.1, torch.zeros_like(kc), kc)
    kc = drop(kc, torch.tensor(0.0, device=dev), 0.1, g)
    eps = torch.randn(x.shape, generator=g, device=dev)
    xk = k[:, None, None, None] * x + (1 - k[:, None, None, None]) * eps
    return x, xk, k, renoise(ctx, kc, g), kc, acts, m, o


def train_video(model, data, drop, steps=5000, bs=64, lr=2e-4, ema_decay=0.999, seed=0, log=None):
    """Train the denoiser with the x-loss ||x_hat - x||^2. Returns the EMA copy of the model and the
    loss curve."""
    dev = next(model.parameters()).device
    D = to_device(data, dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    model.train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    warm = min(1000, steps // 5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (
        0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)) * 0.9 + 0.1))
    losses, t0 = [], time.time()
    for step in range(steps):
        x, xk, k, ctx, kc, acts, m, o = video_batch(D, bs, drop, g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            xh = model(xk, k, ctx, kc, acts, m, o)
        loss = ((xh.float() - x) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        losses.append(loss.item())
        if log and (step % 25 == 0 or step == steps - 1):
            log(step, float(np.mean(losses[-100:])), time.time() - t0)
    model.eval()
    return ema, losses


def idm_pairs(Fr, A, B, g):
    dev = Fr.device
    i = torch.randint(len(A), (B,), generator=g, device=dev)
    t = torch.randint(A.shape[1] - 1, (B,), generator=g, device=dev)
    return to_float(Fr[i, t]), to_float(Fr[i, t + 1]), A[i, t].long()


def train_idm(idm, data, loss_fn, steps=1500, bs=128, lr=1e-3, noise=0.05, seed=0, log=None):
    """Train the inverse dynamics model on real frame pairs, with a little pixel noise so that it
    also reads generated frames. loss_fn(idm, frame_a, frame_b, actions) is Exercise 3."""
    dev = next(idm.parameters()).device
    Fr = torch.from_numpy(data["frames"]).to(dev)
    A = torch.from_numpy(data["actions"]).to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    opt = torch.optim.AdamW(idm.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    idm.train()
    losses, t0 = [], time.time()
    for step in range(steps):
        a, b, y = idm_pairs(Fr, A, bs, g)
        a = a + noise * torch.randn(a.shape, generator=g, device=dev)
        b = b + noise * torch.randn(b.shape, generator=g, device=dev)
        loss = loss_fn(idm, a, b, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log and (step % 25 == 0 or step == steps - 1):
            log(step, float(np.mean(losses[-50:])), time.time() - t0)
    idm.eval()
    return losses


@torch.no_grad()
def idm_accuracy(idm, data, n=2048, seed=1):
    dev = next(idm.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    a, b, y = idm_pairs(torch.from_numpy(data["frames"]).to(dev), torch.from_numpy(data["actions"]).to(dev), n, g)
    return (idm(a, b).argmax(1) == y).float().mean().item()


# ------------------------------------------------------------------------------- evaluation

def psnr(a, b):
    """Peak signal-to-noise ratio in dB for frames in [-1, 1], per frame."""
    return 10 * torch.log10(4 / ((a - b) ** 2).mean((-3, -2, -1)).clamp(min=1e-10))


def turn_onsets(data, H=8, per_class=12, seed=0):
    """States of expert episodes where the agent has just gone forward and, over the next H steps
    (with the same goal), turns left, goes straight or turns right. Returns a list of
    (episode, t, class) with class 0 = left, 1 = straight, 2 = right."""
    cands = {0: [], 1: [], 2: []}
    A, G = data["actions"], data["goals"]
    for i in np.nonzero(data["expert"])[0]:
        for t in range(C, A.shape[1] - H, 3):
            a = A[i, t:t + H]
            if A[i, t - 1] != mw.FORWARD or not np.allclose(G[i, t], G[i, t + H]):
                continue
            turn = (a == mw.RIGHT).sum() - (a == mw.LEFT).sum()
            if turn <= -4:
                cands[0].append((i, t))
            elif turn >= 4:
                cands[2].append((i, t))
            elif (a == mw.FORWARD).all():
                cands[1].append((i, t))
    rng = np.random.default_rng(seed)
    out = []
    for c in (0, 1, 2):
        sel = rng.choice(len(cands[c]), min(per_class, len(cands[c])), replace=False)
        out += [(*cands[c][s], c) for s in sel]
    return out


def state_inputs(data, states, dev):
    """Context frames (S, C, 3, r, r) ending at frame t, and the map at t."""
    ii = np.array([s[0] for s in states])
    tt = np.array([s[1] for s in states])
    ctx = to_float(np.stack([data["frames"][i, t - C + 1:t + 1] for i, t in zip(ii, tt)])).to(dev)
    tt_ = lambda x, dt=torch.float32: torch.as_tensor(np.stack(x), dtype=dt, device=dev)
    m = make_map(tt_([data["grids"][i] for i in ii], torch.uint8), tt_([data["poses"][i, t] for i, t in zip(ii, tt)]),
                 tt_([data["goals"][i, t] for i, t in zip(ii, tt)]), tt_([data["dist"][i, t] for i, t in zip(ii, tt)], torch.uint8))
    return ctx, m


@torch.no_grad()
def net_turn_class(idm, ctx_last, future):
    """Read the actions of imagined futures (B, H, 3, r, r) with the IDM and classify each by its
    net turn: 0 = left, 1 = straight, 2 = right."""
    seq = torch.cat([ctx_last[:, None], future], 1)
    a = idm(seq[:, :-1].flatten(0, 1), seq[:, 1:].flatten(0, 1)).argmax(1).view(future.shape[:2])
    turn = (a == mw.RIGHT).sum(-1) - (a == mw.LEFT).sum(-1)
    return torch.where(turn <= -3, 0, torch.where(turn >= 3, 2, 1))


# ------------------------------------------------------------------------------- closed loop

def imagine(model, frames, k_ctx, map_in, w, n_steps, H, guide, seed):
    """The video policy: imagine the next H frames from the last C frames and the goal map, with
    no actions, goal guidance weight w."""
    return generate(model, frames, H, k_ctx=k_ctx, map_in=map_in, w_map=w, n_steps=n_steps, guide=guide, seed=seed)


@torch.no_grad()
def navigate(envs, model, idm, read_actions, w=3.0, n_steps=1, H=8, n_exec=4, max_steps=400, guide=None,
             seed=0, record=4, progress=None):
    """Closed-loop navigation in all envs at once: every n_exec steps, imagine H frames, read the H
    actions off them with read_actions(idm, frames) (Exercise 4), execute the first n_exec in the
    real maze, repeat. Returns one dict per env with poses, actions, success and, for the first
    `record` envs, the plans (imagined frames and actions)."""
    dev = next(model.parameters()).device
    res = model.res
    runs = []
    for e in envs:
        f0 = w4.render(e["grid"], e["start"][None], e["goal"], size=res)[0]
        runs.append(dict(env=e, pose=e["start"].copy(), poses=[e["start"].copy()], actions=[], frames=[f0],
                         plans=[], done=False, success=False))
    it = 0
    while not all(r["done"] for r in runs):
        act = [r for r in runs if not r["done"]]
        ctx, kc = [], []
        for r in act:
            f = r["frames"][-C:]
            pad = C - len(f)
            ctx.append(np.concatenate([np.zeros((pad, res, res, 3), np.uint8), np.stack(f)]) if pad else np.stack(f))
            kc.append([0.0] * pad + [1.0] * len(f))  # missing frames at the start: level 0
        ctx = to_float(np.stack(ctx)).to(dev)
        kc = torch.tensor(kc, device=dev)
        m = env_map([r["env"] for r in act], [r["pose"] for r in act], dev)
        fut = imagine(model, ctx, kc, m, w, n_steps, H, guide, seed + it)
        plan = read_actions(idm, torch.cat([ctx[:, -1:], fut], 1)).cpu().numpy()
        fut_u8 = to_u8(fut)
        for b, r in enumerate(act):
            if len(r["plans"]) < 200 and runs.index(r) < record:
                r["plans"].append(dict(t=len(r["actions"]), frames=fut_u8[b], actions=plan[b].tolist()))
            e = r["env"]
            for a in plan[b][:n_exec]:
                r["pose"] = mw.step(e["grid"], r["pose"], int(a))
                r["poses"].append(r["pose"])
                r["actions"].append(int(a))
                r["frames"].append(w4.render(e["grid"], r["pose"][None], e["goal"], size=res)[0])
                if np.hypot(*(r["pose"][:2] - e["goal"])) < w4.STOP:
                    r["done"] = r["success"] = True
                    break
                if len(r["actions"]) >= max_steps:
                    r["done"] = True
                    break
        it += 1
        if progress:
            progress(sum(r["done"] for r in runs), len(runs), max(len(r["actions"]) for r in runs))
    for r in runs:
        r["poses"] = np.array(r["poses"])
        r["frames"] = np.stack(r["frames"])
    return runs


# ------------------------------------------------------------------------------------ viewer

def _sheet(frames, cols=16):
    n, h, w, _ = frames.shape
    rows = -(-n // cols)
    s = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, f in enumerate(frames):
        s[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = f
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(s).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def viewer_html(env, actions, plans, H=8, n_exec=4, strip_title="Imagined by the video model",
                note="(plan made at frame {t}; arrows: the actions the IDM reads off; the first {exec} are executed)",
                caption=""):
    """Interactive player: the agent's view (rendered at 64x64), the map with the view cone, the
    path so far and the planned path, the plan's frames, and a 3/4 view of the maze with the
    camera frustum. plans: list of dict(t, frames (H, r, r, 3) uint8, actions (H,))."""
    poses = [env["start"]]
    for a in actions:
        poses.append(mw.step(env["grid"], poses[-1], int(a)))
    frames = w4.render(env["grid"], np.array(poses), env["goal"], size=64)
    imag, pl = [], []
    for p in plans:
        pl.append(dict(t=int(p["t"]), actions="".join("FLR"[int(a)] for a in p["actions"]), idx=len(imag)))
        imag.extend(list(p["frames"]))
    imag = np.stack(imag) if imag else np.zeros((1, 16, 16, 3), np.uint8)
    data = dict(grid=["".join(str(v) for v in r) for r in env["grid"]], start=[float(v) for v in env["start"]],
                goal=[float(v) for v in env["goal"]], actions="".join("FLR"[int(a)] for a in actions),
                n=len(frames), cols=16, plans=pl, H=H, exec=n_exec, isize=int(imag.shape[1]), icols=16,
                note=note, caption=caption)
    html = (ROOT / "course" / "assets" / "viewer4.html").read_text()
    html = html.replace("DATA_JSON", json.dumps(data, separators=(",", ":"))).replace("START_T", "0")
    html = html.replace("STRIP_TITLE", strip_title)
    return html.replace("ISHEET_SRC", _sheet(imag)).replace("SHEET_SRC", _sheet(frames))


def show_viewer(html, height=1130):
    import marimo as mo
    return mo.iframe(html, height=f"{height}px")


def expert_viewer(env, H=8, every=4):
    """The expert driving to the goal; the strip shows the true next H frames."""
    poses, actions = w4.expert_rollout(env)
    frames = w4.render(env["grid"], poses, env["goal"], size=64)
    plans = []
    for t in range(0, len(actions), every):
        fut = frames[t + 1:t + 1 + H]
        if len(fut) < H:
            fut = np.concatenate([fut, np.repeat(frames[-1:], H - len(fut), 0)])
        acts = list(actions[t:t + H]) + [0] * (H - len(actions[t:t + H]))
        plans.append(dict(t=t, frames=fut, actions=acts))
    return viewer_html(env, actions, plans, H, every, strip_title="The next 8 frames (real)",
                       note="(arrows: the expert's actions)", caption=f" · the expert needs {len(actions)} steps")


def reference():
    """Results of the pretrained models, measured on a GPU (64 test mazes)."""
    return json.loads((ROOT / "course" / "assets" / "nb4_reference.json").read_text())
