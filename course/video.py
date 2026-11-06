"""Tiny action-conditioned video diffusion model: a conv-RNN that reads past frames, and a small
conv U-Net that denoises the next frame. Used by 03_video_world_model.py.

Noise levels follow the lecture: a frame at noise level k is  x^k = (1 - k) * x + k * eps,
so k = 0 is the clean frame and k = 1 is pure noise. Every frame t has its own level.

Indexing: frames x_0 .. x_{T-1}; action a_t takes x_t to x_{t+1}.
    h_t = GRU(h_{t-1}, Enc(context version of x_t, its level), a_t)
    x_hat_{t+1} = Den(noisy x_{t+1}, its level | h_t, context version of x_t)
The denoiser sees the previous frame directly (most of the next frame is a shifted copy of it)
and everything older through the RNN state h. Frame 0 is denoised from a learned initial
state and an all-zero "previous frame" at level 1 (pure noise).
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 3


def to_tensor(frames):
    """uint8 (..., H, W, 3) -> float (..., 3, H, W) in [-1, 1]."""
    x = torch.as_tensor(frames).float().div(127.5).sub(1)
    return x.movedim(-1, -3)


def to_uint8(x):
    """float (..., 3, H, W) in [-1, 1] -> uint8 (..., H, W, 3)."""
    return torch.nan_to_num(x, nan=0.0).clamp(-1, 1).add(1).mul(127.5).round().byte().movedim(-3, -1).numpy()


def net_level(k):
    """The level as the network reads it. The shipped checkpoints were trained when this code
    used the opposite convention (k = 1 clean, k = 0 pure noise), so the network gets 1 - k.
    This is the only place where the convention is converted."""
    return 1 - k


class LevelEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(32, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, k):  # k: (N,) noise levels in [0, 1]
        f = torch.exp(torch.linspace(0, math.log(200), 16, device=k.device))
        a = net_level(k)[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class ResBlock(nn.Module):
    """3x3 conv block with a FiLM (scale, shift) from the noise-level embedding."""

    def __init__(self, ch, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, ch), nn.Conv2d(ch, ch, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, ch), nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(emb, 2 * ch)

    def forward(self, x, e):
        s, b = self.film(e)[:, :, None, None].chunk(2, 1)
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h) * (1 + s) + b
        return x + self.c2(F.silu(h))


class ConvGRU(nn.Module):
    def __init__(self, cin, ch):
        super().__init__()
        self.gates = nn.Conv2d(cin + ch, 2 * ch, 3, padding=1)
        self.cand = nn.Conv2d(cin + ch, ch, 3, padding=1)

    def forward(self, x, h):
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - u) * h + u * c


# The pretrained model of notebook 3: the same architecture and training code, wider, at 32x32.
BIG = dict(ch=48, hid=96, emb=128, res=32)


class VideoDenoiser(nn.Module):
    """Frames are res x res; the RNN state and the U-Net's middle level are res/2 x res/2."""

    def __init__(self, ch=16, hid=32, emb=64, res=16):
        super().__init__()
        self.hid, self.res = hid, res
        self.level = LevelEmbed(emb)
        # context encoder: res x res frame -> res/2 x res/2 features
        self.enc_in = nn.Conv2d(3, ch, 3, padding=1)
        self.enc_res = ResBlock(ch, emb)
        self.enc_down = nn.Conv2d(ch, hid, 3, stride=2, padding=1)
        self.action = nn.Embedding(N_ACTIONS, hid)
        self.gru = ConvGRU(hid, hid)
        self.h0 = nn.Parameter(torch.zeros(1, hid, res // 2, res // 2))
        # denoiser U-Net: res -> res/2 (+ h) -> res/4 -> back up (layer names are for res = 16)
        self.d_in = nn.Conv2d(7, ch, 3, padding=1)  # noisy frame + previous frame + its level
        self.d16 = ResBlock(ch, emb)
        self.d_down8 = nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1)
        self.d_mix8 = nn.Conv2d(2 * ch + hid, 2 * ch, 1)
        self.d8 = ResBlock(2 * ch, emb)
        self.d_down4 = nn.Conv2d(2 * ch, 4 * ch, 3, stride=2, padding=1)
        self.d4a, self.d4b = ResBlock(4 * ch, emb), ResBlock(4 * ch, emb)
        self.d_up8 = nn.ConvTranspose2d(4 * ch, 2 * ch, 2, stride=2)
        self.u8 = ResBlock(2 * ch, emb)
        self.d_up16 = nn.ConvTranspose2d(2 * ch, ch, 2, stride=2)
        self.u16 = ResBlock(ch, emb)
        self.d_out = nn.Conv2d(ch, 3, 3, padding=1)

    def encode(self, x_ctx, k_ctx):
        """x_ctx (N, 3, res, res) at levels k_ctx (N,) -> (N, hid, res/2, res/2)."""
        e = self.level(k_ctx)
        return self.enc_down(F.silu(self.enc_res(self.enc_in(x_ctx), e)))

    def step(self, h, feat, action):
        return self.gru(feat + self.action(action)[:, :, None, None], h)

    def denoise(self, x_noisy, k, h, prev, k_prev):
        """Predict the clean frame from a noisy one at level k, given the history state h and the
        previous frame (as the RNN saw it, at level k_prev)."""
        e = self.level(k)
        lv = net_level(k_prev)[:, None, None, None].expand(-1, 1, *x_noisy.shape[-2:])
        a = self.d16(self.d_in(torch.cat([x_noisy, prev, lv], 1)), e)
        b = self.d_down8(a)
        b = self.d8(self.d_mix8(torch.cat([b, h], 1)), e)
        c = self.d4b(self.d4a(self.d_down4(b), e), e)
        b = self.u8(self.d_up8(c) + b, e)
        a = self.u16(self.d_up16(b) + a, e)
        return self.d_out(F.silu(a))

    def forward(self, x_noisy, k, x_ctx, k_ctx, actions):
        """Teacher-forced pass over whole sequences, all frames denoised in parallel.
        x_noisy, x_ctx: (B, T, 3, H, W); k, k_ctx: (B, T); actions: (B, T).
        Returns the clean-frame predictions (B, T, 3, H, W)."""
        B, T = k.shape
        feats = self.encode(x_ctx.flatten(0, 1), k_ctx.flatten()).unflatten(0, (B, T))
        h = self.h0.expand(B, -1, -1, -1)
        hs = [h]
        for t in range(T - 1):
            h = self.step(h, feats[:, t], actions[:, t])
            hs.append(h)
        hs = torch.stack(hs, 1)  # hs[:, t] = state that has seen frames < t
        prev = torch.cat([torch.zeros_like(x_ctx[:, :1]), x_ctx[:, :-1]], 1)
        k_prev = torch.cat([torch.ones_like(k_ctx[:, :1]), k_ctx[:, :-1]], 1)  # frame 0: no previous frame
        out = self.denoise(x_noisy.flatten(0, 1), k.flatten(), hs.flatten(0, 1),
                           prev.flatten(0, 1), k_prev.flatten())
        return out.unflatten(0, (B, T))


def make_noisy(x, k, eps):
    """x^k = (1 - k) x + k eps, with one level per frame. x, eps: (B, T, 3, H, W); k: (B, T)."""
    k = k[:, :, None, None, None]
    return (1 - k) * x + k * eps


def train_video(model, levels_fn, data, steps=2000, batch=16, T=16, lr=2e-3, seed=0, log=None, device="cpu"):
    """Generic training loop. levels_fn(B, T, generator) -> (k, k_ctx), both (B, T) in [0, 1].
    The target of frame t is noised to k[:, t]; the version of frame t the RNN reads is noised
    to k_ctx[:, t] with the *same* noise, so k_ctx == k means 'the RNN sees the noisy frame' and
    k_ctx == 0 means 'the RNN sees the clean frame'."""
    model.to(device)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    frames, actions = data["frames"], data["actions"]
    N, L = actions.shape
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    losses, t0 = [], time.time()
    for s in range(steps):
        i = torch.randint(N, (batch,), generator=g)
        o = torch.randint(L - T + 1, (batch,), generator=g)
        idx = o[:, None] + torch.arange(T)
        x = to_tensor(frames[i[:, None].numpy(), idx.numpy()]).to(device)
        a = torch.as_tensor(actions[i[:, None].numpy(), idx.numpy()]).long().to(device)
        k, k_ctx = (v.to(device) for v in levels_fn(batch, T, g))
        eps = torch.randn(x.shape, generator=g).to(device)
        x_hat = model(make_noisy(x, k, eps), k, make_noisy(x, k_ctx, eps), k_ctx, a)
        loss = ((x_hat - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log is not None and (s % 100 == 0 or s == steps - 1):
            log(s, float(np.mean(losses[-100:])), time.time() - t0)
    return losses


@torch.no_grad()
def sample_frame(model, h, prev, k_prev, n_steps, g, k_start=1.0, x_start=None, trajectory=None):
    """Denoise one frame from level k_start (pure noise by default) down to 0 with Euler steps
    on the flow-matching velocity dx/dk = (x - x_hat) / k. If `trajectory` is a list, every step
    appends (k, x at level k, the model's clean-frame guess x_hat) to it."""
    B = h.shape[0]
    x = torch.randn(B, *prev.shape[1:], generator=g, device=prev.device) if x_start is None else x_start
    ks = torch.linspace(k_start, 0, n_steps + 1, device=prev.device)
    for j in range(n_steps):  # ks[j] > 0 here, only the last step lands on k = 0
        k = ks[j].expand(B)
        x_hat = model.denoise(x, k, h, prev, k_prev)
        if trajectory is not None:
            trajectory.append((float(ks[j]), x.clone(), x_hat.clone()))
        x = x + (ks[j + 1] - ks[j]) * (x - x_hat) / ks[j]
    return x


@torch.no_grad()
def denoise_next_frame(model, context, actions, n_steps=10, seed=0):
    """Read the real frames context (B, C, 3, H, W) with their actions, then generate the next
    frame and record every Euler step. C = 0 generates a first frame, with no history at all.
    Returns (trajectory, final frame); see sample_frame."""
    dev = context.device
    g = torch.Generator(device=dev).manual_seed(seed)
    B, C = context.shape[:2]
    h, zero = model.h0.expand(B, -1, -1, -1), torch.zeros(B, device=dev)
    for t in range(C):
        h = model.step(h, model.encode(context[:, t], zero), actions[:, t])
    if C == 0:  # no previous frame: zeros at level 1, as in training
        prev, k_prev = torch.zeros(B, 3, model.res, model.res, device=dev), torch.ones(B, device=dev)
    else:
        prev, k_prev = context[:, -1], zero
    traj = []
    x = sample_frame(model, h, prev, k_prev, n_steps, g, trajectory=traj)
    return traj, x


class ImaginedExplorer:
    """The explorer policy of mazeworld, but looking at *generated* frames: it walks forward
    and turns for a while when the frame shows a wall right ahead (the pixels above the center
    are bright wall instead of dark ceiling). Lets a model explore its own imagined maze."""

    def __init__(self, B, seed=0):
        self.rng = np.random.default_rng(seed)
        self.prev = np.zeros(B, dtype=np.int64)

    def __call__(self, frames):  # frames (B, 3, H, W) in [-1, 1]
        H, W = frames.shape[-2:]
        top = (frames[:, :, H // 16, W // 2 - 1:W // 2 + 1].amin((1, 2)) + 1) / 2  # brightness above the center
        acts = []
        for b in range(len(frames)):
            p = int(self.prev[b])
            if top[b] > 0.6:  # wall ahead: keep turning
                a = p if p in (1, 2) else int(self.rng.choice([1, 2]))
            elif self.rng.random() < (0.85 if p == 0 else 0.4):
                a = p
            else:
                a = int(self.rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15]))
            acts.append(a)
        self.prev = np.array(acts)
        return torch.as_tensor(self.prev)


@torch.no_grad()
def rollout(model, context, actions, n_frames, k_ctx=0.0, n_steps=10, seed=0, renoise=None, policy=None):
    """Autoregressive generation. context: (B, C, 3, H, W) real frames; actions: (B, >= C+n-1).
    Each finished frame is fed back to the model at level k_ctx (0 = clean; > 0 = re-noised with
    fresh noise, which is what diffusion forcing allows). renoise(x, k_ctx, g) does the
    re-noising. If `policy` is given (e.g. ImaginedExplorer), it picks the actions after the
    context from the generated frames, and actions only needs to cover the context.
    Returns (B, C + n_frames, 3, H, W), and the actions used (B, C + n_frames)."""
    dev = context.device
    g = torch.Generator(device=dev).manual_seed(seed)
    B, C = context.shape[:2]
    h = model.h0.expand(B, -1, -1, -1)
    frames = list(context.unbind(1))

    k = torch.full((B,), float(k_ctx), device=dev)
    prev, k_prev = torch.zeros_like(context[:, 0]), torch.ones(B, device=dev)

    if renoise is None:
        def renoise(x, k, g):
            return (1 - k) * x + k * torch.randn(x.shape, generator=g, device=x.device)

    used = [actions[:, t] for t in range(C)]
    if policy is not None:
        used[-1] = policy(frames[-1])

    def feed(h, x, a):
        xk = renoise(x, k_ctx, g)
        return model.step(h, model.encode(xk, k), a), xk

    for t in range(C):
        h, prev = feed(h, frames[t], used[t])
        k_prev = k
    for t in range(C, C + n_frames):
        x = sample_frame(model, h, prev, k_prev, n_steps, g)
        # The raw output is fed back as is (no clipping to [-1, 1]); only keep the numbers finite.
        x = torch.nan_to_num(x, nan=0.0).clamp(-10, 10)
        frames.append(x)
        used.append(policy(x) if policy is not None else actions[:, t])
        h, prev = feed(h, x, used[-1])
    return torch.stack(frames, 1), torch.stack(used, 1)


@torch.no_grad()
def rollout_full_sequence(model, context, actions, n_frames, chunk=8, n_steps=10, seed=0):
    """How a full-sequence model continues a video. It was only trained with all frames of a
    clip at one shared noise level, so it cannot read clean past frames. Instead we generate
    `chunk` new frames at a time, inside a window that also holds the last `chunk` known frames,
    and at every denoising step we overwrite ('replace') the known frames with fresh noisy
    copies at the current level k. context: (B, C >= chunk, 3, H, W)."""
    dev = context.device
    g = torch.Generator(device=dev).manual_seed(seed)
    B = context.shape[0]
    frames = list(context.unbind(1))
    while len(frames) < context.shape[1] + n_frames:
        t0 = len(frames) - chunk  # window = frames[t0 : t0 + 2 * chunk]
        known = torch.stack(frames[t0:], 1)
        a = actions[:, t0:t0 + 2 * chunk]
        new = torch.randn(B, chunk, *context.shape[2:], generator=g, device=dev)
        ks = torch.linspace(1, 0, n_steps + 1)
        for j in range(n_steps):
            k = ks[j]
            noisy_known = (1 - k) * known + k * torch.randn(known.shape, generator=g, device=dev)
            window = torch.cat([noisy_known, new], 1)
            kk = torch.full(window.shape[:2], float(k), device=dev)
            x_hat = model(window, kk, window, kk, a)[:, chunk:]
            new = new + (ks[j + 1] - k) * (new - x_hat) / k
        frames.extend(new.clamp(-1, 1).unbind(1))
    return torch.stack(frames[:context.shape[1] + n_frames], 1)


@torch.no_grad()
def generate_clip(model, actions, T=16, n_steps=10, seed=0, levels="joint"):
    """Generate a clip from nothing. levels="joint": all frames share one level and are denoised
    together (how a full-sequence model samples). Returns (B, T, 3, res, res)."""
    g = torch.Generator().manual_seed(seed)
    B = actions.shape[0]
    x = torch.randn(B, T, 3, model.res, model.res, generator=g)
    ks = torch.linspace(1, 0, n_steps + 1)
    for j in range(n_steps):
        kk = torch.full((B, T), float(ks[j]))
        x_hat = model(x, kk, x, kk, actions[:, :T])
        x = x + (ks[j + 1] - ks[j]) * (x - x_hat) / ks[j]
    return x.clamp(-1, 1)


POOL = 8  # realism compares frames average-pooled to POOL x POOL: layout and colors, not texture


def _pool(frames):
    """(..., 3, H, W) -> (N, 3 POOL^2): average-pooled and flattened."""
    f = torch.nan_to_num(frames, nan=0.0).clamp(-1, 1).reshape(-1, *frames.shape[-3:])
    return F.adaptive_avg_pool2d(f, POOL).flatten(1)


def frame_bank(n_episodes=500, seed=12345, size=16):
    """Real frames to compare generated ones against, pooled: (M, 3 POOL^2) in [-1, 1]."""
    import mazeworld as mw
    f = mw.dataset(n_episodes, 32, size, seed=seed)["frames"]
    return _pool(to_tensor(f.reshape(-1, size, size, 3)))


def realism(frames, bank):
    """Distance from each generated frame to its nearest real frame, both average-pooled to
    POOL x POOL (lower = more like real data). The pooling makes the measure about the layout and
    the colors of a frame rather than its fine texture, and the same for every resolution.
    frames (..., 3, H, W) float; bank from frame_bank(). Returns (...) tensor."""
    d = torch.cdist(_pool(frames), bank).min(1).values / math.sqrt(bank.shape[1])
    return d.reshape(frames.shape[:-3])
