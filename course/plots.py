"""Small matplotlib helpers shared by the notebooks."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402


def _player(anim, fig):
    """Wrap a matplotlib animation as a marimo element. It needs an iframe, because marimo does
    not run inline scripts in normal outputs."""
    import marimo as mo

    w, h = fig.get_size_inches() * fig.dpi
    html = anim.to_jshtml(default_mode="loop")
    plt.close(fig)
    style = "<style>body{margin:0;font-family:sans-serif;font-size:12px}</style>"
    return mo.iframe(style + html, width=f"{int(w) + 70}px", height=f"{int(h) + 110}px")

DATA = "#b9b9b9"  # training data
SAMPLE = "#d9480f"  # generated samples
ACCENT = "#1c7ed6"
LIM = 2.9
plt.rcParams.update({"figure.dpi": 100, "font.size": 9, "axes.titlesize": 10})


def np_(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def square(ax, lim=LIM, title=None):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def scatter(ax, pts, color=SAMPLE, s=3, alpha=0.6, **kw):
    p = np_(pts)
    return ax.scatter(p[:, 0], p[:, 1], s=s, c=color, alpha=alpha, linewidths=0, **kw)


def samples_vs_data(panels, data, ncols=None, size=2.6, suptitle=None):
    """panels: list of (title, samples). One square plot per panel, data in grey underneath."""
    n = len(panels)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (title, pts) in zip(axes.flat, panels):
        scatter(ax, data, DATA, s=2, alpha=0.5)
        if pts is not None:
            scatter(ax, pts)
        square(ax, title=title)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def animate_samples(snapshots, data, titles=None, size=3.0, interval=120, field=None):
    """Animate a list of (n, 2) sample snapshots over the data. Optionally `field` is a list of
    (n, 2) arrows (one per snapshot) drawn on the samples. Returns a marimo element (a player)."""
    fig, ax = plt.subplots(figsize=(size, size))
    scatter(ax, data, DATA, s=2, alpha=0.5)
    sc = scatter(ax, snapshots[0])
    q = None
    if field is not None:
        p, f = np_(snapshots[0]), np_(field[0])
        q = ax.quiver(p[:, 0], p[:, 1], f[:, 0], f[:, 1], angles="xy", scale_units="xy", scale=1,
                      width=0.004, color=ACCENT, alpha=0.6)
    square(ax, title=titles[0] if titles else None)
    fig.tight_layout()

    def update(i):
        p = np_(snapshots[i])
        sc.set_offsets(p)
        if q is not None:
            q.set_offsets(p)
            f = np_(field[i])
            q.set_UVC(f[:, 0], f[:, 1])
        if titles:
            ax.set_title(titles[i])
        return (sc,)

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=interval)
    return _player(anim, fig)


def trajectories(ax, traj, color=SAMPLE, n=60, lw=0.7, alpha=0.5):
    """traj: (steps, N, 2). Draws n paths, with dots at the end."""
    t = np_(traj)[:, :n]
    for i in range(t.shape[1]):
        ax.plot(t[:, i, 0], t[:, i, 1], color=color, lw=lw, alpha=alpha)
    ax.scatter(t[-1, :, 0], t[-1, :, 1], s=6, c=color, zorder=3)
    ax.scatter(t[0, :, 0], t[0, :, 1], s=4, c="k", alpha=0.4, zorder=3)


def loss_curve(losses, title="training loss", logy=True, size=(4, 2.4)):
    fig, ax = plt.subplots(figsize=size)
    l = np.asarray(losses)
    k = max(1, len(l) // 100)
    ax.plot(np.convolve(l, np.ones(k) / k, mode="valid"), color=ACCENT)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def video_grid(rows, labels=None, every=1, max_frames=None, scale=0.55):
    """Static film strip: rows is a list of uint8 videos (T, H, W, 3)."""
    T = min(len(r) for r in rows)
    idx = list(range(0, T if max_frames is None else min(T, max_frames), every))
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(scale * len(idx), scale * len(rows) * 1.12),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c, t in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(row[t], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(str(t), fontsize=6)
        if labels:
            axes[r, 0].set_ylabel(labels[r], rotation=0, ha="right", va="center", fontsize=8)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.15)
    return fig


def video_html(rows, labels=None, size=1.6, interval=120, extra=None):
    """Animated side-by-side videos. rows: list of uint8 videos (T, H, W, 3).
    extra: optional list of images per frame (e.g. a map), shown in an additional panel."""
    n = len(rows) + (extra is not None)
    T = min(len(r) for r in rows)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size * 1.15), squeeze=False)
    ims = []
    for i, row in enumerate(rows):
        ims.append(axes[0, i].imshow(row[0], interpolation="nearest"))
        if labels:
            axes[0, i].set_title(labels[i], fontsize=8)
    if extra is not None:
        ims.append(axes[0, -1].imshow(extra[0], interpolation="nearest"))
        axes[0, -1].set_title("map", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    txt = fig.text(0.01, 0.02, "", fontsize=7)
    fig.tight_layout()

    def update(t):
        for im, row in zip(ims, rows):
            im.set_data(row[t])
        if extra is not None:
            ims[-1].set_data(extra[t])
        txt.set_text(f"frame {t}")
        return ims

    anim = animation.FuncAnimation(fig, update, frames=T, interval=interval)
    return _player(anim, fig)


def decoded_panels(results, emb, data_u, title):
    """High-D samples, decoded back to the hidden 2D sheet. results: list of (name, samples in
    R^D). Color = distance from the sheet; each title also gives the score of the data mean
    (a perfectly blurry sample) for comparison."""
    import torch

    _, mean_off = emb.decode(emb(data_u).mean(0, keepdim=True))
    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results) + 0.8, 3.8), squeeze=False)
    for ax, (name, y) in zip(axes[0], results):
        u, off = emb.decode(torch.as_tensor(y))
        scatter(ax, data_u, DATA, s=2, alpha=0.5)
        sc = ax.scatter(u[:, 0], u[:, 1], c=off.clamp(max=0.7), cmap="viridis_r", vmin=0, vmax=0.7, s=4)
        square(ax)
        ax.set_title(f"{name}\ndistance from sheet: {off.mean():.2f}", fontsize=9)
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8,
                 label=f"distance from sheet (data mean: {mean_off.item():.2f})")
    fig.suptitle(title, fontsize=10)
    return fig
