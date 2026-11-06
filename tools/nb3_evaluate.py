"""Part 5 of notebook 3 for a set of three trained models: the same rollouts and realism measure.
  python tools/nb3_evaluate.py --dir runs --prefix big_ --res 32 --big --out results/big.npz
Saves the mean realism curves, the broken-frame fractions, and the videos of a few episodes."""
import argparse, time
import numpy as np
import torch

import mazeworld as mw
from course import solutions, video as V

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs")
ap.add_argument("--prefix", required=True)
ap.add_argument("--res", type=int, default=16)
ap.add_argument("--big", action="store_true")
ap.add_argument("--n_test", type=int, default=24)
ap.add_argument("--length", type=int, default=200)
ap.add_argument("--k_ctx", type=float, default=0.1)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--keep", type=int, default=4, help="episodes whose videos are saved")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
THR = 0.2  # broken frame: pooled distance to the nearest real frame above this
torch.set_num_threads(a.threads)

models = {}
for r, tag in [("full sequence", "fs"), ("teacher forcing", "tf"), ("diffusion forcing", "df")]:
    m = V.VideoDenoiser(**V.BIG) if a.big else V.VideoDenoiser(res=a.res)
    m.load_state_dict(torch.load(f"{a.dir}/{a.prefix}{tag}/model.pt", map_location="cpu"))
    models[r] = m.eval().to(a.device)

t0 = time.time()
test = mw.dataset(a.n_test, 304, a.res, seed=99)
ctx = V.to_tensor(test["frames"][:, :8]).to(a.device)
acts = torch.as_tensor(test["actions"]).long().to(a.device)


def renoise(x, k, g):  # the reference re-noising, with the noise drawn on x's device
    return (1 - k) * x + k * torch.randn(x.shape, generator=g, device=x.device)
bank = V.frame_bank(size=a.res)
print(f"data {time.time() - t0:.0f}s", flush=True)
n = a.length
ro = {"full sequence": V.rollout_full_sequence(models["full sequence"], ctx, acts, n - 4, chunk=8).cpu()}
print(f"fs {time.time() - t0:.0f}s", flush=True)
for r in ["teacher forcing", "diffusion forcing"]:
    for lab, kk in [("clean", 0.0), (f"re-noised {a.k_ctx:.2f}", a.k_ctx)]:
        ro[f"{r} · {lab}"] = V.rollout(models[r], ctx[:, :4], acts, n, k_ctx=kk, renoise=renoise)[0].cpu()
        print(f"{r} {lab} {time.time() - t0:.0f}s", flush=True)

real = V.realism(V.to_tensor(test["frames"][:, :n]), bank)
dist = {k: V.realism(v, bank) for k, v in ro.items()}
print("real frames: mean distance %.3f, 99th pct %.3f" % (real.mean(), real.quantile(0.99)))
for k, d in dist.items():
    q = d[:, -n // 4:]
    print(f"{k:32s} mean {d.mean():.3f}  last quarter mean {q.mean():.3f}  broken {(q > THR).float().mean():.2f}  episodes with >10% broken {((q > THR).float().mean(1) > 0.1).float().mean():.2f}")
# episodes to keep: where teacher forcing breaks the most, then the first few others
order = list(np.argsort(-(dist["teacher forcing · clean"] > THR).sum(1).numpy()))
keep = order[:a.keep]  # keep[0]: where teacher forcing breaks the most
out = dict(names=np.array(list(ro)), keep=np.array(keep), real_dist=real.numpy(),
           real_frames=test["frames"][keep, :n])
for i, (k, v) in enumerate(ro.items()):
    out[f"dist_{i}"] = dist[k].numpy()
    out[f"video_{i}"] = V.to_uint8(v[keep])
np.savez_compressed(a.out, **out)
print("saved", a.out, f"{time.time() - t0:.0f}s")
