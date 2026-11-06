"""Train one notebook-3 video model on a GPU with the notebook's own code (course/video.py:
VideoDenoiser + train_video, reference noise levels).
  python tools/nb3_train_gpu.py --regime "teacher forcing" --res 32 --big --steps 30000 --out runs/big_tf
Episodes are those of mw.dataset(n, 32, res, seed=0), generated in parallel and cached."""
import argparse, os, time
from multiprocessing import Pool

import numpy as np
import torch

import mazeworld as mw
from course import solutions, video as V


def _ep(a):
    i, res = a
    e = mw.episode(i, 32, res)
    return e["frames"], e["actions"]


def data(n, res):
    path = f"data/res{res}_n{n}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return dict(frames=d["frames"], actions=d["actions"])
    t = time.time()
    with Pool(48) as p:
        out = p.map(_ep, [(i, res) for i in range(n)], chunksize=16)  # seed 0: episode seeds 0..n-1
    d = dict(frames=np.stack([o[0] for o in out]), actions=np.stack([o[1] for o in out]))
    os.makedirs("data", exist_ok=True)
    np.savez(path, **d)
    print(f"generated {n} episodes at {res}x{res} in {time.time() - t:.0f}s", flush=True)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--res", type=int, default=16)
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n_eps", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    d = data(a.n_eps, a.res)
    model = V.VideoDenoiser(**V.BIG) if a.big else V.VideoDenoiser(res=a.res)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    r = a.regime

    def log(s, loss, secs):
        print(f"step {s} loss {loss:.4f} {secs:.0f}s", flush=True)
        if s % 5000 == 0 and s > 0:
            torch.save(model.state_dict(), f"{a.out}/model.pt")

    V.train_video(model, lambda B, T, g: solutions.noise_levels(r, B, T, g), d, steps=a.steps,
                  batch=a.batch, lr=a.lr, log=log, device="cuda")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"{a.out}/model.pt")
    print("done", flush=True)
