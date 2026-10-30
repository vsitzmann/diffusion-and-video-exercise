"""Train the pretrained checkpoints shipped in checkpoints/ (the notebook's "load the pretrained
checkpoints" option). On srg-lab-server: ./run python tools/train_checkpoints.py 3000
Takes ~5 min per regime on 4 CPU cores."""
import sys

import torch

import mazeworld as mw
from course import solutions
from course import video as V

torch.set_num_threads(4)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
regimes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["full sequence", "teacher forcing", "diffusion forcing"]
data = mw.dataset(2000, 32, 16, seed=0)  # the same data the notebook trains on
for r in regimes:
    model = V.VideoDenoiser()
    V.train_video(model, lambda B, T, g, r=r: solutions.noise_levels(r, B, T, g), data, steps=steps,
                  log=lambda s, l, t, r=r: print(f"{r} step {s} loss {l:.4f} {t:.0f}s", flush=True))
    torch.save(model.state_dict(), f"checkpoints/{r.replace(' ', '_')}.pt")
