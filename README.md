# From regression to world models: hands-on exercises

Exercises for the course **"Video Generative Models: The LLM Moment for Embodied
Intelligence?"** (Vincent Sitzmann, MIT, Munich, Nov 6, 2026).

There are three short notebooks, meant for self-study, about 20–25 minutes each. Everything runs
on a laptop CPU. In each one you implement the key few lines yourself, with the math and a
picture of what you are building right next to the code.

| notebook | what you build | what you see |
|---|---|---|
| `01_regression_to_drifting.py` | regression onto random data; a primitive drifting model (nearest-neighbor attraction + repulsion) | regression gives the mean; drifting works in 2D and breaks when the data lives on a crumpled sheet in high dimensions |
| `02_flow_matching.py` | conditional path, conditional and marginal vector fields, the posterior over data points, the flow-matching loss, an Euler sampler | forward and reverse process, conditional vs. marginal flow paths, why flow matching (with x-prediction) still works in high dimensions |
| `03_video_world_model.py` | the noise levels of full-sequence diffusion, teacher forcing and diffusion forcing; re-noising the past | a tiny action-conditioned video model of a maze world; exposure bias and how a little noise on the past frames fixes it; play inside the model |

## Setup (5 minutes)

You need Python 3.10 or newer. The easiest way is [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:vsitzmann/diffusion-and-video-exercise.git
cd diffusion-and-video-exercise
uv sync                                          # creates .venv with torch (CPU), marimo, ...
uv run marimo edit 01_regression_to_drifting.py  # opens the notebook in your browser
```

Without uv:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # Linux/Windows: add --extra-index-url https://download.pytorch.org/whl/cpu
marimo edit 01_regression_to_drifting.py
```

Run the notebooks from the repository folder, in order.

## How the notebooks work

These are [marimo](https://marimo.io) notebooks. They are plain Python files, but they open in the
browser like Jupyter. Cells re-run automatically when something they depend on changes, so there
is no hidden state and no "run all cells again".

- Cells marked **✏️ Exercise** contain a `raise NotImplementedError`. Replace it with your code.
  A check right below tells you whether it works.
- Stuck? Each notebook has a switch at the top that uses the reference solutions instead
  (`course/solutions.py`), so you can always continue.
- Sliders and dropdowns change datasets and hyperparameters. Play with them.
- Notebook 3 trains three small video models (2–3 minutes each). If you are short on time, choose
  "load the pretrained checkpoints".

## What is in here

```
01_…, 02_…, 03_….py   the notebooks
course/               shared code: toy datasets and embeddings, networks, plots, the video
                      model, the exercise checks, and the reference solutions
mazeworld/            the maze world: maze generation, explorer policy, NumPy raycaster
rust/                 the same raycaster in Rust (optional, ~100x faster)
checkpoints/          pretrained video models for notebook 3
```

**Optional: the Rust renderer.** The maze world renders with NumPy out of the box, which is fast
enough for the exercises. If you have a [Rust toolchain](https://rustup.rs), you can install the
compiled renderer. It draws exactly the same frames, at about 250,000 frames per second on 4 cores:

```bash
uv pip install ./rust     # or: pip install ./rust
```

## References

- Lecture notation for flow matching: Holderrieth & Erives, *An Introduction to Flow Matching
  and Diffusion Models* (MIT 6.S184).
- Drifting models: Deng et al., *Generative Modeling via Drifting* (2026).
- x-prediction in high dimensions: Li & He, *Back to Basics: Let Denoising Generative Models
  Denoise* (JiT, 2025).
- Diffusion forcing: Chen et al., *Diffusion Forcing: Next-token Prediction Meets Full-Sequence
  Diffusion* (2024).
- Noise augmentation for game world models: Valevski et al., *Diffusion Models Are Real-Time
  Game Engines* (GameNGen, 2024).
