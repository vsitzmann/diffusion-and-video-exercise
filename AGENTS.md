# Course exercises: notes for agents

The hands-on part of Vincent's course "Video Generative Models: The LLM Moment
for Embodied Intelligence?" (Munich, Nov 6, 2026). The course context (who, when, the agreed
abstract, the mail threads) is in the deck's brief:
the AGENTS.md of the course's deck in `~/Decks/Vincents talks/`.
GitHub: `vsitzmann/diffusion-and-video-exercise` (private).

**Audience.** Economists, statisticians, data scientists. They are quantitative,
but not deep-learning specialists. They self-study on their own laptops (CPU only, possibly
weak), in about 30–60 minutes in total. Evaluation does not matter. Understanding the techniques
and the math does. Every exercise has its LaTeX math and a visualization right next to the code.

## Layout

```
01_regression_to_drifting.py  marimo notebooks (plain .py). Exercises raise NotImplementedError;
02_flow_matching.py           a check cell below each one; a switch at the top swaps in
03_video_world_model.py       course/solutions.py for every exercise.
course/toys.py        2D toys (TOYS) and Embedding: identity / linear (JiT) / crumpled cos(Wu+b), decode()
course/nets.py        Generator (one-step) and FlowNet (pred="v" or "x")
course/plots.py       matplotlib helpers. Animations go through _player() -> mo.iframe, because
                      marimo does not run <script> in mo.Html.
course/video.py       VideoDenoiser (ConvGRU over context frames + small U-Net denoiser that also gets
                      the previous frame), train_video, rollout, rollout_full_sequence, realism
course/check.py       exercise checks; QC_SOLUTIONS=1 turns the solutions switch on by default
course/solutions.py   reference solutions (students may read them)
mazeworld/            maze generation, explorer policy, dynamics (NumPy) + render_numpy.py
rust/                 PyO3/maturin crate mazeworld_rs: the same raycaster, rayon-parallel
checkpoints/          shipped video checkpoints (1500 steps, seed 0, = the notebook default; tools/train_checkpoints.py)
tools/                train_checkpoints.py, check_parity.py, test_notebooks.sh
run                   resource-capped wrapper (4 cores, 4 GB, nice 19), same as gen_models_exploration
dev/                  scratch (gitignored): prototypes, exported HTML, screenshots
```

## Working here

- Run Python through `./run` (this is srg-lab-server, shared with Vincent's services). The venv is
  `.venv` (Python 3.14, CPU torch, marimo, maturin, mazeworld_rs installed from `rust/`).
- **Test the notebooks** with `tools/test_notebooks.sh`. It runs each notebook as a script and
  exports it with outputs to `dev/*.html`. To look at the result, screenshot that file with
  headless chromium (`chromium --headless --no-sandbox --window-size=1100,14000
  --virtual-time-budget=30000 --screenshot=out.png file://…`). Animations show only frame 0
  there. `QC_AUTORUN=1` presses notebook 3's training button.
- marimo rules: every global name may be defined in only one cell; prefix cell-local names with
  `_`. Keep heavy work behind `mo.stop` / run buttons when it takes more than a few seconds.
- After changing either raycaster, run `tools/check_parity.py`. The two must agree pixel for
  pixel (they do: max difference 0). Rebuild the Rust wheel with
  `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -m rust/Cargo.toml -o dev/wheels`
  and `pip install` it. The toolchain is user-local (rustup, `~/.cargo`).
- Notation follows the lecture (MIT 6.S184): noise $X_0$ at $t=0$, data $z$ at $t=1$,
  $x = t z + (1-t)\epsilon$. Notebook 1 uses the deck's earlier notation ($z$ = noise, $x$ = data);
  notebook 2 flags the switch. In notebook 3 the noise level is $k$ (1 = clean), because $t$ is the frame index.

## What the experiments showed (Oct 30, 2026), and the settings that depend on it

- **Primitive drifting** (NN in batch + λ·softmax-kernel repulsion, τ = 0.07 × median data distance,
  1500 steps, Adam 1e-3): needs **λ = 1**. At λ = 0.5 it is cleaner in 2D, but the *linear* R⁶⁴
  embedding collapses in 2 of 3 seeds. At λ = 1: 2D and linear R⁶⁴ cover 8/8 modes (3/3 seeds), and
  crumpled R⁶⁴ (ω = 4) collapses to 1 mode (3/3 seeds, for any ω in 1–8 and τ in 0.03–0.2). With the
  narrow default kernel the collapse is onto one mode *on* the sheet, not a blurry mean. This
  matches Vincent's own `gen_models_exploration` exp 4: dimension alone is not the problem,
  crumpling is.
- **Flow matching**: in 2D, v-prediction is better (91% of samples on a mode vs 52% for
  x-prediction with uniform t). On the crumpled sheet in R²⁵⁶, v-prediction fails (distance from
  the sheet 0.68, worse than the data mean's 0.63), while x-prediction works (0.09, 8/8 modes).
  So notebook 2 uses v in Parts 5 and x/v side by side in Part 6, at D = 256 by default.
- **Video model**: 16×16 RGB, ~336k parameters, batch 16 × 16 frames, ~0.1 s/step on 4 cores,
  so 1500 steps ≈ 2.5 min per regime. The denoiser *must* see the previous frame (concatenated
  input). Without it, all regimes stay blurry at CPU budgets.
- **Exposure bias is fragile in a world this simple. Don't change the motion speed or the
  training length without re-measuring.** With faster motion (turn π/12, step 0.3), teacher
  forcing diverged in only 1 of 3 training seeds (white-out), and a 3000-step model never
  diverged, even from corrupted context. Generated frames were too close to real ones. Clipping the
  fed-back frames to [-1, 1] also hid most of it. The current world (turn **π/16**, step **0.15**:
  every frame is mostly a copy of the last) makes it robust. At 1500 steps, 200-frame rollouts (32
  episodes, GT actions) had these fractions of broken frames (distance to the nearest real frame
  > 0.3) in the last quarter: teacher forcing with clean context 0.32 / 0.23 / 0.24 (seeds 0/1/2),
  with context re-noised to k = 0.9 0.02–0.04, diffusion forcing 0.00. Teacher forcing also freezes
  (frame-to-frame motion drops from 0.13 to 0.08). Rollouts feed the raw output back (only
  nan_to_num and ±10 to keep numbers finite).
- `ImaginedExplorer` (course/video.py) picks actions from generated frames. It looks nicer, but
  it shows *less* divergence than replaying the test episode's actions, so the notebook uses
  the GT actions.
- Full-sequence continuation uses chunked replacement sampling (8 known + 8 new frames).
