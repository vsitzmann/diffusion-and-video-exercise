# Course exercises: notes for maintainers

For whoever develops the course (people or agents). `AGENTS.md` is the brief for *students'* AI
assistants (tutor mode) and must not point here.

The hands-on part of Vincent's course "Video Generative Models: The LLM Moment for Embodied
Intelligence?" (Munich, Nov 6, 2026). The course context (who, when, the agreed abstract, the mail
threads) is in the AGENTS.md of the course's deck in `~/Decks/Vincents talks/`.
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
04_video_policy.py            (notebook 4 uses course/check4.py and course/solutions4.py)
course/toys.py        2D toys (TOYS) and Embedding: identity / linear (JiT) / crumpled cos(Wu+b), decode()
course/nets.py        Generator (one-step) and FlowNet (pred="v" or "x")
course/plots.py       matplotlib helpers. Animations go through _player() -> mo.iframe, because
                      marimo does not run <script> in mo.Html.
course/video.py       VideoDenoiser (ConvGRU over context frames + small U-Net denoiser that also gets
                      the previous frame), train_video, rollout, rollout_full_sequence, realism
course/check.py       exercise checks; QC_SOLUTIONS=1 turns the solutions switch on by default
course/solutions.py   reference solutions (students may read them)
course/policy.py      notebook 4: next-frame U-Net denoiser (DIAMOND-style frame stacking, per-frame
                      context noise levels, actions or null, egocentric goal map or null), guided
                      Euler sampling, generate/imagine, train_video, IDM + train_idm, navigate
                      (closed loop), viewer_html (three.js viewer, template course/assets/viewer4.html)
course/assets/        viewer4.html; nb4_reference.json = GPU results of the pretrained nb4 models
mazeworld/            maze generation, explorer policy, dynamics (NumPy)
                      museum.py: notebook 3's renderer since Nov 6 (brick walls, paintings, ceiling lamps;
                      NumPy only, reuses world4.cast); world.render calls it
                      render_numpy.py + rust/: the old flat-colored raycaster, no longer used by any notebook
                      world4.py: notebook 4's 64x64 textured world (NumPy only), expert, dataset()
rust/                 PyO3/maturin crate mazeworld_rs: the old raycaster, rayon-parallel (unused)
checkpoints/          shipped video checkpoints (1500 steps, seed 0, = the notebook default; tools/train_checkpoints.py)
                      nb3_big_*.pt + nb3_big_results.npz: notebook 3's big 32x32 models and their Part 5 results
                      nb4_video16/64.pt, nb4_idm16/64.pt: notebook 4 (fp16 EMA weights, trained on schadenfreude)
tools/                train_checkpoints.py, check_parity.py, test_notebooks.sh,
                      nb3_train_gpu.py + nb3_evaluate.py (notebook 3 checkpoints and Part 5 results on a GPU)
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
- `tools/check_parity.py` compares the old NumPy and Rust raycasters (render_numpy, rust/).
  Notebook 3 no longer uses either; it renders with mazeworld/museum.py (about 1 ms per 16x16
  frame, 1.6 ms per 32x32 frame on one core).
- Notation (Vincent's decision, Nov 5): the lecture's, which is Sander Dieleman's. Data $x$ at
  $t=0$, noise $\epsilon \sim \mathcal N(0, I)$ at $t=1$, noisy point $x_t = \alpha_t x + \sigma_t\epsilon$,
  here the straight line $x_t = (1-t)x + t\epsilon$ with velocity $\epsilon - x$; sampling runs from
  $t=1$ down to $t=0$ ($x_{t-h} = x_t - h\,u_t(x_t)$). $p_t(x_t \mid x)$, $u_t(x_t \mid x) = (x_t - x)/t$,
  posterior mean $\hat x(x_t, t) = \mathbb E[x \mid x_t]$. Code: data `x`, training set `X`, noisy
  point `x_t`, noise `eps`. The one-step generator of notebook 1 also takes noise $\epsilon$. Never
  use $z$ for noise or data ($z$ only for learned latents), and never Lipman's $x_1$-as-data.
  Plots keep noise on the left and data on the right. In notebooks 3 and 4 the noise level is $k$,
  because $t$ is the frame index: $x_t^k = (1-k)x_t + k\epsilon$, $k = 0$ clean, $k = 1$ pure noise,
  and sampling runs from $k=1$ down to $k=0$. A missing frame ("no history") is level 1.
  **The shipped checkpoints were trained with the old convention** ($k = 1$ clean), so the networks
  read $1-k$. That conversion happens in exactly one place per model: `net_level()` in
  course/video.py and the top of `Denoiser.forward` in course/policy.py. Keep it there. Before and
  after the switch (Nov 5), rollouts and closed-loop runs on fixed seeds matched to float precision.

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
  with context re-noised to k = 0.1 0.02–0.04, diffusion forcing 0.00. Teacher forcing also freezes
  (frame-to-frame motion drops from 0.13 to 0.08). Rollouts feed the raw output back (only
  nan_to_num and ±10 to keep numbers finite).
- `ImaginedExplorer` (course/video.py) picks actions from generated frames. It looks nicer, but
  it shows *less* divergence than replaying the test episode's actions, so the notebook uses
  the GT actions.
- Full-sequence continuation uses chunked replacement sampling (8 known + 8 new frames).

## Notebook 3 since Nov 6, 2026: the museum world and the big model

- **World.** `mazeworld/museum.py` draws the same block mazes and dynamics as before: cream brick
  walls with framed procedural paintings (muted palettes; CIFAR is not available on laptops),
  a slate ceiling with round lamps, a grey floor with light pools. The old Oct 30 measurements
  below were made in the old flat-colored world; they no longer apply in detail.
- **Realism measure.** `video.realism` now compares frames average-pooled to 8x8 (`POOL`) with the
  nearest of 16k real frames; a frame is "broken" above 0.2 (notebook `BROKEN`). At full
  resolution textured frames are rarely close to any bank frame (real frames already scored 0.125),
  so nothing separated. Known blind spots: blotchy texture artifacts (the big teacher-forced model
  with re-noised context scores like real video) and freezing in front of a wall.
- **Small checkpoints** (`checkpoints/{full_sequence,teacher_forcing,diffusion_forcing}.pt`): the
  notebook's own settings (1500 steps, batch 16, lr 2e-3, `mw.dataset(2000, 32, 16, seed=0)`),
  trained on schadenfreude with `tools/nb3_train_gpu.py`.
- **Big model** (`checkpoints/nb3_big_*.pt`, fp16): `VideoDenoiser(**video.BIG)` (ch 48, hid 96,
  emb 128, res 32; 2.88M parameters) with the unchanged `train_video`, 20k steps, batch 32,
  lr 1e-3, 20k episodes at 32x32, about 50 min per regime on an A6000. Runs in
  `/nobackup/nvme1/sitzmann/video_nb3/runs/big_*`. `nb3_big_results.npz` = Part 5 for the
  big models (`tools/nb3_evaluate.py --device cuda`, 32 test episodes, 200 frames; videos of the
  three episodes where teacher forcing breaks most). Rolling them out takes about 15 s on a GPU,
  minutes on a laptop CPU, hence the precomputed file. The bonus play cell uses `nb3_big_diffusion_forcing.pt`.
- **Measured** (32 episodes, last quarter of 200 frames, broken = pooled distance > 0.2; real video
  about 1%): small models: teacher forcing clean 12% (25% of episodes), re-noised 0.1 0%,
  diffusion forcing 2–3%, full sequence 0%. Big models: teacher forcing clean 11% (19% of
  episodes), re-noised 0% by the measure but visibly covered in blue blotches (it copies the noise),
  diffusion forcing 1–2%, full sequence 0%. During training the big diffusion-forcing model was
  much worse: 58% broken at 5k steps (it walked into large dark paintings and stayed there), 5–12%
  at 10k. Judge only final checkpoints.
- **A variant with smaller, lighter paintings** (Nov 6, `/nobackup/nvme1/sitzmann/video_nb3_v2`)
  was cleaner for the small models (teacher forcing 22%, diffusion forcing 0%) but the big
  teacher-forced model hardly broke (1%), so exposure bias was invisible at 32x32. Not used.
- Rollout code (`sample_frame`, `rollout`, `rollout_full_sequence`) draws its noise on the
  context's device, so it runs on a GPU; on CPU the random streams are unchanged.

## Notebook 4 (Nov 4, 2026)

- **Provenance.** The models were trained on schadenfreude (`/nobackup/nvme1/sitzmann/video_nb4`,
  scripts in `dev/nb4/gpu/`, report in `dev/nb4/results/report.html`). `nb4_video16` = run
  `s16_n` (about 40k steps), `nb4_video64` = run `big64m` (30k steps, 30% uniform noise levels).
  The 16×16 model was trained on box-filtered 64×64 frames; the notebook renders 16×16 directly
  with 4×4 supersampling (mean difference 1.4/255), and closed-loop success is unchanged.
- **What the goal conditioning needed.** An egocentric map (not the global one) and the plain
  x-loss with noise levels biased toward noise (logit-normal, mean +0.8 in the noise-level
  convention; it was −0.8 when $k = 1$ meant clean). JiT's v-loss weighting
  made the model ignore the map. Don't change these without re-measuring the direction match.
- **Measured** (64 held-out mazes, 400-step budget, `course/assets/nb4_reference.json`): closed-loop
  success at w = 0/1/3, 10 Euler steps: 16×16 12/86/92%, 64×64 12/77/97%; one Euler step at w = 3:
  97% / 100%. A student model trained from scratch on CPU (3000 episodes, 5000 steps at batch 64,
  ~12 min on 4 cores) reached 13/50/78% (32 mazes, one step).
- History guidance does **not** help this model (the history is in the input channels): over frames
  20–40, PSNR 15.2 / 15.0 / 14.3 / 12.8 dB at w = 1 / 1.5 / 2 / 3. The notebook says so.
- Test runs: `QC_RES=64` forces the 64×64 setting. On CPU the 64×64 path is slow (minutes per part).
- molab: the notebook has a PEP 723 header and clones the repository if `course` is missing. It needs
  the GitHub repository to be public. Not yet tried on molab itself.
