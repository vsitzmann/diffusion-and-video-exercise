# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.25", "torch>=2.2", "numpy>=1.24", "matplotlib>=3.8"]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="4 · From world model to policy")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    try:
        import course  # noqa: F401
    except ImportError:  # opened on its own (e.g. on molab from a GitHub link): fetch the rest of the repository
        _repo = Path("diffusion-and-video-exercise")
        if not _repo.exists():
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/vsitzmann/diffusion-and-video-exercise.git", str(_repo)], check=True)
        sys.path.insert(0, str(_repo.resolve()))
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from course import policy as P, check4, plots
    from mazeworld import world4 as w4
    torch.set_num_threads(4)
    DEV = P.device()
    AUTORUN = os.environ.get("QC_AUTORUN") == "1"
    RES_DEFAULT = int(os.environ.get("QC_RES") or (64 if DEV.type == "cuda" else 16))  # QC_RES: for test runs
    return AUTORUN, DEV, P, RES_DEFAULT, check4, mo, np, plots, plt, time, torch, w4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4 · From world model to policy

    In notebook 3 the video model was a **world model**: given the past frames and the actions,
    it imagines what happens next. Here we turn the same kind of model into a **policy** that
    drives an agent to a goal, the recipe behind UniPi and CueNav:

    1. **imagine** a video of reaching the goal,
    2. **read off** the actions that would produce that video with an *inverse dynamics model*,
    3. **execute** the first few actions in the real world, look again, and repeat.

    The steering wheel is **classifier-free guidance**, the same trick that makes text-to-image
    models follow their prompt. You will implement it, and then use one guidance weight to make
    the model's memory more reliable and another to make it head for the goal.

    Time: about 30 minutes. ✏️ marks an exercise, and the switch uses the reference solutions.
    Notebook 3 is not needed, but it explains diffusion forcing in more detail.
    """)
    return


@app.cell(hide_code=True)
def _(DEV, RES_DEFAULT, check4, mo):
    use_solutions = check4.solutions_switch()
    res_choice = mo.ui.dropdown({"16×16 (laptop)": 16, "64×64 (best on a GPU, e.g. molab)": 64},
                                value="64×64 (best on a GPU, e.g. molab)" if RES_DEFAULT == 64 else "16×16 (laptop)",
                                label="resolution")
    mo.vstack([use_solutions, mo.hstack([res_choice, mo.md(f"device: **{DEV.type}**")], justify="start", gap=2)])
    return res_choice, use_solutions


@app.cell(hide_code=True)
def _(res_choice):
    RES = res_choice.value
    return (RES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 · The task

    The world from notebook 3 has grown: 6×6-cell mazes with stone walls, a tiled floor and a sky,
    rendered at 64×64 (the laptop version of the model sees the same views at 16×16). The
    dynamics are the same: **forward**, **turn left**, **turn right**. The **goal** is a pink orb
    with a beam of light. The agent must reach it in a maze it has never seen.

    Below, an *expert* drives to the goal. It cheats: it knows the maze and follows the shortest
    path (a breadth-first search on the grid). We record such drives as training data. The
    3/4 view at the bottom shows the agent's camera floating above the maze; drag to orbit.
    """)
    return


@app.cell(hide_code=True)
def _(P, w4):
    demo_env = w4.test_env(9)
    P.show_viewer(P.expert_viewer(demo_env))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 · One model, three jobs

    The model is a next-frame denoiser, like the ones that simulate games (DIAMOND, GameNGen). A
    U-Net receives the noisy next frame $x_t^{k}$ at noise level $k$ (as in notebook 3,
    $x^{k} = (1-k)\,x + k\,\epsilon$, so $k = 0$ is clean), the last 4 frames stacked as extra
    input channels, and predicts the clean frame:

    $$
    \hat x_t = D_\theta\big(x_t^{k},\, k \;\big|\; \underbrace{x_{t-4}^{\tilde k_1}, \dots, x_{t-1}^{\tilde k_4}}_{\text{history}},\; \underbrace{a_{t-4:t-1}}_{\text{actions}},\; \underbrace{m}_{\text{goal map}}\big).
    $$

    - **History.** Each past frame comes with its *own* noise level $\tilde k_j$, as in diffusion
      forcing. Level 1 means the frame is pure noise, so the model has to do without it.
    - **Actions** $a$: forward, left, right, or a fourth token "no action given".
    - **Goal map** $m$: a small map centered on the agent and rotated so that "ahead" is up
      (as in CueNav): walls, the goal, and how far each cell is from the goal. Or "no map".

    Which conditions we provide decides what the model *is*:

    | given | the model is | it imagines |
    |---|---|---|
    | history + actions | a **world model** $p(x_t \mid x_{<t}, a)$ | what you will see if you take these actions |
    | history + goal map | a **video policy** $p(x_t \mid x_{<t}, m)$ | a video of an expert driving to the goal |
    | history only | an **explorer** $p(x_t \mid x_{<t})$ | a video of somebody wandering around |

    One network covers all three because, during training, we randomly **drop** each condition
    and replace it by its "nothing" token. Exploration episodes never get a map, so "no map"
    means "wander". Dropping is also what makes **classifier-free guidance** possible later: the
    same network can then predict both *with* and *without* a condition, and we can compare.

    Here is what the map looks like at a few moments of the expert's drive:
    """)
    return


@app.cell(hide_code=True)
def _(P, RES, np, plt, torch, w4):
    _env = w4.test_env(9)
    _poses, _ = w4.expert_rollout(_env)
    _ts = [0, 30, 60, 90]
    _fr = w4.render(_env["grid"], _poses[_ts], _env["goal"], size=RES)
    _m = P.env_map([_env] * len(_ts), list(_poses[_ts]), torch.device("cpu"))
    _fig, _axes = plt.subplots(2, len(_ts), figsize=(1.7 * len(_ts), 3.6))
    for _j, _t in enumerate(_ts):
        _axes[0, _j].imshow(_fr[_j], interpolation="nearest")
        _axes[0, _j].set_title(f"frame {_t}", fontsize=8)
        _axes[1, _j].imshow(P.map_rgb(_m[_j]), interpolation="nearest")
        _axes[1, _j].plot(8, 8, marker="^", color="white", ms=7, mec="black")
    for _ax in _axes.flat:
        _ax.set_xticks([])
        _ax.set_yticks([])
    _axes[0, 0].set_ylabel(f"view ({RES}×{RES})", fontsize=8)
    _axes[1, 0].set_ylabel("goal map", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the map the agent ▲ sits in the middle looking up, and the map covers 4 cells in every
    direction. Light floor is closer to the goal than where the agent stands, dark blue floor is
    farther away, walls are black, and the goal appears in pink once it is within reach of the
    map.

    ### ✏️ Exercise 1 · Dropping a condition

    For a batch of conditions $c_1, \dots, c_B$ (one row per training example), replace each row
    independently with probability $p$ by the "nothing" value:

    $$
    \tilde c_i = \begin{cases} \varnothing & \text{with probability } p, \\ c_i & \text{otherwise.} \end{cases}
    $$

    The training code calls your function three times per batch: for the actions
    (`null` = the "no action" token, $p = 0.7$ for expert drives), for the map (`null` = NaN,
    which the network reads as "no map") and for the whole history (`null` = noise level 1,
    $p = 0.1$). `p` is either a number or a tensor with one probability per row. Use
    `torch.rand(B, generator=g, device=cond.device)` and `torch.where`. Careful: one decision
    per *row*, so reshape the mask to `(B, 1, 1, ...)` to match the shape of `cond`.
    """)
    return


@app.cell
def _(torch):
    def drop(cond, null, p, g=None):
        """cond: (B, ...) tensor, one condition per row. null: a 0-dim tensor (the "nothing" value).
        p: a number or a (B,) tensor. Returns cond with each row replaced by null with probability p."""
        # ✏️ your code here
        raise NotImplementedError
    return (drop,)


@app.cell(hide_code=True)
def _(check4, drop, use_solutions):
    drop_impl = check4.pick(drop, "drop", use_solutions.value)
    ok_drop, _msg = check4.drop(drop_impl)
    _msg
    return drop_impl, ok_drop


@app.cell(hide_code=True)
def _(RES, mo, ok_drop):
    mo.stop(not ok_drop, mo.md("*Finish Exercise 1 to continue.*"))
    video_source = mo.ui.radio(["load the pretrained model", "train the small model myself"],
                               value="load the pretrained model", label="video model")
    train_steps = mo.ui.slider(1000, 10000, step=500, value=5000, label="training steps", show_value=True)
    train_go = mo.ui.run_button(label="▶ go")
    mo.vstack([
        mo.md(rf"""
        ## Part 3 · Training

        Training uses 3,000 episodes of 96 frames: one third with an explorer that ignores the
        goal, two thirds with the expert. The expert makes a random move now and then, so the
        data also shows how to recover from mistakes. Each training example is one frame with its
        4-frame history; the loss is the plain squared error of the clean-frame prediction,
        $\|\hat x - x\|^2$, with the noise level $k$ drawn mostly toward the noisy end. (That
        choice matters: when $k$ is close to 0, the noisy frame already shows where the agent
        went, and the model learns to ignore the map.)

        - **Pretrained:** {"a 3.7M-parameter model at 16×16, about 40k steps on a GPU" if RES == 16 else "a 16.8M-parameter model at 64×64, 30k steps (about 1.5 hours) on a GPU"}.
        - **Train it yourself** (16×16 only): generating the data takes about 2 minutes, and 5,000
          steps take about 10 minutes on a laptop CPU (about 3 minutes on a GPU). It will be worse
          than the pretrained one, but good enough to navigate: in our test, 5,000 steps reached
          the goal in 78% of the mazes at guidance $w = 3$ (pretrained: 97%). Saved to
          `checkpoints/mine_nb4_video16.pt` and reloaded next time (delete it to retrain).
        """),
        mo.hstack([video_source, train_steps, train_go], justify="start", gap=2),
    ])
    return train_go, train_steps, video_source


@app.cell(hide_code=True)
def _(AUTORUN, DEV, P, RES, drop_impl, mo, ok_drop, plt, torch, train_go, train_steps, video_source, w4):
    mo.stop(not ok_drop)
    mo.stop(not (train_go.value or AUTORUN), mo.md("*Press **go** to load or train the video model.*"))
    _mine = P.ROOT / "checkpoints" / "mine_nb4_video16.pt"
    _out = []
    if video_source.value.startswith("load") or RES != 16:
        if not video_source.value.startswith("load"):
            _out.append(mo.callout(mo.md("Training is only offered at 16×16. Loading the pretrained 64×64 model."), kind="warn"))
        video_model = P.load_video(RES, DEV)
        _out.append(mo.md(f"Loaded the pretrained {RES}×{RES} model ({P.n_params(video_model) / 1e6:.1f}M parameters)."))
    elif _mine.exists():
        _ck = torch.load(_mine, map_location="cpu")
        video_model = P.Denoiser(**_ck["cfg"])
        video_model.load_state_dict(_ck["model"])
        video_model = video_model.to(DEV).eval()
        _out.append(mo.md(f"Loaded your model from `{_mine.name}` (delete the file to train again)."))
    else:
        with mo.status.progress_bar(total=3000, title="rendering the training data") as _bar:
            _train_data = w4.dataset(1000, 2000, 96, 16, seed=0, procs=4,
                                     progress=lambda n, N: _bar.update(increment=1))
        _model = P.Denoiser(**P.CONFIGS[16]).to(DEV)
        with mo.status.progress_bar(total=train_steps.value, title="training the video model") as _bar:
            _done = [0]

            def _log(s, loss, secs, bar=_bar, done=_done):
                bar.update(increment=s + 1 - done[0], subtitle=f"loss {loss:.4f}, {secs:.0f}s")
                done[0] = s + 1

            video_model, _losses = P.train_video(_model, _train_data, drop_impl, steps=train_steps.value, log=_log)
        del _train_data
        torch.save({"cfg": video_model.cfg, "model": video_model.state_dict()}, _mine)
        _fig, _ax = plt.subplots(figsize=(4, 2.2))
        _ax.plot(_losses, lw=0.5)
        _ax.set_yscale("log")
        _ax.set_title("training loss", fontsize=9)
        _fig.tight_layout()
        _out += [mo.md(f"Trained and saved to `{_mine.name}`."), _fig]
    mo.vstack(_out)
    return (video_model,)


@app.cell(hide_code=True)
def _(RES, mo, w4):
    with mo.status.spinner(title="rendering test episodes..."):
        # held-out mazes: 24 explorer and 72 expert episodes, for testing and for the IDM's validation
        val = w4.dataset(24, 72, 96, RES, seed=1, procs=4)
        # training data for the inverse dynamics model
        idm_data = w4.dataset(60, 120, 96, RES, seed=2, procs=4)
    return idm_data, val


@app.cell(hide_code=True)
def _(DEV, P, RES, mo, plots, plt, torch, val, video_model):
    # Dreams: the same 4 history frames and the same actions, 4 different noise samples.
    _T = 48 if RES == 16 or DEV.type == "cuda" else 16
    _ep, _S = 35, 4  # an expert episode in a held-out maze
    _real = P.to_float(val["frames"][_ep:_ep + 1, :P.C + _T]).to(DEV)
    _acts = torch.from_numpy(val["actions"][_ep:_ep + 1, :P.C + _T]).to(DEV)
    with mo.status.spinner(title="dreaming..."):
        _dreams = P.generate(video_model, _real[:, :P.C].expand(_S, -1, -1, -1, -1), _T,
                             actions=_acts.expand(_S, -1), n_steps=10, seed=0)
    _labels = ["real"] + [f"dream {_i + 1}" for _i in range(_S)]
    _vids = [P.to_u8(_real[0, P.C:])] + [P.to_u8(_dreams[_i]) for _i in range(_S)]
    _strip = plots.video_grid(_vids, _labels, every=4, scale=0.75 if RES == 16 else 0.9)
    plt.close(_strip)
    mo.vstack([
        mo.md(r"""
        ### What the model dreams

        The model gets the first 4 frames of a held-out episode and the actions that were really
        taken, and generates the next frames one at a time, feeding its own frames back as
        history. We do this 4 times with different noise. For the first frames all 4 dreams agree
        with the real video: what is in view, and the actions, fix them. Then the agent walks into
        parts of the maze that none of the 4 frames showed, and every dream imagines its own
        walls. Each of these mazes is a plausible sample, not an error: the model has learned a
        *distribution* over what could come next.
        """),
        _strip,
        plots.video_html(_vids, _labels, size=1.8, interval=150),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4 · The inverse dynamics model

    A video does not come with actions. To act on an imagined video, we need a second, much
    simpler network that looks at two consecutive frames and tells which action leads from one
    to the other: an **inverse dynamics model** (IDM),

    $$
    p_\phi(a \mid x_t, x_{t+1}), \qquad a \in \{\text{forward}, \text{left}, \text{right}\}.
    $$

    It is an ordinary classifier, trained on real frame pairs from the data with the
    cross-entropy loss

    $$
    \mathcal L(\phi) = -\,\mathbb E\big[\log p_\phi(a_t \mid x_t, x_{t+1})\big].
    $$

    ### ✏️ Exercise 2 · The IDM loss

    `idm(frame_a, frame_b)` returns logits of shape `(B, 3)`. Return the mean cross-entropy with
    the true actions (`F.cross_entropy` does the softmax and the log for you).
    """)
    return


@app.cell
def _(torch):
    import torch.nn.functional as F

    def idm_loss(idm, frame_a, frame_b, actions):
        """frame_a, frame_b: (B, 3, r, r); actions: (B,) long. Returns a 0-dim tensor."""
        # ✏️ your code here
        raise NotImplementedError
    return (idm_loss,)


@app.cell(hide_code=True)
def _(check4, idm_loss, use_solutions):
    idm_loss_impl = check4.pick(idm_loss, "idm_loss", use_solutions.value)
    ok_idm, _msg = check4.idm_loss(idm_loss_impl)
    _msg
    return idm_loss_impl, ok_idm


@app.cell(hide_code=True)
def _(DEV, P, RES, idm_data, idm_loss_impl, mo, ok_idm, val):
    mo.stop(not ok_idm, mo.md("*Finish Exercise 2 to train the inverse dynamics model.*"))
    idm = P.IDM(RES).to(DEV)
    if RES == 16 or DEV.type == "cuda":
        _n = 1500
        with mo.status.progress_bar(total=_n, title="training the IDM") as _bar:
            _done = [0]

            def _log(s, loss, secs, bar=_bar, done=_done):
                bar.update(increment=s + 1 - done[0], subtitle=f"loss {loss:.4f}")
                done[0] = s + 1

            P.train_idm(idm, idm_data, idm_loss_impl, steps=_n, log=_log)
        _how = f"Trained for {_n} steps."
    else:
        idm = P.load_idm(RES, DEV)
        _how = "Training it at 64×64 on a CPU takes several minutes, so this is the pretrained IDM."
    mo.md(f"{_how} Accuracy on held-out mazes: **{100 * P.idm_accuracy(idm, val):.1f}%**.")
    return (idm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The IDM is nearly perfect on real frames. It has an easy job: turning shifts the whole
    picture sideways, and moving forward makes the floor tiles flow toward the camera. It will
    now grade the world model: if we ask the world model to turn left, does the IDM see a left
    turn in the generated frames?

    ## Part 5 · The world model, and how many steps to take

    We give the model the first 4 frames of a held-out episode and the actions that were really
    taken, and it generates the next frames one at a time, feeding its own frames back as
    history. Each frame is sampled with deterministic Euler steps from noise ($k = 1$) to clean
    ($k = 0$), as in notebook 2.

    **🤔 Predict first.** Everyone samples with many steps, often 10 to 50. What happens with only
    **one** Euler step? (Hint: notebook 1. What does a network trained with a squared error
    predict, when it sees only noise?)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Answer (click after you have guessed)": mo.md(r"""
    One Euler step from pure noise gives the model's prediction $\hat x = \mathbb E[x \mid \text{history}, \text{actions}]$:
    the **average** over every future the model considers possible. Where the future is
    uncertain, that average is a blur, exactly like the regression in notebook 1.

    Here, though, the near future is nearly certain: the history and the actions fix almost
    everything, except the parts of the maze that come into view for the first time (that is where
    the dreams in Part 3 went their own ways). So one step
    is nearly as good as ten, and it even scores *better* on PSNR (a squared error rewards the
    average). It looks a little flatter in the newly revealed parts, which is easiest to see at
    64×64. With fewer things fixed, as in the policy below (no actions, only a goal), there is
    more to average over.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    wm_go = mo.ui.run_button(label="▶ generate with 1, 2, 4 and 10 steps")
    wm_go
    return (wm_go,)


@app.cell(hide_code=True)
def _(AUTORUN, DEV, P, RES, idm, mo, np, plots, plt, torch, val, video_model, wm_go):
    mo.stop(not (wm_go.value or AUTORUN), mo.md("*Press the button to generate.*"))
    _B, _T = 8, (24 if RES == 16 or DEV.type == "cuda" else 12)
    _ep = np.arange(_B) * 3 + 24  # expert episodes
    _real = P.to_float(val["frames"][_ep, :P.C + _T]).to(DEV)
    _acts = torch.from_numpy(val["actions"][_ep, :P.C + _T]).to(DEV)
    wm_runs = {}
    _rows = []
    with mo.status.spinner(title="generating..."):
        for _n in (1, 2, 4, 10):
            _gen = P.generate(video_model, _real[:, :P.C], _T, actions=_acts, n_steps=_n, seed=0)
            _seq = torch.cat([_real[:, P.C - 1:P.C], _gen], 1)
            with torch.no_grad():
                _read = idm(_seq[:, :-1].flatten(0, 1), _seq[:, 1:].flatten(0, 1)).argmax(1).view(_B, _T)
            _agree = (_read == _acts[:, P.C - 1:P.C - 1 + _T]).float().mean().item()
            _ps = P.psnr(_gen, _real[:, P.C:]).mean(0).cpu().numpy()
            wm_runs[_n] = _gen
            _rows.append(f"| {_n} | {_ps[0]:.1f} | {_ps[3]:.1f} | {_ps[7]:.1f} | {_ps[_T - 1]:.1f} | {100 * _agree:.0f}% |")
    _table = mo.md("| Euler steps | PSNR, frame 1 | frame 4 | frame 8 | frame " + str(_T) +
                   " | IDM reads the commanded action |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))
    _strip = plots.video_grid([P.to_u8(_real[0, P.C:])] + [P.to_u8(wm_runs[n][0]) for n in (1, 10)],
                              ["real", "1 step", "10 steps"], every=2, scale=0.75 if RES == 16 else 0.9)
    plt.close(_strip)
    mo.vstack([
        _table,
        mo.md("PSNR (peak signal-to-noise ratio, in dB) compares a generated frame with the real one: higher is closer. "
              "After a few frames the model and reality may legitimately disagree about the parts "
              "of the maze that neither has seen yet."),
        _strip,
        plots.video_html([P.to_u8(_real[0, P.C:]), P.to_u8(wm_runs[1][0]), P.to_u8(wm_runs[10][0])],
                         ["real", "1 step", "10 steps"], size=2.0, interval=150),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The 10 Euler steps of one frame.** In each pair of rows, the top row is the noisy frame $x^k$
    and the bottom row is the model's guess of the clean frame $\hat x$ at that step. The guess in
    the first column, at $k = 1$, is exactly what one-step sampling returns.

    - **No history** (the history dropped, as in Exercise 1): the first guess is the average of
      every view the model has seen, a blur. Step by step it commits to one view, and the details
      appear.
    - **After 4 frames and their actions:** the past fixes almost everything, so the guess is the
      right frame from the first step on. That is why one step is nearly as good as ten here.
    """)
    return


@app.cell(hide_code=True)
def _(DEV, P, plots, plt, torch, val, video_model):
    _ep = 24  # the first expert episode of the held-out mazes
    _g = torch.Generator(device=DEV).manual_seed(3)
    _ctx = P.to_float(val["frames"][_ep:_ep + 1, :P.C]).to(DEV)
    _acts = torch.from_numpy(val["actions"][_ep:_ep + 1, :P.C]).long().to(DEV)
    _ones = torch.ones(1, P.C, device=DEV)
    _cases = [("no history", P.renoise(_ctx, _ones, _g), _ones, torch.full_like(_acts, P.NULL_ACT)),
              ("after 4 frames", _ctx, torch.zeros_like(_ones), _acts)]
    _rows = []
    with torch.no_grad():
        for _lab, _c, _kc, _a in _cases:
            _traj = []
            _x = P.sample_frame(video_model, _c, _kc, _a, n_steps=10, g=_g, trajectory=_traj)
            _rows.append((_lab, [(_k, P.to_u8(_xk[0]), P.to_u8(_xh[0])) for _k, _xk, _xh in _traj], P.to_u8(_x[0])))
    _fig = plots.denoising(_rows, real=[None, val["frames"][_ep, P.C]])
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    k_sliders = mo.ui.array([mo.ui.slider(0.0, 1.0, step=0.1, value=1.0, label=f"frame t−{4 - j}", show_value=True)
                             for j in range(4)], label="noise level of each history frame (0 = clean, 1 = pure noise)")
    mo.vstack([
        mo.md(r"""
        ## Part 6 · Diffusion forcing: choosing what to condition on

        Because every history frame was trained at its own noise level, we can decide at test
        time **how much the model should trust each past frame**, frame by frame. Level 0 shows
        a frame as it is, level 1 hides it completely, anything in between keeps the coarse
        layout and hides the details. The sliders start with the whole history hidden. Below
        them are 4 samples of the next 6 frames: watch how different they are, then make the
        history frames clean one by one and watch the samples agree.

        **🤔 Predict first.** (a) If all four history frames are at level 1, what will the
        model imagine? (b) If only the most recent frame $t-1$ is at 1 and the older three are
        clean, how good is the next frame?
        """),
        mo.accordion({"Answer (click after you have guessed)": mo.md(r"""
        (a) With no history the model is a plain image generator: the samples are plausible
        views of *some* maze, a different one each time, unrelated to the real episode.

        (b) Much better than nothing, but clearly worse than a clean history. Measured on 32
        held-out episodes with the pretrained 16×16 model, the first generated frame has a PSNR of
        31 dB with a clean history, 14 dB with none, and 27 dB with only frame $t-1$ hidden: the
        older frames and the actions recover most of it. Try the opposite: keep only $t-1$ and hide
        the rest (30 dB). Almost all the information is in the most recent frame, which is why
        errors in the model's own last frame (exposure bias, notebook 3) do so much damage.
        """)}),
        k_sliders.hstack(),
    ])
    return (k_sliders,)


@app.cell(hide_code=True)
def _(DEV, P, RES, k_sliders, mo, np, plt, torch, val, video_model):
    _ks = torch.tensor([s.value for s in k_sliders], dtype=torch.float32)
    _ep, _t0, _T, _K = 30, 20, 6, 4
    _real = P.to_float(val["frames"][_ep, _t0:_t0 + P.C + _T]).to(DEV)
    _acts = torch.from_numpy(val["actions"][_ep, _t0:_t0 + P.C + _T]).to(DEV)
    _g = torch.Generator(device=DEV).manual_seed(0)
    _seen = P.renoise(_real[None, :P.C], _ks[None].to(DEV), _g)[0]
    with mo.status.spinner(title="sampling..."):
        _gen = P.generate(video_model, _real[None, :P.C].expand(_K, -1, -1, -1, -1), _T,
                          k_ctx=_ks[None].expand(_K, -1).to(DEV), actions=_acts[None].expand(_K, -1), n_steps=10, seed=1)
        # the same setting on 32 episodes: how close is the first generated frame to the real one?
        _ep32 = np.arange(32) * 3
        _r32 = P.to_float(val["frames"][_ep32, 10:10 + P.C + 1]).to(DEV)
        _a32 = torch.from_numpy(val["actions"][_ep32, 10:10 + P.C + 1]).to(DEV)
        _g32 = P.generate(video_model, _r32[:, :P.C], 1, k_ctx=_ks[None].expand(32, -1).to(DEV), actions=_a32, n_steps=10, seed=2)
        _p = P.psnr(_g32[:, 0], _r32[:, P.C]).mean().item()
    _n = P.C + _T
    _fig, _axes = plt.subplots(2 + _K, _n, figsize=(0.62 * _n * (1.3 if RES == 64 else 1), 0.7 * (2 + _K) * (1.3 if RES == 64 else 1)))
    _rows = [("real", P.to_u8(_real)), ("what the\nmodel sees", list(P.to_u8(_seen)) + [None] * _T)]
    _rows += [(f"sample {i + 1}", [None] * P.C + list(P.to_u8(_gen[i]))) for i in range(_K)]
    for _r, (_lab, _imgs) in enumerate(_rows):
        for _c in range(_n):
            _ax = _axes[_r, _c]
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _imgs[_c] is not None:
                _ax.imshow(_imgs[_c], interpolation="nearest")
            else:
                _ax.axis("off")
            if _r == 0:
                _ax.set_title(f"t−{P.C - _c}" if _c < P.C else ("t" if _c == P.C else f"t+{_c - P.C}"), fontsize=7)
        _axes[_r, 0].set_ylabel(_lab, rotation=0, ha="right", va="center", fontsize=7)
        if _imgs[0] is None:
            _axes[_r, 0].axis("on")
            _axes[_r, 0].set_xticks([])
            _axes[_r, 0].set_yticks([])
            for _s in _axes[_r, 0].spines.values():
                _s.set_visible(False)
    _fig.subplots_adjust(wspace=0.05, hspace=0.08, left=0.16)
    _spread = _gen.std(0).mean().item()  # how much the 4 samples differ, per pixel
    mo.vstack([_fig, mo.md(f"The 4 samples differ by **{_spread:.2f}** per pixel on average (about 0.25 with the whole "
                           "history hidden, about 0.02 with a clean one, for the pretrained 16×16 model). "
                           f"Over 32 held-out episodes with these history levels, the first generated frame has a PSNR of **{_p:.1f} dB** "
                           "against the real one (all clean: 31 dB; all hidden: 14 dB).")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 7 · Classifier-free guidance

    Dropping conditions during training gives one network that can predict the next frame both
    with a condition $c$ and without it ($\varnothing$). **Classifier-free guidance** extrapolates
    from the second toward the first:

    $$
    \hat x_w = \hat x(\varnothing) + w\,\big(\hat x(c) - \hat x(\varnothing)\big).
    $$

    $w = 1$ is the ordinary conditional model, $w = 0$ ignores the condition, and $w > 1$
    exaggerates what the condition changes. (Usually guidance is written for the velocity $u$
    instead of $\hat x$. Here the Euler velocity is $(x - \hat x)/k$, which is linear in
    $\hat x$, so the two are the same thing.) The sampler runs both predictions as one batch and
    combines them with your function at every Euler step.

    ### ✏️ Exercise 3 · Guidance
    """)
    return


@app.cell
def _():
    def guide(x_cond, x_uncond, w):
        """Clean-frame predictions with and without the condition (same shape); w: a number."""
        # ✏️ your code here
        raise NotImplementedError
    return (guide,)


@app.cell(hide_code=True)
def _(check4, guide, use_solutions):
    guide_impl = check4.pick(guide, "guide", use_solutions.value)
    ok_guide, _msg = check4.guide(guide_impl)
    _msg
    return guide_impl, ok_guide


@app.cell(hide_code=True)
def _(mo, ok_guide):
    mo.stop(not ok_guide, mo.md("*Finish Exercise 3 to continue.*"))
    w_hist = mo.ui.slider(1.0, 4.0, step=0.5, value=3.0, label="history guidance w", show_value=True)
    hist_go = mo.ui.run_button(label="▶ generate 40 frames")
    mo.vstack([
        mo.md(r"""
        ### History guidance

        The first condition to guide is the **history**. The unconditional prediction gets every
        history frame at level 1 (it never sees the past) but still knows the actions. With
        $w > 1$ the model should lean harder on what the past frames tell it, and long videos
        should stay more consistent. Large video models that tend to forget their past use this
        (Song et al., *History-Guided Video Diffusion*, 2025).

        **🤔 Predict first.** Will a large history weight, say $w = 3$, make the 40-frame videos
        closer to the real ones?
        """),
        mo.accordion({"Answer (click after you have guessed)": mo.md(r"""
        No, it makes them worse. Our model already uses its 4 history frames well: they enter
        the network directly, as input channels. So there is little to gain, and $w = 1.5$ is about
        the same as $w = 1$. Larger weights extrapolate past anything the model saw in training:
        every difference between "with history" and "without" gets exaggerated, and the frames
        drift and lose contrast. On 24 held-out episodes, the PSNR over frames 20–40 was 15.2 dB
        at $w = 1$, 15.0 at $w = 1.5$, 14.3 at $w = 2$ and 12.8 at $w = 3$. Guidance is a strong
        medicine: use as little as works.
        """)}),
        mo.hstack([w_hist, hist_go], justify="start", gap=2),
    ])
    return hist_go, w_hist


@app.cell(hide_code=True)
def _(AUTORUN, DEV, P, RES, guide_impl, hist_go, idm, mo, np, ok_guide, plots, plt, torch, val, video_model, w_hist):
    mo.stop(not ok_guide)
    mo.stop(not (hist_go.value or AUTORUN), mo.md("*Press the button to generate.*"))
    _B, _T = (16, 40) if RES == 16 or DEV.type == "cuda" else (4, 16)
    _ep = np.arange(_B) * 4
    _real = P.to_float(val["frames"][_ep, :P.C + _T]).to(DEV)
    _acts = torch.from_numpy(val["actions"][_ep, :P.C + _T]).to(DEV)
    _res = {}
    with mo.status.spinner(title="generating..."):
        for _w in sorted({1.0, w_hist.value}):
            _res[_w] = P.generate(video_model, _real[:, :P.C], _T, actions=_acts, w_hist=_w, n_steps=4, guide=guide_impl, seed=0)
    _fig, _ax = plt.subplots(figsize=(6.5, 2.6))
    for _w, _g in _res.items():
        _seq = torch.cat([_real[:, P.C - 1:P.C], _g], 1)
        with torch.no_grad():
            _rd = idm(_seq[:, :-1].flatten(0, 1), _seq[:, 1:].flatten(0, 1)).argmax(1).view(_B, _T)
        _ag = (_rd == _acts[:, P.C - 1:P.C - 1 + _T]).float().mean().item()
        _ax.plot(np.arange(1, _T + 1), P.psnr(_g, _real[:, P.C:]).mean(0).cpu().numpy(),
                 label=f"w = {_w:g} (IDM reads the commanded action {100 * _ag:.0f}%)")
    _ax.set_xlabel("frames generated")
    _ax.set_ylabel("PSNR (dB)")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _rows = [P.to_u8(_real[0, P.C:])] + [P.to_u8(_g[0]) for _g in _res.values()]
    _labs = ["real"] + [f"w = {_w:g}" for _w in _res]
    _strip = plots.video_grid(_rows, _labs, every=4, scale=0.75 if RES == 16 else 0.9)
    plt.close(_strip)
    mo.vstack([_fig, _strip, plots.video_html(_rows, _labs, size=2.0, interval=120)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 8 · Goal guidance

    Now the policy: no actions, and the condition is the goal map. The unconditional prediction
    has no map, so it is the explorer. Guidance pushes the imagined video away from "wander
    somewhere" and toward "drive to the goal".

    To see what the model wants to do, we take moments where the expert had just driven
    straight and was about to turn left, go straight on, or turn right, imagine 8 frames, and
    let the IDM read the actions off the imagined frames. A net turn of at least 3 steps to one
    side counts as a turn.

    **🤔 Predict first.** At $w = 0$, how often will the imagined video turn the same way as the
    expert? And at $w = 3$?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Answer (click after you have guessed)": mo.md(r"""
    At $w = 0$ the model has no idea where the goal is. It turns where the corridor turns and
    otherwise goes straight on, so it agrees with the expert only by chance, about half the time
    (mostly in corridors where there is only one way to go). With guidance it picks the expert's
    direction most of the time: 51% → 72% → 80% at $w$ = 0, 1, 3 for the pretrained 16×16 model
    (72 turning points × 16 imagined futures each). Large weights help less and less, and push
    the frames away from what the model has learned.
    """)})
    return


@app.cell(hide_code=True)
def _(mo, ok_guide):
    mo.stop(not ok_guide)
    dir_ws = mo.ui.multiselect(["0", "0.5", "1", "2", "3", "5"], value=["0", "1", "3"], label="weights w")
    dir_steps = mo.ui.dropdown(["1", "4", "10"], value="4", label="Euler steps")
    dir_go = mo.ui.run_button(label="▶ imagine")
    mo.hstack([dir_ws, dir_steps, dir_go], justify="start", gap=2)
    return dir_go, dir_steps, dir_ws


@app.cell(hide_code=True)
def _(AUTORUN, DEV, P, RES, dir_go, dir_steps, dir_ws, guide_impl, idm, mo, np, ok_guide, plt, torch, val, video_model):
    mo.stop(not ok_guide)
    mo.stop(not (dir_go.value or AUTORUN), mo.md("*Press **imagine**.*"))
    _states = P.turn_onsets(val, per_class=12 if RES == 16 or DEV.type == "cuda" else 4)
    _ctx, _m = P.state_inputs(val, _states, DEV)
    _truth = np.array([s[2] for s in _states])
    _K = 4
    _ws = sorted(float(w) for w in dir_ws.value) or [1.0]
    dir_results, dir_examples = {}, {}
    with mo.status.progress_bar(total=len(_ws), title="imagining") as _bar:
        for _w in _ws:
            _fut = P.imagine(video_model, _ctx.repeat_interleave(_K, 0), None, _m.repeat_interleave(_K, 0), _w,
                             int(dir_steps.value), 8, guide_impl, seed=0)
            _cls = P.net_turn_class(idm, _ctx[:, -1].repeat_interleave(_K, 0), _fut).view(len(_states), _K).cpu().numpy()
            dir_results[_w] = _cls
            dir_examples[_w] = P.to_u8(_fut.view(len(_states), _K, *_fut.shape[1:])[:, 0])
            _bar.update()
    _names = ["left", "straight", "right"]
    _cols = ["#4c6ef5", "#adb5bd", "#f76707"]
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 2.9), sharey=True)
    for _c in range(3):
        _ax = _axes[_c]
        _bottom = np.zeros(len(_ws))
        for _j in range(3):
            _frac = np.array([(dir_results[w][_truth == _c] == _j).mean() for w in _ws])
            _ax.bar([f"{w:g}" for w in _ws], _frac, bottom=_bottom, color=_cols[_j], label=f"imagined: {_names[_j]}")
            _bottom += _frac
        _ax.set_title(f"the expert turned {_names[_c]}" if _c != 1 else "the expert went straight", fontsize=9)
        _ax.set_xlabel("guidance weight w")
    _axes[0].set_ylabel("fraction of imagined futures")
    _h, _l = _axes[0].get_legend_handles_labels()
    _fig.legend(_h, _l, fontsize=7, loc="lower center", ncol=3, frameon=False)
    _fig.tight_layout(rect=(0, 0.08, 1, 1))
    _match = " · ".join(f"w = {w:g}: **{100 * (dir_results[w] == _truth[:, None]).mean():.0f}%**" for w in _ws)
    mo.vstack([_fig, mo.md(f"Imagined futures that go the expert's way ({len(_states)} turning points × {_K} futures): {_match}")])
    return dir_examples, dir_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 9 · Closing the loop

    Now the agent drives on its own, in mazes it has never seen. Every 4 steps:

    ```
    frames  = the last 4 real frames                          # what the agent has seen
    m       = goal map at the current position
    future  = imagine(frames, m, w)                          # 8 frames, with goal guidance w
    actions = read_actions(idm, [frames[-1]] + future)        # Exercise 4
    execute actions[:4] in the real maze, render what the agent now sees, repeat
    ```

    ### ✏️ Exercise 4 · Reading the plan

    `frames` has shape `(B, H+1, 3, r, r)`: the last real frame followed by $H$ imagined ones.
    Return the $H$ actions `(B, H)` that the IDM reads off the consecutive pairs
    $(x_j, x_{j+1})$, taking the most likely action for each. Do all pairs in one call:
    flatten the batch and time dimensions into one with `.flatten(0, 1)`, and reshape back at
    the end.
    """)
    return


@app.cell
def _():
    def read_actions(idm, frames):
        """frames: (B, H+1, 3, r, r). Returns (B, H) long: the action from frame j to frame j+1."""
        # ✏️ your code here
        raise NotImplementedError
    return (read_actions,)


@app.cell(hide_code=True)
def _(check4, read_actions, use_solutions):
    read_actions_impl = check4.pick(read_actions, "read_actions", use_solutions.value)
    ok_read, _msg = check4.read_actions(read_actions_impl)
    _msg
    return ok_read, read_actions_impl


@app.cell(hide_code=True)
def _(mo, ok_guide, ok_read):
    mo.stop(not (ok_read and ok_guide), mo.md("*Finish Exercises 3 and 4 to drive.*"))
    cl_w = mo.ui.slider(0.0, 5.0, step=0.5, value=3.0, label="goal guidance w", show_value=True)
    cl_steps = mo.ui.dropdown(["1", "4", "10"], value="1", label="Euler steps")
    cl_n = mo.ui.dropdown(["4", "8", "16", "32"], value="8", label="mazes")
    cl_go = mo.ui.run_button(label="▶ drive")
    mo.vstack([
        mo.md(r"""
        **🤔 Predict first.** (a) Will the agent reach the goal at $w = 0$? At $w = 1$? (b) In
        Part 5, one Euler step was fine for the world model. Is it also fine for the policy,
        where the future is much less certain?
        """),
        mo.accordion({"Answer (click after you have guessed)": mo.md(r"""
        (a) At $w = 0$ the agent wanders like the explorer and finds the goal only by luck
        (about 10–20% of mazes within 400 steps). Guidance turns it into a navigator: on 64
        held-out mazes the pretrained 16×16 model reached the goal in 86% of them at $w = 1$
        and 92% at $w = 3$; the 64×64 model 77% and 97%.

        (b) Surprisingly, yes: with $w = 3$, one step reached the goal *more* often (16×16:
        97%, 64×64: 100%) and on shorter paths than ten steps. A one-step plan is the average of
        the futures the model considers; at a junction that is a blurry mix of "left" and
        "right". But strong guidance has already made one direction much more likely than the
        others, and the agent replans every 4 steps anyway, so an average plan that roughly
        points the right way is all it needs. Without guidance there is no favored direction,
        and neither setting finds the goal.
        """)}),
        mo.hstack([cl_w, cl_steps, cl_n, cl_go], justify="start", gap=2),
    ])
    return cl_go, cl_n, cl_steps, cl_w


@app.cell(hide_code=True)
def _(AUTORUN, P, cl_go, cl_n, cl_steps, cl_w, guide_impl, idm, mo, np, ok_guide, ok_read, read_actions_impl, video_model, w4):
    mo.stop(not (ok_read and ok_guide))
    mo.stop(not (cl_go.value or AUTORUN), mo.md("*Press **drive**.*"))
    _envs = [w4.test_env(1000 + i) for i in range(int(cl_n.value))]
    with mo.status.progress_bar(total=len(_envs), title="driving") as _bar:
        _done = [0]

        def _prog(n_done, n, steps, bar=_bar, done=_done):
            bar.update(increment=n_done - done[0], subtitle=f"{steps} steps")
            done[0] = n_done

        cl_runs = P.navigate(_envs, video_model, idm, read_actions_impl, w=cl_w.value, n_steps=int(cl_steps.value),
                             guide=guide_impl, record=len(_envs), progress=_prog)
    _succ = np.array([r["success"] for r in cl_runs])
    _ratio = [len(r["actions"]) / r["env"]["n_expert"] for r in cl_runs if r["success"]]
    _rows = "\n".join(f"| {i} | {'✓' if r['success'] else '✗'} | {len(r['actions'])} | {r['env']['n_expert']} |"
                      for i, r in enumerate(cl_runs))
    mo.vstack([
        mo.md(f"**Reached the goal in {_succ.sum()} of {len(_succ)} mazes** (w = {cl_w.value:g}, {cl_steps.value} Euler step(s))"
              + (f"; successful drives took {np.mean(_ratio):.2f}× as many steps as the expert." if _ratio else ".")),
        mo.accordion({"per maze": mo.md("| maze | reached | steps | expert's steps |\n|---|---|---|---|\n" + _rows)}),
    ])
    return (cl_runs,)


@app.cell(hide_code=True)
def _(cl_runs, mo):
    cl_pick = mo.ui.dropdown({f"maze {i} ({'reached' if r['success'] else 'not reached'})": i for i, r in enumerate(cl_runs)},
                             value=f"maze 0 ({'reached' if cl_runs[0]['success'] else 'not reached'})", label="show")
    cl_pick
    return (cl_pick,)


@app.cell(hide_code=True)
def _(P, cl_pick, cl_runs, mo):
    _r = cl_runs[cl_pick.value]
    _cap = f" · {'reached the goal' if _r['success'] else 'did not reach the goal'} in {len(_r['actions'])} steps (expert: {_r['env']['n_expert']})"
    mo.vstack([
        mo.md("The pink dashed line on the map is the current plan: the actions the IDM read off "
              "the imagined frames, replayed from where the plan was made. The strip shows the "
              "imagined frames themselves."),
        P.show_viewer(P.viewer_html(_r["env"], _r["actions"], _r["plans"], caption=_cap)),
    ])
    return


@app.cell(hide_code=True)
def _(P, RES, mo):
    _ref = P.reference()[str(RES)]["closed_loop"]
    _r = lambda k: f"{100 * _ref[k]['success']:.0f}%" if k in _ref else "–"
    _s1 = "steps1" if "steps1" in _ref else "steps1_w3"
    mo.md(f"""
    For comparison, the pretrained {RES}×{RES} model on 64 held-out mazes (400-step budget):

    | goal guidance w | 0 | 1 | 3 | 5 |
    |---|---|---|---|---|
    | 10 Euler steps | {_r('w0')} | {_r('w1')} | {_r('w3')} | {_r('w5')} |
    | 1 Euler step | {_r('steps1_w0')} | {_r('steps1_w1')} | {_r(_s1)} | – |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we left out

    - **Real video models are latent transformers.** They compress frames with an autoencoder
      and denoise the latents with a transformer over space and time, trained on millions of
      videos. Ours denoises pixels with a small U-Net, which is enough for a world of colored
      blocks.
    - **The map makes the task easy.** Our map tells the agent which neighboring cell is closer
      to the goal, so the video policy mostly has to stay in the corridor and turn at the right
      place. A harder version would give only a picture of the goal, so that the agent has to
      search for it.
    - **The policy needs no recorded actions.** The video policy learns from videos of experts
      plus the goal, and the inverse dynamics model, trained on a few labeled frame pairs,
      supplies the actions. That is why video pretraining might matter for robots: there is far
      more video of people doing things than recorded robot actions.
    """)
    return


if __name__ == "__main__":
    app.run()
