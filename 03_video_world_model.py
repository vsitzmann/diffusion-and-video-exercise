import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="3 · A tiny world model")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import time
    from pathlib import Path
    import mazeworld as mw
    from course import plots, check, video as V
    torch.set_num_threads(4)
    ROOT = Path(__file__).parent
    return ROOT, V, check, mo, mw, np, os, plots, plt, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3 · A tiny world model

    Now we train a video model of a (very) small world on your laptop, conditioned on actions:
    a *world model*. Then we let it dream long videos on its own and watch what goes wrong. This
    is the exposure-bias problem from the lecture, and the fix (diffusion forcing) comes down to
    a few lines of code.

    The model generates a video one frame at a time, like an LLM generates text one token at a time:

    $$
    p(x_1, \dots, x_T \mid a_{1:T}) = \prod_t p(x_{t+1} \mid x_{\le t},\, a_{\le t}),
    $$

    but each factor is a continuous distribution over images, so each one is a flow-matching
    model as in notebook 2. Time: about 25 minutes, of which 5–10 are training. Same rules:
    ✏️ marks an exercise, and the switch uses the reference solutions.
    """)
    return


@app.cell(hide_code=True)
def _(check):
    use_solutions = check.solutions_switch()
    use_solutions
    return (use_solutions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 · The world

    A random maze built as a small museum: brick walls with paintings, lamps on the ceiling. A
    tiny raycaster (the technique behind Wolfenstein 3D) draws what a camera sees, and the model
    you train sees it at 16×16 pixels. An explorer walks through the maze with three actions:
    **forward**, **turn left**, **turn right**. Every episode is a new random maze with its own
    paintings, so the model can't memorize a map: when it turns a corner, it has to *imagine* what
    comes next.
    """)
    return


@app.cell(hide_code=True)
def _(mo, mw, np, plots, time):
    _ep = mw.episode(seed=7, length=48, size=16)
    _ep32 = mw.episode(seed=7, length=48, size=32)  # what the pretrained big model sees
    _big = mw.episode(seed=7, length=48, size=64)  # the same episode at 64x64, for reference
    _maps = [mw.topdown(_ep["grid"], _ep["poses"][: t + 1]) for t in range(48)]
    _t = time.time()
    _d = mw.dataset(100, 32, 16, seed=1)
    _fps = 100 * 32 / (time.time() - _t)
    mo.vstack([
        mo.md(f"Generating episodes runs at about **{_fps:,.0f} frames/s** here (including the maze "
              "generation and the explorer's choices)."),
        plots.video_html([_ep["frames"], _ep32["frames"], _big["frames"]],
                         ["your model sees 16×16", "the big pretrained model: 32×32", "the same at 64×64"],
                         extra=_maps, size=2.0, interval=150),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 · The model: a conv-RNN plus a denoiser

    Two small convolutional networks:

    - a **conv-RNN** (a ConvGRU) reads the past frames and actions, one at a time, and keeps a
      memory $h_t$ (an 8×8 feature map): $\;h_t = \mathrm{GRU}\big(h_{t-1},\ \mathrm{Enc}(\tilde x_t),\ a_t\big)$.
    - a **denoiser** (a small U-Net) predicts the clean next frame from a noisy one, given the
      memory and the previous frame: $\;\hat x_{t+1} = \mathrm{Den}\big(x^{k}_{t+1},\ k \mid h_t,\ \tilde x_t\big)$.

    Noise levels work as in notebook 2, with $k$ playing the role of the time $t$ there (the letter
    $t$ is now the frame index): a frame at level $k$ is

    $$
    x^{k} = (1-k)\, x + k\, \epsilon, \qquad k = 0: \text{clean}, \quad k = 1: \text{pure noise}.
    $$

    The key detail: **every frame has its own noise level**, and there are two versions of each
    frame: the one the denoiser must clean up (level $k_t$), and the one the RNN *reads* as
    context, $\tilde x_t$ (level $\tilde k_t$, made with the same $\epsilon$). Training minimizes
    $\sum_t \|\hat x_t - x_t\|^2$ over all frames of a 16-frame clip at once.
    """)
    return


@app.cell(hide_code=True)
def _(plots):
    plots.model_diagram()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 · Three ways to train the same network

    The architecture and the loss are identical. The only difference is **which noise levels
    we draw** for a training clip of $T$ frames:

    | regime | target levels $k_t$ | context levels $\tilde k_t$ | intuition |
    |---|---|---|---|
    | **full sequence** | one $k$ for the whole clip | $=k$ | treat the clip like one big image |
    | **teacher forcing** | independent $k_t \sim U[0,1]$ | $0$ (clean) | like an LLM: always condition on the *true* past |
    | **diffusion forcing** | independent $k_t \sim U[0,1]$ | $=k_t$ | every frame at its own level, past frames are noisy too |

    ### ✏️ Exercise 1

    Return the two `(B, T)` tensors `(k, k_ctx)`. Use `torch.rand(..., generator=g)` so runs are
    reproducible. For "full sequence", draw one level per clip with shape `(B, 1)` and
    `.expand(B, T)` it.
    """)
    return


@app.cell
def _(torch):
    def noise_levels(regime, B, T, g=None):
        """regime: "full sequence", "teacher forcing" or "diffusion forcing".
        Returns (k, k_ctx), both of shape (B, T) with values in [0, 1]."""
        if regime == "full sequence":
            # ✏️ your code here
            raise NotImplementedError
        if regime == "teacher forcing":
            # ✏️ your code here
            raise NotImplementedError
        if regime == "diffusion forcing":
            # ✏️ your code here
            raise NotImplementedError
        raise ValueError(regime)
    return (noise_levels,)


@app.cell(hide_code=True)
def _(check, noise_levels, use_solutions):
    noise_levels_impl = check.pick(noise_levels, "noise_levels", use_solutions.value)
    ok_levels, _msg = check.noise_levels(noise_levels_impl)
    _msg
    return noise_levels_impl, ok_levels


@app.cell(hide_code=True)
def _(mo, ok_levels):
    mo.stop(not ok_levels, mo.md("*Finish Exercise 1 to continue.*"))
    steps = mo.ui.slider(300, 4000, step=100, value=1500, label="training steps per model", show_value=True)
    source = mo.ui.radio(["train them myself", "load the pretrained checkpoints"], value="train them myself",
                         label="models")
    train_button = mo.ui.run_button(label="▶ go")
    mo.vstack([
        mo.md(r"""
        ## Part 4 · Training

        Training each model takes about 2–3 minutes per 1500 steps on a laptop CPU (3 models).
        The models are saved in `checkpoints/mine_*.pt`, and later runs reload them instead of
        training again (delete the files to retrain). If you are short on time, load the
        pretrained checkpoints, which come from the same code.
        """),
        mo.hstack([source, steps, train_button], justify="start", gap=2),
    ])
    return source, steps, train_button


@app.cell(hide_code=True)
def _(ROOT, V, mo, mw, noise_levels_impl, ok_levels, os, source, steps, torch, train_button):
    mo.stop(not ok_levels)
    mo.stop(not (train_button.value or os.environ.get("QC_AUTORUN") == "1"),
            mo.md("*Press **go** to train (or load) the three models.*"))
    REGIMES = ["full sequence", "teacher forcing", "diffusion forcing"]
    models, losses = {}, {}
    _data = None
    for _r in REGIMES:
        _name = _r.replace(" ", "_")
        _pre = ROOT / "checkpoints" / f"{_name}.pt"
        _mine = ROOT / "checkpoints" / f"mine_{_name}.pt"
        _m = V.VideoDenoiser()
        if source.value.startswith("load"):
            _m.load_state_dict(torch.load(_pre))
        elif _mine.exists():
            _m.load_state_dict(torch.load(_mine))
        else:
            if _data is None:
                with mo.status.spinner(title="rendering 2,000 training episodes (about a minute)..."):
                    _data = mw.dataset(2000, 32, 16, seed=0)
            with mo.status.progress_bar(total=steps.value, title=f"training: {_r}") as _bar:
                _done = [0]

                def _log(s, loss, secs, bar=_bar, done=_done):
                    bar.update(increment=s + 1 - done[0], subtitle=f"loss {loss:.4f}, {secs:.0f}s")
                    done[0] = s + 1

                losses[_r] = V.train_video(_m, lambda B, T, g, r=_r: noise_levels_impl(r, B, T, g), _data,
                                           steps=steps.value, log=_log)
            _mine.parent.mkdir(exist_ok=True)
            torch.save(_m.state_dict(), _mine)
        _m.eval()
        models[_r] = _m
    mo.md("Models ready: " + ", ".join(f"**{r}**" for r in REGIMES) + ".")
    return REGIMES, models


@app.cell(hide_code=True)
def _(ROOT, V, torch):
    # the big pretrained diffusion-forcing model (32x32): same code, wider, trained on a GPU
    big_df = V.VideoDenoiser(**V.BIG)
    big_df.load_state_dict(torch.load(ROOT / "checkpoints" / "nb3_big_diffusion_forcing.pt"))
    _ = big_df.eval()  # (assigned, so that marimo does not print the network)
    return (big_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How one frame is generated

    Every frame is sampled as in notebook 2: start from pure noise at level $k = 1$ and take Euler
    steps down to $k = 0$. Here are the 10 steps for one frame, from the big pretrained
    diffusion-forcing model (32×32, more about it in Part 5). In each pair of rows, the top row is
    the noisy frame $x^k$, and the bottom row is the model's guess of the clean frame $\hat x$ at
    that step, which sets the direction of the step.

    - **No history:** the very first frame of a video. At $k = 1$ the model knows nothing, so its
      guess is the average of all views it has seen: a gray blur. Each step removes some noise,
      the guess commits to a corridor, and the details appear.
    - **After 4 frames:** the frame after 4 real frames and their actions. The past already fixes
      almost everything, so the guess is the right frame from the first step on, and denoising
      only fills in the fine details.
    """)
    return


@app.cell(hide_code=True)
def _(V, big_df, mw, plots, plt, torch):
    _ep = mw.episode(99 * 1_000_003 + 1, 8, 32)  # test episode 1, rendered at 32x32
    _ctx = V.to_tensor(_ep["frames"])[None]
    _acts = torch.as_tensor(_ep["actions"])[None]
    _rows = []
    for _lab, _C in [("no history", 0), ("after 4 frames", 4)]:
        _traj, _x = V.denoise_next_frame(big_df, _ctx[:, :_C], _acts, seed=3)
        _rows.append((_lab, [(_k, V.to_uint8(_a[0]), V.to_uint8(_b[0])) for _k, _a, _b in _traj], V.to_uint8(_x[0])))
    _fig = plots.denoising(_rows, real=[None, _ep["frames"][4]])
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5 · Letting the models dream

    We give each model the first 4 frames of a test episode and the explorer's actions, and let it
    generate the rest on its own, feeding every generated frame back in as context. The
    generated frames are never perfect. The question is what the model does with its own small
    mistakes.

    - **Teacher forcing** and **diffusion forcing** generate one frame at a time. Before a finished
      frame is fed back, we can **re-noise it to a level $\tilde k$** (slightly above 0).
    - **Full sequence** can't read clean past frames at all: it was only trained with all frames
      at the same level. It generates 8 frames at a time and conditions on the last 8 frames by
      re-noising them to the current level at every denoising step (the "replacement" trick).
      It gets 8 real frames to start from instead of 4.

    ### ✏️ Exercise 2

    Implement the re-noising of a generated frame, $\tilde x = (1 - \tilde k)\, x + \tilde k\,\epsilon$,
    with fresh noise $\epsilon$ (use `torch.randn(x.shape, generator=g)`).
    """)
    return


@app.cell
def _(torch):
    def renoise(x, k_ctx, g=None):
        """Re-noise a generated frame x (B, 3, 16, 16) to level k_ctx (a number in [0, 1])."""
        # ✏️ your code here
        raise NotImplementedError
    return (renoise,)


@app.cell(hide_code=True)
def _(check, renoise, use_solutions):
    renoise_impl = check.pick(renoise, "renoise", use_solutions.value)
    ok_renoise, _msg = check.renoise(renoise_impl)
    _msg
    return ok_renoise, renoise_impl


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.callout(mo.vstack([
            mo.md(r"""
            **🤔 Predict first.** All three models feed their own frames back, with clean context
            ($\tilde k = 0$) where they can. Which of the three training regimes will break first in
            a long rollout of 200 frames?
            """),
            mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
            **Teacher forcing**, the way LLMs are trained. During training it only ever read
            *perfect* past frames. In the rollout it reads its own frames, which are slightly off,
            and its errors compound. Diffusion forcing stays stable. Full sequence does not blow up,
            but it jumps every 8 frames. The plots below show all three.
            """)}),
        ]), kind="info"),
        mo.callout(mo.vstack([
            mo.md(r"""
            **🤔 Predict first.** Now we re-noise each generated frame to $\tilde k = 0.1$ before the
            teacher-forced model reads it. This model never saw a noisy past frame during training.
            Will the noise make its rollouts better or worse? And what if you raise $\tilde k$ to 0.5?
            """),
            mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
            **Better.** With $\tilde k = 0.1$ almost all broken frames disappear. The noise washes out
            the fine details in which generated and real frames differ, so the model's own frames
            look like training inputs again.

            More noise does not help more. In our tests with $\tilde k = 0.5$ the frames hardly
            break either, but they look less like real frames and change more from one frame to the
            next than real video does: the noise now also destroys information about the past that
            the model needs. Try it with the slider below.
            """)}),
        ]), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    length = mo.ui.slider(40, 300, step=20, value=200, label="frames to generate", show_value=True)
    k_ctx = mo.ui.slider(0.01, 0.5, step=0.01, value=0.1, label="context level k̃ for 're-noised'", show_value=True)
    mo.hstack([length, k_ctx], justify="start", gap=2)
    return k_ctx, length


@app.cell(hide_code=True)
def _(V, mw, torch):
    # 24 test episodes (new mazes) and a bank of real frames to judge realism against
    N_TEST = 24
    test_set = mw.dataset(N_TEST, 304, 16, seed=99)
    test_ctx = V.to_tensor(test_set["frames"][:, :8])
    test_actions = torch.as_tensor(test_set["actions"]).long()
    bank = V.frame_bank()
    BROKEN = 0.2  # a frame counts as broken above this distance to the nearest real frame
    return BROKEN, bank, test_actions, test_ctx, test_set


@app.cell(hide_code=True)
def _(V, k_ctx, length, mo, models, ok_renoise, renoise_impl, test_actions, test_ctx):
    mo.stop(not ok_renoise, mo.md("*Finish Exercise 2 to run the rollouts.*"))
    _n = length.value
    _k = k_ctx.value
    with mo.status.spinner(title="generating videos..."):
        rollouts = {
            "full sequence": V.rollout_full_sequence(models["full sequence"], test_ctx, test_actions, _n - 4,
                                                     chunk=8),
        }
        for _r in ["teacher forcing", "diffusion forcing"]:
            for _label, _kk in [("clean", 0.0), (f"re-noised {_k:.2f}", _k)]:
                rollouts[f"{_r} · {_label}"], _ = V.rollout(models[_r], test_ctx[:, :4], test_actions, _n,
                                                           k_ctx=_kk, renoise=renoise_impl)
    return (rollouts,)


@app.cell(hide_code=True)
def _(BROKEN, V, bank, mo, plots, rollouts, test_set):
    _real = V.realism(V.to_tensor(test_set["frames"][:, : next(iter(rollouts.values())).shape[1]]), bank)
    _thr = BROKEN
    _fig = plots.rollout_curves({_k: V.realism(_o, bank).numpy() for _k, _o in rollouts.items()},
                                _real.numpy(), _thr, "your models (16×16)")
    _show = ["full sequence", "teacher forcing · clean", [k for k in rollouts if k.startswith("diffusion forcing · re")][0]]
    # show the episode where teacher forcing breaks the most
    _ep = int((V.realism(rollouts["teacher forcing · clean"], bank) > _thr).sum(1).argmax())
    _vids = [V.to_uint8(rollouts[k][_ep]) for k in _show]
    _strip_rows = [test_set["frames"][_ep][: len(_vids[0])]] + _vids
    mo.vstack([
        _fig,
        mo.md(f"**Episode {_ep}** (the one where teacher forcing breaks the most), as a film strip "
              "(every 8th frame) and as a video. The real video only shares the first frames; after "
              "that each model imagines its own maze, driven by the same actions."),
        plots.video_grid(_strip_rows, ["real"] + _show, every=8),
        plots.video_html(_vids, _show, size=2.0, interval=80),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same experiment with a bigger model

    Your models are tiny: about 0.3 million parameters, trained for a few minutes on a CPU, and
    they see 16×16 pixels. We trained the same architecture, wider (about 2.9 million
    parameters), at 32×32, on a GPU for 20,000 steps, with exactly the code you just ran:
    `V.train_video` with the three noise-level regimes of Exercise 1. Rolling it out on a laptop
    CPU would take a while, so here are its results on 32 test episodes (the 24 above and 8 more),
    computed in advance. The checkpoints are in `checkpoints/nb3_big_*.pt`, and the bonus at the end lets you
    play with one of them.
    """)
    return


@app.cell(hide_code=True)
def _(BROKEN, ROOT, mo, np, plots):
    _r = np.load(ROOT / "checkpoints" / "nb3_big_results.npz")
    _names = [str(n) for n in _r["names"]]
    _fig = plots.rollout_curves({n: _r[f"dist_{i}"] for i, n in enumerate(_names)}, _r["real_dist"],
                                BROKEN, "the big model (32×32), computed in advance")
    _show = [0, 1, 2, 4]  # full sequence, teacher forcing (clean, re-noised), diffusion forcing (re-noised)
    _j = 1  # of the saved episodes (the three where teacher forcing breaks most), the clearest one
    _vids = [_r[f"video_{i}"][_j] for i in _show]
    _ep = int(_r["keep"][_j])
    mo.vstack([
        _fig,
        mo.md(f"**Episode {_ep}**, as a film strip (every 8th frame) and as a video. Watch the "
              "teacher-forced model with re-noised context: the curves above miss what goes wrong there."),
        plots.video_grid([_r["real_frames"][_j]] + _vids, ["real"] + [_names[i] for i in _show], every=8),
        plots.video_html(_vids, [_names[i] for i in _show], size=2.2, interval=80),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - **Teacher forcing with clean context falls apart**, the small model and the big one. The
      fraction of broken frames keeps growing, the picture smears, and the motion slows down.
      This is **exposure bias**. The camera moves slowly, so every frame is mostly a copy of the previous one, and
      the teacher-forced model learned to copy fine details from a *perfect* past frame. At test
      time it sees its *own* frames, which are slightly off. It copies the errors, adds new ones,
      and the errors compound.
    - **Diffusion forcing is stable.** It was trained with past frames at every noise level, so it
      learned that the past is unreliable in its details, and it leans on what it knows about the
      world instead.
    - **A tiny bit of noise on the past frames rescues your small teacher-forced model.** The
      noise washes out exactly the fine details where generated and real frames differ, so the
      model's own frames look like training inputs again. Try the $\tilde k$ slider. **The big
      model is different:** it never saw a noisy past frame either, and it is sharp enough to
      copy the noise itself into the next frame, as blue blotches. (Our distance measure compares
      8×8 thumbnails and misses them; the video shows them.) Noise on the past has to be part of
      *training*, and that is diffusion forcing. Game world models like GameNGen add exactly this
      kind of noise to the past frames during training ("noise augmentation").
    - **Full sequence** doesn't blow up, but look at the jumps every 8 frames. Re-noising the
      past to the current level destroys most of the information in it, so each new chunk is only
      loosely tied to what came before.

    How strongly teacher forcing breaks varies from one training run to the next. In our tests
    (32 test episodes, 200 frames each), teacher forcing with clean context had 12% broken frames
    in the last quarter of the video (small model) and 11% (big model), and broke in a fifth to a
    quarter of the episodes; diffusion forcing had 1–3%. Diffusion forcing is not perfect either:
    now and then it gets stuck in front of a wall or a painting.

    These are exactly the problems that methods like diffusion forcing and self forcing address in
    today's large video world models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus · Play inside your world model

    A diffusion-forcing model, driven by *you*. Every click generates the next frame from the
    model's memory. There is no game engine underneath: the maze only exists in the model's
    imagination.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    play_model = mo.ui.radio(["the big pretrained model (32×32)", "your diffusion-forcing model (16×16)"],
                             value="the big pretrained model (32×32)", label="play with")
    play_model
    return (play_model,)


@app.cell(hide_code=True)
def _(V, big_df, k_ctx, mo, models, mw, ok_renoise, play_model, renoise_impl, test_actions, test_ctx, torch):
    mo.stop(not ok_renoise)
    if play_model.value.startswith("the big"):
        _model = big_df
        # test episode 1 again, rendered at 32x32 (mw.dataset gives episode i the seed 99 * 1_000_003 + i)
        _ctx = V.to_tensor(mw.episode(99 * 1_000_003 + 1, 8, 32)["frames"])[None]
    else:
        _model, _ctx = models["diffusion forcing"], test_ctx[1:2]
    _g = torch.Generator().manual_seed(0)

    def start():
        """Read the first 4 real frames of test episode 1; the 4th becomes the current frame."""
        h = _model.h0.expand(1, -1, -1, -1)
        k = torch.full((1,), k_ctx.value)
        for t in range(3):
            p = renoise_impl(_ctx[:, t], k_ctx.value, _g)
            h = _model.step(h, _model.encode(p, k), test_actions[1:2, t])
        x = _ctx[:, 3]
        return {"h": h, "prev": renoise_impl(x, k_ctx.value, _g), "frames": [_ctx[:, t] for t in range(4)]}

    @torch.no_grad()
    def act(state, action):
        """The RNN reads the current frame together with the chosen action, then the denoiser
        imagines the next frame."""
        k = torch.full((1,), k_ctx.value)
        h = _model.step(state["h"], _model.encode(state["prev"], k), torch.tensor([action]))
        x = V.sample_frame(_model, h, state["prev"], k, 10, _g).clamp(-1, 1)
        return {"h": h, "prev": renoise_impl(x, k_ctx.value, _g), "frames": state["frames"] + [x]}

    get_world, set_world = mo.state(start())
    def act2(state, action):  # two frames per click; one step only turns by 11 degrees
        return act(act(state, action), action)

    _buttons = [mo.ui.button(label=lbl, on_change=lambda _, a=a: set_world(act2(get_world(), a)))
                for lbl, a in [("← turn left", 1), ("↑ forward", 0), ("turn right →", 2)]]
    _reset = mo.ui.button(label="restart", on_change=lambda _: set_world(start()))
    play_controls = mo.hstack(_buttons + [_reset], justify="start")
    return get_world, play_controls


@app.cell(hide_code=True)
def _(V, get_world, mo, play_controls, plt):
    _frames = get_world()["frames"]
    _fig, _axes = plt.subplots(1, 2, figsize=(7, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
    _axes[0].imshow(V.to_uint8(_frames[-1][0]), interpolation="nearest")
    _axes[0].set_title(f"frame {len(_frames)}")
    _hist = [V.to_uint8(f[0]) for f in _frames[-8:]]
    _axes[1].imshow(__import__("numpy").concatenate(_hist, 1), interpolation="nearest")
    _axes[1].set_title("the last frames")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([play_controls, _fig])
    return


if __name__ == "__main__":
    app.run()
