import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="1 · From regression to drifting")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from course import toys, plots, check
    from course.nets import Generator
    torch.set_num_threads(4)
    return Generator, check, mo, np, plots, plt, torch, toys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1 · From regression to a (primitive) generative model

    A generative model is a machine that turns random numbers into samples that look like the
    training data. In this notebook we build one from scratch, on 2D toy data where we can *see*
    everything. Then we hide the same data in a high-dimensional space and watch our method break.
    That failure is the reason the field moved to diffusion and flow matching (notebook 2).

    **How to work here.** This is a [marimo](https://marimo.io) notebook: cells re-run
    automatically when something they depend on changes. Exercises are marked **✏️ Exercise**:
    replace the `raise NotImplementedError` with your code. A check below each exercise tells you
    whether it works. If you get stuck, flip the switch below to use the reference solutions
    (they are in `course/solutions.py`). Time: about 20 minutes.

    **Everything is PyTorch.** The networks are `torch.nn.Module`s, and the data and noise are
    `torch` tensors.

    **Ask an AI assistant.** If you use a coding agent (Claude Code, Codex, Cursor, ...), start it
    in this folder and ask it anything: what `torch.cdist` does, why your shapes don't match, what
    an equation means, why a method fails. The folder has an `AGENTS.md` file that tells the agent
    to be a tutor. It answers PyTorch questions directly and helps you find the answer with hints
    and questions, but it won't write the exercise code for you.
    """)
    return


@app.cell(hide_code=True)
def _(check, mo, toys):
    use_solutions = check.solutions_switch()
    toy = mo.ui.dropdown(toys.TOYS, value="8 gaussians", label="toy dataset")
    mo.hstack([toy, use_solutions], justify="start", gap=2)
    return toy, use_solutions


@app.cell(hide_code=True)
def _(plots, plt, toys):
    _fig, _axes = plt.subplots(1, len(toys.TOYS), figsize=(2.0 * len(toys.TOYS), 2.1))
    for _ax, _name in zip(_axes, toys.TOYS):
        plots.scatter(_ax, toys.sample(_name, 2000, seed=0), plots.ACCENT, s=1.5)
        plots.square(_ax, title=_name)
    _fig.suptitle("The toy datasets. Pick one with the dropdown above; everything below uses it.", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 · The obvious idea: regression

    A one-step generator is a neural network $G_\theta$ that maps noise to data:

    $$
    z \sim \mathcal N(0, I), \qquad x = G_\theta(z).
    $$

    We know how to train networks with regression. So let's do the obvious thing: draw some
    noise $z$, draw a random training point $x$, and make $G_\theta(z)$ close to $x$:

    $$
    \min_\theta\; \mathbb E_{z,\,x}\, \big\| G_\theta(z) - x \big\|^2, \qquad z \perp x .
    $$

    ### ✏️ Exercise 1

    Implement this loss for a batch: `z` and `x` both have shape `(batch, 2)`. Return the squared
    distance summed over the 2 coordinates and averaged over the batch.
    """)
    return


@app.cell
def _(torch):
    def regression_loss(G, z, x):
        """G: the generator network. z: noise (B, 2). x: random training points (B, 2)."""
        # ✏️ your code here (one line)
        raise NotImplementedError
    return (regression_loss,)


@app.cell(hide_code=True)
def _(check, regression_loss, use_solutions):
    regression_loss_impl = check.pick(regression_loss, "regression_loss", use_solutions.value)
    ok_regression, _msg = check.regression_loss(regression_loss_impl)
    _msg
    return ok_regression, regression_loss_impl


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.vstack([
        mo.md(r"""
        **🤔 Predict first.** The next cell trains $G_\theta$ with this loss and shows where 600
        fixed noise vectors $z$ end up. Will the samples spread over the data, like a generative
        model should? If not, what will they converge to?
        """),
        mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
        They don't spread out. All 600 samples converge to **one point, the mean of the data**,
        whatever their $z$ was. The box below the animation shows why.
        """)}),
    ]), kind="info")
    return


@app.cell
def _(Generator, ok_regression, mo, plots, regression_loss_impl, torch, toy, toys):
    mo.stop(not ok_regression, mo.md("*Finish Exercise 1 to train the generator.*"))

    def train_regression(steps=1500, batch=256, seed=0):
        torch.manual_seed(seed)
        G = Generator(D=2)
        opt = torch.optim.Adam(G.parameters(), lr=1e-3)
        z_show = torch.randn(600, 2)  # the same noise every time, so we can watch it move
        snaps, titles = [], []
        for step in range(steps + 1):
            if step % 50 == 0:
                with torch.no_grad():
                    snaps.append(G(z_show))
                titles.append(f"step {step}")
            z = torch.randn(batch, 2)
            x = toys.sample(toy.value, batch)
            loss = regression_loss_impl(G, z, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return snaps, titles

    _snaps, _titles = train_regression()
    _data = toys.sample(toy.value, 2000, seed=0)
    mo.vstack([
        mo.md("**Regression onto random training points.** Grey: data. Orange: $G_\\theta(z)$ for 600 fixed noise vectors."),
        (plots.animate_samples(_snaps, _data, _titles)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every sample ends up at the same point: the **mean** of the data. Before you open the box,
    try to explain why.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Why the mean? (click)": mo.md(r"""
    Because $z$ and $x$ are independent, for every fixed $z$ the loss is

    $$
    \mathbb E_x \|G(z) - x\|^2 = \|G(z) - \mathbb E[x]\|^2 + \mathbb E_x\|x - \mathbb E[x]\|^2 .
    $$

    The second term does not depend on $G$. The first is smallest when $G(z) = \mathbb E[x]$,
    *for every* $z$. The noise carries no information about which target it will be compared to,
    so the best the network can do is to hedge: output the average.

    With a condition $c$ (a text prompt, the past frames of a video), the same argument gives
    $G(z, c) = \mathbb E[x \mid c]$: the *conditional* mean. For images, that is the familiar
    blurry average. Regression is not generative modeling.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 · A primitive drifting model

    To break the symmetry, the target for a sample must depend on **the sample itself**. Let
    $y_i = G_\theta(z_i)$ be a batch of generated samples and $X = \{x_1, \dots, x_M\}$ a batch of
    training points. We give every sample a **drift** $V(y_i)$, a little arrow that says where it
    should move, made of two parts:

    1. **Attraction:** move toward your nearest training point, $\;\mathrm{NN}_X(y_i) - y_i$.
    2. **Repulsion:** move away from the other generated samples, weighted by how close they are:

    $$
    R(y_i) = \sum_{j \neq i} w_{ij}\,(y_i - y_j), \qquad
    w_{ij} = \frac{\exp(-\|y_i - y_j\|/\tau)}{\sum_{l \neq i} \exp(-\|y_i - y_l\|/\tau)} .
    $$

    So $V(y_i) = \mathrm{NN}_X(y_i) - y_i + \lambda\, R(y_i)$, where $\lambda$ sets the strength of the
    repulsion and $\tau$ its range. We train by regressing each sample onto its own nudged
    position, with the target held fixed ("stop-gradient", $\mathrm{sg}$):

    $$
    \mathcal L(\theta) = \frac1B \sum_i \big\| G_\theta(z_i) - \mathrm{sg}\big[\,y_i + V(y_i)\,\big] \big\|^2 .
    $$

    The gradient of this loss with respect to the output $y_i$ is $-2V(y_i)$, so each training step
    moves every sample a bit along its arrow. This is a stripped-down version of *drifting models*
    (Deng et al., 2026), where attraction and repulsion use the same kernel.

    ### ✏️ Exercise 2

    Three small functions. `y` is a batch of generated samples `(B, D)`, `x` a batch of training
    points `(M, D)`. Useful tools: `torch.cdist(a, b)` gives all pairwise distances,
    `t.argmin(dim=1)`, `torch.softmax(t, dim=1)`, and `t.fill_diagonal_(float("-inf"))`.
    """)
    return


@app.cell
def _(torch):
    def nearest_neighbor(y, x):
        """(a) For every row y_i return the closest row of x. Shape (B, D)."""
        # ✏️ your code here
        raise NotImplementedError

    def repulsion(y, tau):
        """(b) R(y_i) = sum_{j != i} w_ij (y_i - y_j), with w_ij = softmax_j(-|y_i - y_j| / tau),
        excluding j = i. Shape (B, D). Hint: since the weights of a row sum to 1,
        sum_j w_ij (y_i - y_j) = y_i - sum_j w_ij y_j = y - w @ y."""
        # ✏️ your code here
        raise NotImplementedError

    def drift_target(y, x, lam, tau):
        """(c) The regression target y + V(y), with
        V(y) = nearest_neighbor(y, x) - y + lam * repulsion(y, tau)."""
        # ✏️ your code here
        raise NotImplementedError
    return drift_target, nearest_neighbor, repulsion


@app.cell(hide_code=True)
def _(check, drift_target, mo, nearest_neighbor, repulsion, use_solutions):
    _s = use_solutions.value
    _ok_a, _ma = check.nearest_neighbor(check.pick(nearest_neighbor, "nearest_neighbor", _s))
    _ok_b, _mb = check.repulsion(check.pick(repulsion, "repulsion", _s))
    drift_target_impl = check.pick(drift_target, "drift_target", _s)
    _ok_c, _mc = check.drift_target(drift_target_impl)
    ok_drift = _ok_a and _ok_b and _ok_c
    mo.vstack([_ma, _mb, _mc])
    return drift_target_impl, ok_drift


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.vstack([
        mo.md(r"""
        **🤔 Predict first.** The animation below trains two drifting models. On the left there is
        no repulsion ($\lambda = 0$): every sample just moves toward its nearest training point.
        Will those samples cover all of the data?
        """),
        mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
        No. They collapse: on the 8 gaussians, all of them end up on one small spot of a single
        mode, and the rest of the data stays empty. Nothing in the attraction looks at the other
        samples, so nothing stops them from crowding together. The repulsion on the right is
        there to fix exactly this. The list below the animation has more.
        """)}),
    ]), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    lam = mo.ui.slider(0.0, 3.0, step=0.25, value=1.0, label="repulsion strength λ", show_value=True)
    tau_rel = mo.ui.slider(0.02, 0.3, step=0.01, value=0.07, label="kernel width τ (× median distance)", show_value=True)
    mo.hstack([lam, tau_rel], justify="start", gap=2)
    return lam, tau_rel


@app.cell
def _(Generator, drift_target_impl, torch):
    def train_drifting(sample_data, D, lam, tau_rel, steps=1500, batch=256, seed=0, snap_every=50):
        """Train G: R^2 -> R^D with the primitive drifting field. sample_data(n) -> (n, D)."""
        torch.manual_seed(seed)
        G = Generator(D=D)
        opt = torch.optim.Adam(G.parameters(), lr=1e-3)
        x_ref = sample_data(1000)
        tau = tau_rel * torch.cdist(x_ref, x_ref).median()  # kernel width in units of the data scale
        z_show = torch.randn(600, 2)
        snaps, arrows, titles = [], [], []
        for step in range(steps + 1):
            if step % snap_every == 0:
                with torch.no_grad():
                    y_show = G(z_show)
                    snaps.append(y_show)
                    arrows.append(drift_target_impl(y_show, sample_data(256), lam, tau) - y_show)
                titles.append(f"step {step}")
            y = G(torch.randn(batch, 2))
            with torch.no_grad():
                target = drift_target_impl(y, sample_data(batch), lam, tau)
            loss = ((y - target) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return G, snaps, arrows, titles
    return (train_drifting,)


@app.cell
def _(lam, mo, ok_drift, plots, tau_rel, toy, toys, train_drifting):
    mo.stop(not ok_drift, mo.md("*Finish Exercise 2 to train the drifting model.*"))
    _sample = lambda n: toys.sample(toy.value, n)
    _, _snaps0, _arrows0, _titles0 = train_drifting(_sample, 2, 0.0, tau_rel.value)
    _, _snaps, _arrows, _titles = train_drifting(_sample, 2, lam.value, tau_rel.value)
    _data = toys.sample(toy.value, 2000, seed=0)
    mo.vstack([
        mo.md("**Left: attraction only (λ = 0). Right: attraction + repulsion (λ from the slider).** "
              "Blue arrows: the drift $V$ on each sample. "),
        mo.hstack([(plots.animate_samples(_snaps0, _data, _titles0, field=_arrows0)),
                   (plots.animate_samples(_snaps, _data, _titles, field=_arrows))]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What to look for.**

    - **Attraction only collapses.** Each sample looks only at the data, never at the other
      samples, so nothing stops them from all piling onto the same few training points. The
      network shares its weights across all $z$, so once a few samples move to one place, their
      neighbors follow.
    - **Repulsion fixes it.** Crowded samples push each other apart, so they spread over all the
      data. That is a real (if crude) generative model: new $z$ gives new, plausible samples.
    - **It is crude.** Samples also land in the gaps (try the rings or the checkerboard). The
      attraction is zero anywhere on the data, so only the repulsion decides how samples spread,
      and it spreads them evenly instead of following the data density.
    - **The thin bridges between modes** are not a bug in your code. $G$ is a continuous map from
      the connected plane $\mathbb R^2$, so it cannot jump between separated modes; the best it can
      do is make the bridges thin.

    Try other datasets and slider values. What happens with a very large $\lambda$? (A smaller
    $\lambda$ gives cleaner samples in 2D, but the high-dimensional experiment below needs
    $\lambda \approx 1$ to work reliably.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 · Hiding the data in many dimensions

    Real data is not 2D. A 64×64 color image has 12,288 numbers, but natural images occupy a tiny,
    curved, low-dimensional sheet inside that space. We can imitate this: keep our 2D toy as a
    hidden "latent" $u$, and only show the model $x = f(u) \in \mathbb R^D$.

    - **Linear** (the toy from the JiT paper, Li & He 2025): $f(u) = P u$ with a random
      $D \times 2$ matrix $P$ with orthonormal columns. A flat 2D sheet, tilted in $\mathbb R^D$.
    - **Crumpled**: $f(u) = \cos(W u + b)$, with each row of $W$ drawn from $\mathcal N(0, \omega^2 I)$.
      A 2D sheet crumpled up like paper. The larger $\omega$, the more crumpled.

    To look at a generated sample $y \in \mathbb R^D$, we find the closest point on the sheet
    and plot its 2D latent position. We color it by its distance from the sheet: $0$ means the
    sample is a perfect data point, and large means it is a blurry mix.

    For comparison, the color bar also gives the distance of the *mean of the data* from the
    sheet, which is what a completely blurry sample would score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.vstack([
        mo.md(r"""
        **🤔 Predict first.** We now train the same drifting model (with the $\lambda$ and $\tau$
        from the sliders above) on data hidden in $D$ dimensions (64 by default). Will it still work on the
        flat (linear) sheet? On the crumpled sheet?
        """),
        mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
        **Flat sheet: yes.** It covers all the data, as in 2D. **Crumpled sheet: no.** The samples
        collapse onto a single spot. So the number of dimensions alone is not the problem; the
        crumpling is. The distance plot below is a hint, and the explanation follows the results.
        """)}),
    ]), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    D_slider = mo.ui.slider(8, 256, step=8, value=64, label="dimension D", show_value=True)
    omega = mo.ui.slider(1.0, 8.0, step=0.5, value=4.0, label="crumpling ω", show_value=True)
    mo.hstack([D_slider, omega], justify="start", gap=2)
    return D_slider, omega


@app.cell(hide_code=True)
def _(D_slider, omega, plt, toy, toys, torch):
    emb_linear = toys.Embedding("linear", D_slider.value)
    emb_crumpled = toys.Embedding("crumpled", D_slider.value, omega=omega.value)
    # How do distances in R^D relate to distances on the hidden 2D sheet?
    _u = toys.sample(toy.value, 400, seed=1)
    _du = torch.pdist(_u)
    _fig, _axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for _ax, _emb, _name in [(_axes[0], emb_linear, "linear"), (_axes[1], emb_crumpled, "crumpled")]:
        _dx = torch.pdist(_emb(_u))
        _ax.scatter(_du[::7], _dx[::7], s=1, alpha=0.3)
        _ax.set_xlabel("distance on the hidden 2D sheet")
        _ax.set_ylabel(f"distance in R^{_emb.D}")
        _ax.set_title(_name)
    _fig.suptitle("Crumpling: beyond a short range, every pair of points is about equally far apart", fontsize=9)
    _fig.tight_layout()
    _fig
    return emb_crumpled, emb_linear


@app.cell
def _(
    emb_crumpled,
    emb_linear,
    lam,
    mo,
    ok_drift,
    plots,
    tau_rel,
    toy,
    toys,
    torch,
    train_drifting,
):
    mo.stop(not ok_drift, mo.md("*Finish Exercise 2 first.*"))

    _u = toys.sample(toy.value, 2000, seed=0)
    _res = []
    for _name, _emb in [("linear", emb_linear), ("crumpled", emb_crumpled)]:
        _G, *_ = train_drifting(lambda n: _emb(toys.sample(toy.value, n)), _emb.D, lam.value, tau_rel.value, snap_every=10**9)
        with torch.no_grad():
            _y = _G(torch.randn(1500, 2, generator=torch.Generator().manual_seed(1)))
        _res.append(plots.decoded_panels([(f"drifting · {_name} R^{_emb.D}", _y)], _emb, _u, ""))
    mo.vstack([mo.md("**The same drifting model, with the data hidden in high dimensions**, decoded back to 2D."),
               mo.hstack(_res, justify="start")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What happened?**

    - **Linear: still fine.** Dimension alone is not the problem. For a flat sheet,
      $\|Pu - y\|^2 = \|u - P^\top y\|^2 + \text{(a term that is the same for every data point)}$, so
      "nearest neighbor" means the same thing as in 2D.
    - **Crumpled: it falls apart.** The samples collapse onto a single spot. With a small kernel width
      $\tau$ that spot is one mode (sharp, but no diversity). Increase $\tau$ and the samples drift
      off the sheet toward a blurry average. The reason is the distance plot above. Our drift is
      built entirely from *distances*: nearest neighbors and kernel weights. In a crumpled space,
      a sample that is not already close to the data is about equally far from *every* training
      point. So the attraction carries no information about *where* on the sheet to go (its
      "nearest neighbor" is essentially random, as in Part 1), and the samples settle wherever
      they first touch the sheet. The repulsion doesn't help either. It pushes the samples apart
      in all $D$ directions, and $D - 2$ of those lead straight off the sheet.

    Images behave like the crumpled sheet: shift a photo by a few pixels and, in pixel space, it is
    about as far from the original as an unrelated photo is. (Real drifting models avoid this by
    computing distances in the feature space of a pretrained network.)

    **Next:** notebook 2 trains a flow-matching model on exactly these datasets, including the
    crumpled one. Flow matching never compares a sample to a far-away data point; it only
    ever solves *local* denoising problems.
    """)
    return


if __name__ == "__main__":
    app.run()
