import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="2 · Flow matching")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from course import toys, plots, check, solutions
    from course.nets import FlowNet, Generator
    torch.set_num_threads(4)
    return FlowNet, Generator, check, mo, np, plots, plt, solutions, torch, toys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2 · Flow matching, one picture at a time

    In notebook 1, a one-step generator had to learn the whole map from noise to data at once,
    and the only training signal was "how far is this sample from the data". Flow matching
    instead builds a *path* from noise to data and learns to follow it in many small steps. Each
    step only asks one *local* question: "given this noisy point, where is the data?".

    Notation: the same as in the lecture. A time $t$ runs from data ($t = 0$) to noise ($t = 1$):

    - Data: $x \sim p_{\text{data}}$, at time $t = 0$.
    - Noise: $\epsilon \sim \mathcal N(0, I)$, reached at time $t = 1$.
    - $x_t$ is the noisy point in between: $x_t = \alpha_t x + \sigma_t \epsilon$. We use the
      simplest schedule, the straight line $\alpha_t = 1 - t$, $\sigma_t = t$.
    - Conditional (Gaussian) probability path: $p_t(x_t \mid x) = \mathcal N\big((1-t)\,x,\ t^2 I\big)$.
    - Sampling runs backward in time, from noise at $t = 1$ to data at $t = 0$.

    Careful when you read papers: Lipman et al. (*Flow Matching for Generative Modeling*, 2023) and
    many others run time the other way, with noise at $t = 0$ and data at $t = 1$.

    Time: about 25 minutes. Same rules as before: ✏️ marks an exercise, and the switch uses the
    reference solutions.
    """)
    return


@app.cell(hide_code=True)
def _(check, mo, toys):
    use_solutions = check.solutions_switch()
    toy = mo.ui.dropdown(toys.TOYS, value="8 gaussians", label="toy dataset")
    mo.hstack([toy, use_solutions], justify="start", gap=2)
    return toy, use_solutions


@app.cell(hide_code=True)
def _(toy, toys):
    # The training set: a fixed, finite set of data points x^(1), ..., x^(N).
    X = toys.sample(toy.value, 1000, seed=0)
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 · The forward process: from data to noise

    To get a point on the conditional path at time $t$, take a data point $x$ and mix it with noise:

    $$
    x_t = \alpha_t x + \sigma_t \epsilon = (1 - t)\, x + t\, \epsilon, \qquad \epsilon \sim \mathcal N(0, I).
    $$

    Doing this for all data points gives the **marginal** path $p_t(x_t) = \int p_t(x_t \mid x)\, p_{\text{data}}(x)\, \mathrm dx$,
    which slides from the data ($t=0$) to pure noise ($t=1$).

    ### ✏️ Exercise A
    """)
    return


@app.cell
def _():
    def conditional_path(x, t, eps):
        """Sample x_t ~ p_t(x_t | x). x (data points) and eps (noise): (N, 2);
        t: a number or an (N, 1) tensor."""
        # ✏️ your code here
        raise NotImplementedError
    return (conditional_path,)


@app.cell(hide_code=True)
def _(check, conditional_path, use_solutions):
    conditional_path_impl = check.pick(conditional_path, "conditional_path", use_solutions.value)
    ok_A, _msg = check.conditional_path(conditional_path_impl)
    _msg
    return conditional_path_impl, ok_A


@app.cell(hide_code=True)
def _(X, conditional_path_impl, mo, ok_A, plots, plt, torch):
    mo.stop(not ok_A)
    _eps = torch.randn(X.shape, generator=torch.Generator().manual_seed(0))
    _ts = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    _fig, _axes = plt.subplots(1, len(_ts), figsize=(2.0 * len(_ts), 2.2))
    for _ax, _t in zip(_axes, _ts):
        plots.scatter(_ax, conditional_path_impl(X, _t, _eps), plots.ACCENT, s=2)
        plots.square(_ax, lim=3.5, title=f"t = {_t}")
    _fig.suptitle("The marginal path p_t: noise (t = 1) on the left, data (t = 0) on the right. "
                  "Noising runs right to left.", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 · The conditional vector field

    If we knew which data point $x$ we are heading to, following the path would be easy. Fix a
    data point $x$ and a noise sample $\epsilon$. Then $x_t = (1 - t)\, x + t\, \epsilon$ moves on a
    straight line, and its velocity is just the time derivative:

    $$
    \frac{\mathrm d}{\mathrm dt}\, x_t = \epsilon - x .
    $$

    To sample, we will walk this line backward, from $\epsilon$ at $t = 1$ to $x$ at $t = 0$. But
    while we walk, we only know where we are, $x_t$, and not which $\epsilon$ we started from. So we
    need the velocity as a function of $x_t$ and $x$: the **conditional vector field**
    $u_t^{\text{target}}(x_t \mid x)$.

    ### ✏️ Exercise B

    Solve $x_t = (1 - t)\, x + t\, \epsilon$ for $\epsilon$ and plug it into $\epsilon - x$. You should
    get something very short. Then implement it.

    (The lecture's table writes the same computation for a general schedule
    $x_t = \alpha_t x + \sigma_t \epsilon$. There the velocity is $\dot\alpha_t x + \dot\sigma_t \epsilon$,
    and eliminating $\epsilon$ gives the formula with $\dot\alpha_t$ and $\dot\sigma_t$.)
    """)
    return


@app.cell
def _():
    def conditional_velocity(x_t, x, t):
        """u_t(x_t | x) for x_t = (1 - t) x + t eps. x_t (noisy point), x (data point): (N, 2);
        t: number or (N, 1), with t > 0."""
        # ✏️ your code here
        raise NotImplementedError
    return (conditional_velocity,)


@app.cell(hide_code=True)
def _(check, conditional_velocity, use_solutions):
    conditional_velocity_impl = check.pick(conditional_velocity, "conditional_velocity", use_solutions.value)
    ok_B, _msg = check.conditional_velocity(conditional_velocity_impl)
    _msg
    return conditional_velocity_impl, ok_B


@app.cell(hide_code=True)
def _(X, conditional_velocity_impl, mo, ok_B, plots, plt, torch):
    mo.stop(not ok_B)
    _x = X[:1]  # one data point
    # walk backward in time, from t = 1 (noise) to t = 0: x_{t-h} = x_t - h u_t(x_t | x)
    _xt = torch.randn(40, 2, generator=torch.Generator().manual_seed(1))
    _traj = [_xt]
    for _i in range(50):
        _t = 1 - _i / 50
        _xt = _xt - (1 / 50) * conditional_velocity_impl(_xt, _x, _t)
        _traj.append(_xt)
    _fig, _ax = plt.subplots(figsize=(3.2, 3.2))
    plots.scatter(_ax, X, plots.DATA, s=2, alpha=0.4)
    plots.trajectories(_ax, torch.stack(_traj), plots.ACCENT, n=40)
    _ax.scatter(*_x[0], marker="*", s=150, c="k", zorder=4)
    plots.square(_ax, lim=3.5, title="Conditional flow paths to one x (star)")
    mo.hstack([_fig, mo.md(r"""
    Starting from 40 noise samples (grey dots) and following $u_t(x_t \mid x)$ backward in time,
    every path is a **straight line** into the star. But to sample, we don't get to choose $x$: that
    is exactly what we want the model to produce. We need a velocity that only depends on $x_t$.
    """)], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 · The posterior: which data point did this noisy point come from?

    Call the training points $x^{(1)}, \dots, x^{(N)}$. Given a noisy point $x_t$ at time $t$,
    Bayes' rule tells us how likely each training point $x^{(i)}$ is to be its origin:

    $$
    p(x^{(i)} \mid x_t) = \frac{p_t(x_t \mid x^{(i)})}{\sum_j p_t(x_t \mid x^{(j)})}
    = \operatorname{softmax}_i\!\Big( -\frac{\|x_t - (1-t)\, x^{(i)}\|^2}{2t^2} \Big).
    $$

    Its mean, $\hat x(x_t, t) = \mathbb E[x \mid x_t] = \sum_i p(x^{(i)} \mid x_t)\, x^{(i)}$,
    is the **best denoised guess** of the clean data. This is what a "denoiser" network learns.

    ### ✏️ Exercise C

    `x_t` has shape `(N, 2)`, `X` holds the training set `(M, 2)`. Return the `(N, M)` matrix of
    posterior weights (each row sums to 1).
    """)
    return


@app.cell
def _():
    def posterior_weights(x_t, t, X):
        """p(x^(i) | x_t) for all noisy points x_t (rows) and training points x^(i) (the rows of X;
        columns of the output). t is a number, 0 < t <= 1."""
        # ✏️ your code here (hint: torch.cdist and torch.softmax)
        raise NotImplementedError
    return (posterior_weights,)


@app.cell(hide_code=True)
def _(check, posterior_weights, use_solutions):
    posterior_weights_impl = check.pick(posterior_weights, "posterior_weights", use_solutions.value)
    ok_C, _msg = check.posterior_weights(posterior_weights_impl)
    _msg
    return ok_C, posterior_weights_impl


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.vstack([
        mo.md(r"""
        **🤔 Predict first.** Take a time $t$ close to $1$, so that $x_t$ is almost pure noise. Where
        will the posterior mean $\hat x$ be? Does it matter where you put $x_t$? And what happens as
        $t$ goes to $0$?
        """),
        mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
        - **$t$ near 1:** $x_t$ says almost nothing about $x$. The posterior is spread over the
          whole dataset, and $\hat x$ sits near the **mean of the data**, wherever you put $x_t$.
          At $t = 1$ exactly, $p(x \mid x_1) = p_{\text{data}}(x)$ and $\hat x$ *is* the
          mean. That is the blurry answer of notebook 1, Part 1!
        - **$t$ near 0:** the posterior focuses on a few training points near $x_t/(1-t)$, and
          $\hat x$ moves onto the data.
        """)}),
    ]), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    t_post = mo.ui.slider(0.02, 1.0, step=0.02, value=0.5, label="time t", show_value=True)
    qx = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.0, label="noisy point x_t: horizontal", show_value=True)
    qy = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.3, label="vertical", show_value=True)
    mo.hstack([t_post, qx, qy], justify="start", gap=1)
    return qx, qy, t_post


@app.cell(hide_code=True)
def _(X, mo, ok_C, plots, plt, posterior_weights_impl, qx, qy, t_post, torch):
    mo.stop(not ok_C)
    _t = t_post.value
    _xt = torch.tensor([[qx.value, qy.value]])
    _w = posterior_weights_impl(_xt, _t, X)[0]
    _xhat = _w @ X
    _fig, _axes = plt.subplots(1, 2, figsize=(6.6, 3.3))
    _ax = _axes[0]
    _order = _w.argsort()
    _ax.scatter(X[_order, 0], X[_order, 1], c=_w[_order], cmap="magma_r", s=6, vmin=0, vmax=max(_w.max().item(), 1e-3))
    _ax.scatter(*_xt[0], marker="x", s=80, c=plots.ACCENT, label="noisy point x_t")
    _ax.scatter(*_xhat, marker="*", s=160, c="k", label="posterior mean x̂")
    _ax.legend(loc="lower left", fontsize=7)
    plots.square(_ax, lim=3.1, title=f"posterior p(x | x_t) at t = {_t:.2f}")
    _ax = _axes[1]
    _ax.plot(_w.sort(descending=True).values[:50].numpy(), color=plots.ACCENT)
    _ax.set_title("the 50 largest weights")
    _ax.set_xlabel("rank")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(r"""
    Move the sliders and check your guess.

    Diffusion and flow matching never ask a network to solve the impossible problem in one
    step. At every $t$ they ask for the posterior mean, which is a well-posed regression.
    """)], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4 · The marginal vector field

    We don't know $x$, so we average the conditional velocities over the posterior:

    $$
    u_t^{\text{target}}(x_t) = \int u_t^{\text{target}}(x_t \mid x)\, \frac{p_t(x_t \mid x)\, p_{\text{data}}(x)}{p_t(x_t)}\, \mathrm dx
    = \sum_i p(x^{(i)} \mid x_t)\, \frac{x_t - x^{(i)}}{t} = \frac{x_t - \hat x(x_t, t)}{t}.
    $$

    The lecture's theorem: following this field backward in time from $x_1 = \epsilon \sim \mathcal N(0, I)$
    gives $x_t \sim p_t$ for all $t$, so $x_0 \sim p_{\text{data}}$.

    ### ✏️ Exercise D
    """)
    return


@app.cell
def _(posterior_weights_impl):
    def marginal_velocity(x_t, t, X):
        """u_t(x_t) = (x_t - posterior mean) / t. Use posterior_weights_impl(x_t, t, X)."""
        # ✏️ your code here
        raise NotImplementedError
    return (marginal_velocity,)


@app.cell(hide_code=True)
def _(check, marginal_velocity, use_solutions):
    marginal_velocity_impl = check.pick(marginal_velocity, "marginal_velocity", use_solutions.value)
    ok_D, _msg = check.marginal_velocity(marginal_velocity_impl)
    _msg
    return marginal_velocity_impl, ok_D


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.vstack([
        mo.md(r"""
        **🤔 Predict first.** Below we start from fresh noise and follow the *exact* marginal field
        of our 1000 training points down to $t = 0$. Where will the end points land? Will they be new
        samples from the toy distribution?
        """),
        mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
        They land on the **training points themselves**, not on new points between them. The field
        is built from the 1000 training points only, so it transports the noise onto the
        *empirical* distribution: 1000 spikes, one per training point. Near $t = 0$ the posterior
        puts all its weight on one training point (Part 3), and the field pulls $x_t$ straight onto
        it. The cell after the plot measures how close they get.
        """)}),
    ]), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    t_field = mo.ui.slider(0.05, 1.0, step=0.05, value=0.4, label="time t for the vector field", show_value=True)
    t_field
    return (t_field,)


@app.cell(hide_code=True)
def _(X, marginal_velocity_impl, mo, ok_D, plots, plt, t_field, torch):
    mo.stop(not ok_D)
    # integrate the exact marginal ODE (on the training set) backward, from noise at t = 1 to t = 0
    _x = torch.randn(400, 2, generator=torch.Generator().manual_seed(2))
    _traj = [_x]
    _n = 100
    for _i in range(_n):
        _x = _x - (1 / _n) * marginal_velocity_impl(_x, 1 - _i / _n, X)
        _traj.append(_x)
    exact_traj = torch.stack(_traj)
    _fig, _axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    _ax = _axes[0]
    _g = torch.linspace(-3, 3, 21)
    _grid = torch.stack(torch.meshgrid(_g, _g, indexing="xy"), -1).reshape(-1, 2)
    # draw -u_t: the direction a sample moves as t decreases
    _v = -marginal_velocity_impl(_grid, t_field.value, X)
    _v = _v / (_v.norm(dim=1, keepdim=True) + 1e-6) * _v.norm(dim=1, keepdim=True).clamp(max=1.5)
    plots.scatter(_ax, X, plots.DATA, s=2, alpha=0.4)
    _ax.quiver(_grid[:, 0], _grid[:, 1], _v[:, 0], _v[:, 1], color=plots.ACCENT, angles="xy",
               scale_units="xy", scale=4, width=0.004)
    plots.square(_ax, lim=3.1, title=f"marginal field −u_t(x_t) at t = {t_field.value:.2f}")
    _ax = _axes[1]
    plots.scatter(_ax, X, plots.DATA, s=2, alpha=0.4)
    plots.trajectories(_ax, exact_traj, plots.SAMPLE, n=80)
    plots.square(_ax, lim=3.1, title="marginal flow paths (exact field)")
    _fig.tight_layout()
    _fig
    return (exact_traj,)


@app.cell(hide_code=True)
def _(X, exact_traj, mo, torch):
    _d = torch.cdist(exact_traj[-1], X).min(1).values
    mo.md(rf"""
    Compare with Part 2: the **marginal** paths are curved and they fan out to every part of the
    data, while each **conditional** path was a straight line to a fixed $x$.

    The median distance from an end point to its nearest training point is
    **{_d.median():.4f}**. The exact marginal field for a finite training set lands *exactly on the
    training points*: it memorizes. A neural network trained on the same objective can't represent
    this field exactly (it is smoother), and that is where generalization comes from.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5 · Training a network: flow matching

    The marginal field needs a sum over the whole dataset for every point, which is hopeless for
    images. The trick of flow matching: regress a network $u^\theta_t(x_t)$ onto the **conditional**
    target, which needs just one data point at a time:

    $$
    \mathcal L(\theta) = \mathbb E_{t \sim U[0,1],\ x \sim p_{\text{data}},\ \epsilon \sim \mathcal N(0,I)}
    \big\| u_t^\theta(x_t) - u_t^{\text{target}}(x_t \mid x) \big\|^2,
    \qquad x_t = (1-t)\, x + t\,\epsilon,
    $$

    and for our schedule $u_t^{\text{target}}(x_t \mid x) = \frac{x_t - x}{t} = \epsilon - x$.
    The minimizer is the marginal field. Regression gives conditional means (notebook 1!), and the
    mean of the conditional velocity given $x_t$ is exactly $u_t^{\text{target}}(x_t)$.

    The network is a small MLP that takes $(x_t, t)$ and returns a velocity: call it as
    `net.velocity(x_t, t)`.

    ### ✏️ Exercise E

    Implement one batch of the loss. `x` is a batch of data points, `(B, D)`; draw `t` with shape `(B, 1)`.
    """)
    return


@app.cell
def _(torch):
    def flow_matching_loss(net, x):
        """Draw t ~ U[0, 1] (shape (B, 1)) and eps ~ N(0, I), build x_t, and return the mean
        squared error between net.velocity(x_t, t) and the target eps - x."""
        # ✏️ your code here (about 5 lines)
        raise NotImplementedError
    return (flow_matching_loss,)


@app.cell(hide_code=True)
def _(check, flow_matching_loss, use_solutions):
    flow_matching_loss_impl = check.pick(flow_matching_loss, "flow_matching_loss", use_solutions.value)
    ok_E, _msg = check.flow_matching_loss(flow_matching_loss_impl)
    _msg
    return flow_matching_loss_impl, ok_E


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ✏️ Exercise F · the sampler

    To sample, start from noise $x_1 = \epsilon$ and integrate the ODE $\frac{\mathrm d}{\mathrm dt}x_t = u^\theta_t(x_t)$
    backward in time with the Euler method from the lecture: $x_{t-h} = x_t - h\, u^\theta_t(x_t)$, with
    $h = 1/n$, until you reach $t = 0$. Return the **whole trajectory** so we can plot it.
    """)
    return


@app.cell
def _(torch):
    def euler_sample(velocity, eps, n_steps):
        """velocity(x_t, t) with t of shape (N, 1). Returns a tensor (n_steps + 1, N, D) holding
        x_1 = eps, x_{1-h}, x_{1-2h}, ..., x_0."""
        # ✏️ your code here
        raise NotImplementedError
    return (euler_sample,)


@app.cell(hide_code=True)
def _(check, euler_sample, use_solutions):
    euler_sample_impl = check.pick(euler_sample, "euler_sample", use_solutions.value)
    ok_F, _msg = check.euler_sample(euler_sample_impl)
    _msg
    return euler_sample_impl, ok_F


@app.cell
def _(FlowNet, flow_matching_loss_impl, torch):
    def train_flow(sample_data, D, steps=3000, batch=256, pred="v", seed=0):
        torch.manual_seed(seed)
        net = FlowNet(D, pred=pred)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        losses = []
        for step in range(steps):
            loss = flow_matching_loss_impl(net, sample_data(batch))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        return net, losses
    return (train_flow,)


@app.cell
def _(euler_sample_impl, mo, ok_E, ok_F, toy, toys, torch, train_flow):
    mo.stop(not (ok_E and ok_F), mo.md("*Finish Exercises E and F to train and sample.*"))
    flow_net, flow_losses = train_flow(lambda n: toys.sample(toy.value, n), D=2)
    with torch.no_grad():
        flow_traj = euler_sample_impl(flow_net.velocity, torch.randn(1500, 2, generator=torch.Generator().manual_seed(3)), 100)
    return flow_net, flow_losses, flow_traj


@app.cell(hide_code=True)
def _(X, flow_losses, flow_traj, mo, plots, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    plots.scatter(_axes[0], X, plots.DATA, s=2, alpha=0.5)
    plots.scatter(_axes[0], flow_traj[-1])
    plots.square(_axes[0], title="samples from the trained network")
    plots.scatter(_axes[1], X, plots.DATA, s=2, alpha=0.4)
    plots.trajectories(_axes[1], flow_traj, plots.SAMPLE, n=80)
    plots.square(_axes[1], lim=3.1, title="learned marginal flow paths")
    _l = flow_losses
    _axes[2].plot(plots.np.convolve(_l, plots.np.ones(50) / 50, mode="valid"), color=plots.ACCENT)
    _axes[2].set_title("loss (does not go to 0!)")
    _axes[2].set_xlabel("step")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md("**The reverse process**: the same samples, moving from noise ($t=1$) to data ($t=0$)."),
        (plots.animate_samples(list(flow_traj[::4]) + [flow_traj[-1]] * 5, X,
                                      [f"t = {1 - i * 4 / 100:.2f}" for i in range(26)] + ["t = 0.00"] * 5,
                                      interval=100)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why doesn't the loss go to zero? The target $\epsilon - x$ is random even for a fixed $x_t$
    (many $(x, \epsilon)$ pairs give the same $x_t$), and the network can only learn its average.
    The remaining loss is the variance of the target, the same irreducible term as in notebook 1.
    The difference: here, *learning the average is exactly what we want*.

    ### How many Euler steps do we need?

    Above we sampled with $n = 100$ Euler steps. Plain Euler steps like these are what most
    flow-matching models use in practice. But every step costs one evaluation of the network,
    which for a large image or video model is expensive, so we want as few steps as possible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.vstack([
        mo.md(r"""
        **🤔 Predict first.** What happens to the samples as you lower the number of steps $n$?
        What do you get with a **single** step from pure noise?
        """),
        mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
        With **one step** ($h = 1$, from $t = 1$), the sample is $x_1 - u_1(x_1)$. At $t = 1$ the
        point $x_1 = \epsilon$ is pure noise and says nothing about $x$, so the posterior mean is
        just the data mean, $\hat x(x_1, 1) = \mathbb E[x]$, and

        $$
        x_1 - u_1(x_1) = x_1 - \big(x_1 - \mathbb E[x]\big) = \mathbb E[x].
        $$

        **Every sample lands on the mean of the data**: the regression result of notebook 1, Part 1.

        In general, one Euler step moves a fraction $\frac{h}{t}$ of the way to the current
        denoised guess: $x_t - h\, u_t(x_t) = x_t + \frac{h}{t}\big(\hat x(x_t, t) - x_t\big)$. The last
        step has $h = t$, so it lands exactly on $\hat x$, the posterior mean at the time of
        the last step. With few steps that time is still close to the noise, the posterior is spread
        over several modes, and the samples land between the modes. With more steps, the last guess
        is made close to $t = 0$, where the posterior is sharp, and the samples separate into the modes. For the 8
        gaussians the samples are already good at about 8 steps, and more steps change little.
        """)}),
    ]), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    n_euler = mo.ui.slider(steps=[1, 2, 3, 4, 8, 16, 32, 100], value=1, label="number of Euler steps n", show_value=True)
    n_euler
    return (n_euler,)


@app.cell(hide_code=True)
def _(X, euler_sample_impl, flow_net, mo, n_euler, plots, plt, torch):
    with torch.no_grad():
        _traj = euler_sample_impl(flow_net.velocity, torch.randn(1500, 2, generator=torch.Generator().manual_seed(3)), n_euler.value)
    _fig, _axes = plt.subplots(1, 2, figsize=(6.6, 3.3))
    for _ax in _axes:
        plots.scatter(_ax, X, plots.DATA, s=2, alpha=0.5)
    plots.scatter(_axes[0], _traj[-1])
    _axes[0].scatter(*X.mean(0), marker="*", s=160, c="k", zorder=4, label="mean of the data")
    _axes[0].legend(loc="lower left", fontsize=7)
    plots.square(_axes[0], lim=3.1, title=f"samples with n = {n_euler.value}")
    plots.trajectories(_axes[1], _traj, plots.SAMPLE, n=80)
    plots.square(_axes[1], lim=3.1, title=f"paths: {n_euler.value} straight piece{'s' if n_euler.value > 1 else ''}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"Spread of the samples (standard deviation per coordinate): **{_traj[-1].std(0).mean():.2f}**, "
        f"spread of the data: **{X.std(0).mean():.2f}**. Move the slider."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 6 · The high-dimensional test

    Now the setting where the drifting model of notebook 1 fell apart: the data hidden on a
    crumpled 2D sheet in $\mathbb R^D$. The flow is now in $\mathbb R^D$: the noise is $D$-dimensional,
    and the network sees $D$ numbers. Nothing else changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One more ingredient. So far the network output the velocity directly ("$v$-prediction").
    Alternatively it can output a guess $\hat x_\theta(x_t, t)$ of the **clean data point** and
    convert it, $u^\theta_t(x_t) = (x_t - \hat x_\theta(x_t,t))/t$, with the same loss as before
    ("$x$-prediction"; Li & He, *Back to Basics: Let Denoising Generative Models Denoise*, 2025,
    the "JiT" paper from Kaiming He's group). In 2D, $v$-prediction is actually the better choice. Below we
    train both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.vstack([
        mo.md(r"""
        **🤔 Predict first.** The data is a 2D sheet crumpled into $D = 256$ dimensions. Which of
        $x$-prediction and $v$-prediction will work there: both, neither, or only one of them?
        """),
        mo.accordion({"Answer (click after you've guessed)": mo.md(r"""
        Only **$x$-prediction**. Its samples cover every mode and lie close to the sheet.
        $v$-prediction, the better choice in 2D, falls apart: its samples end up about as far from
        the sheet as the blurry data mean.

        The two networks are trained on the same loss, but they have to output different things.
        The velocity target $\epsilon - x$ contains the full $D$-dimensional noise, which a small
        network struggles to represent. The clean data point $x$ lies on a 2D sheet, so
        predicting it is a much simpler function.
        """)}),
    ]), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    D_slider = mo.ui.slider(16, 512, step=16, value=256, label="dimension D", show_value=True)
    omega = mo.ui.slider(1.0, 8.0, step=0.5, value=4.0, label="crumpling ω", show_value=True)
    compare = mo.ui.checkbox(value=True, label="also train notebook 1's drifting model (reference solution) for comparison")
    mo.vstack([mo.hstack([D_slider, omega], justify="start", gap=1), compare])
    return D_slider, compare, omega


@app.cell
def _(
    D_slider,
    Generator,
    compare,
    euler_sample_impl,
    mo,
    ok_E,
    ok_F,
    omega,
    plots,
    solutions,
    toy,
    toys,
    torch,
    train_flow,
):
    mo.stop(not (ok_E and ok_F))
    emb = toys.Embedding("crumpled", D_slider.value, omega=omega.value)
    _sample = lambda n: emb(toys.sample(toy.value, n))

    def train_drifting_reference(sample_data, D, lam=1.0, tau_rel=0.07, steps=1500, batch=256):
        torch.manual_seed(0)
        G = Generator(D=D)
        opt = torch.optim.Adam(G.parameters(), lr=1e-3)
        x_ref = sample_data(1000)
        tau = tau_rel * torch.cdist(x_ref, x_ref).median()
        for _ in range(steps):
            y = G(torch.randn(batch, 2))
            with torch.no_grad():
                target = solutions.drift_target(y, sample_data(batch), lam, tau)
            loss = ((y - target) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return G

    _res = []
    with mo.status.spinner(title="training in R^D..."):
        if compare.value:
            _G = train_drifting_reference(_sample, emb.D)
            with torch.no_grad():
                _res.append(("drifting (notebook 1)", _G(torch.randn(1500, 2, generator=torch.Generator().manual_seed(1)))))
        for _pred in ["v", "x"]:
            _net, _ = train_flow(_sample, emb.D, pred=_pred)
            with torch.no_grad():
                _eps = torch.randn(1500, emb.D, generator=torch.Generator().manual_seed(1))
                _res.append((f"flow matching, {_pred}-prediction", euler_sample_impl(_net.velocity, _eps, 100)[-1]))
    plots.decoded_panels(_res, emb, toys.sample(toy.value, 2000, seed=0),
                         f"Data on a crumpled 2D sheet in R^{emb.D}, samples decoded back to 2D")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Flow matching with x-prediction covers every mode, and its samples lie close to the
    sheet.** The drifting model collapses. The difference is the training signal. The drifting model compared samples
    to data points through long-range distances, which are meaningless on a crumpled sheet. Flow
    matching only ever regresses a noisy version of a data point back to that data point, a
    *local* problem with a well-defined answer at every noise level.

    **v-prediction falls apart** in high dimensions (its samples end up about as far from the sheet
    as the blurry data mean), although it was the better choice in 2D. Its target $\epsilon - x$
    contains the full $D$-dimensional noise, while the clean data $x$ lives on a 2D sheet.
    Lower $D$ with the slider and the gap shrinks. This is the point of the JiT paper, and why it
    can train diffusion transformers directly on pixels.

    **Next:** notebook 3 trains the same kind of model on video.
    """)
    return


if __name__ == "__main__":
    app.run()
