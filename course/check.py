"""Quick checks for the exercises. Each returns (passed, marimo element to display)."""
import os
import traceback

import marimo as mo
import torch

from . import solutions

# For automated test runs of the notebooks: QC_SOLUTIONS=1 starts with the solutions switched on.
SOLUTIONS_DEFAULT = os.environ.get("QC_SOLUTIONS", "0") == "1"


def solutions_switch():
    return mo.ui.switch(value=SOLUTIONS_DEFAULT, label="use the reference solutions for every exercise")


def pick(student_fn, name, use_solutions):
    return getattr(solutions, name) if use_solutions else student_fn


def _run(name, test):
    try:
        msg = test()
    except NotImplementedError:
        return False, mo.callout(mo.md(f"**{name}**: not implemented yet."), kind="warn")
    except Exception as e:  # show the student what went wrong, briefly
        tb = traceback.format_exception_only(type(e), e)[-1].strip()
        return False, mo.callout(mo.md(f"**{name}** raised an error: `{tb}`"), kind="danger")
    if msg:
        return False, mo.callout(mo.md(f"**{name}**: {msg}"), kind="danger")
    return True, mo.callout(mo.md(f"**{name}**: looks right ✓"), kind="success")


def _close(a, b, tol=1e-4):
    return a.shape == b.shape and torch.allclose(a, b, atol=tol, rtol=tol)


def regression_loss(fn):
    def test():
        G = torch.nn.Linear(2, 2)
        eps, x = torch.randn(64, 2), torch.randn(64, 2)
        out, ref = fn(G, eps, x), solutions.regression_loss(G, eps, x)
        if out.ndim != 0:
            return f"should return a single number (a 0-dim tensor), got shape {tuple(out.shape)}."
        if not _close(out, ref):
            return f"got {out.item():.4f}, expected {ref.item():.4f}. Sum the squares over the 2 coordinates, average over the batch."
        out.backward()  # must be differentiable
    return _run("Exercise 1 · regression loss", test)


def nearest_neighbor(fn):
    def test():
        y, x = torch.randn(50, 3), torch.randn(20, 3)
        out = fn(y, x)
        if not _close(out, solutions.nearest_neighbor(y, x)):
            return "for each row of y, return the row of x that is closest to it (shape like y)."
    return _run("Exercise 2a · nearest neighbor", test)


def repulsion(fn):
    def test():
        y = torch.randn(40, 2)
        out = fn(y, 0.5)
        ref = solutions.repulsion(y, 0.5)
        if out.shape != ref.shape:
            return f"expected shape {tuple(ref.shape)}, got {tuple(out.shape)}."
        if _close(out, -ref):
            return "the sign is flipped: the force should point *away* from the other samples."
        if not _close(out, ref):
            return "not quite. Check that a sample gets no weight on itself (w_ii = 0) and that the weights of each row sum to 1."
    return _run("Exercise 2b · repulsion", test)


def drift_target(fn):
    def test():
        y, x = torch.randn(40, 2), torch.randn(30, 2)
        out = fn(y, x, 0.7, 0.3)
        if not _close(out, solutions.drift_target(y, x, 0.7, 0.3)):
            return "the target is y + V(y), with V = (nearest neighbor - y) + lam * repulsion."
    return _run("Exercise 2c · drifting target", test)


def conditional_path(fn):
    def test():
        x, eps, t = torch.randn(10, 2), torch.randn(10, 2), torch.rand(10, 1)
        if not _close(fn(x, t, eps), solutions.conditional_path(x, t, eps)):
            return "x_t should be alpha_t x + sigma_t eps with alpha_t = 1 - t and sigma_t = t."
    return _run("Exercise A · conditional path", test)


def conditional_velocity(fn):
    def test():
        x_t, x, t = torch.randn(10, 2), torch.randn(10, 2), torch.rand(10, 1) * 0.9 + 0.1
        if not _close(fn(x_t, x, t), solutions.conditional_velocity(x_t, x, t)):
            return "eps = (x_t - (1 - t) x) / t, so eps - x simplifies to (x_t - x) / t."
    return _run("Exercise B · conditional vector field", test)


def posterior_weights(fn):
    def test():
        x_t, X = torch.randn(7, 2), torch.randn(30, 2)
        out = fn(x_t, 0.6, X)
        ref = solutions.posterior_weights(x_t, 0.6, X)
        if out.shape != ref.shape:
            return f"expected shape (number of x, number of training points) = {tuple(ref.shape)}, got {tuple(out.shape)}."
        if not torch.allclose(out.sum(1), torch.ones(7), atol=1e-4):
            return "each row should sum to 1 (it is a distribution over the training points)."
        if not _close(out, ref):
            return "close, but not equal. Use the log-density -|x_t - (1-t) x_i|^2 / (2 t^2) inside a softmax."
    return _run("Exercise C · posterior over data points", test)


def marginal_velocity(fn):
    def test():
        x_t, X = torch.randn(7, 2), torch.randn(30, 2)
        if not _close(fn(x_t, 0.4, X), solutions.marginal_velocity(x_t, 0.4, X)):
            return "average the conditional velocities (x_t - x_i)/t with the posterior weights."
    return _run("Exercise D · marginal vector field", test)


def flow_matching_loss(fn):
    def test():
        # A fixed toy "network" whose velocity depends on x_t and t, so that a flipped target
        # (x - eps) or a flipped time (x_t = t x + (1 - t) eps) gives a clearly different loss.
        w = torch.tensor(2.0, requires_grad=True)

        class ToyNet:
            def velocity(self, x_t, t):
                return w * x_t * t

        torch.manual_seed(0)
        x = torch.randn(16384, 2) * 0.5 + 1.0
        torch.manual_seed(1)
        out = fn(ToyNet(), x)
        torch.manual_seed(2)
        ref = solutions.flow_matching_loss(ToyNet(), x)
        if out.ndim != 0:
            return "should return a single number."
        if abs(out.item() - ref.item()) > 0.06 * ref.item():
            return (f"got {out.item():.3f}, expected about {ref.item():.3f}. Check x_t = (1 - t) x + t eps "
                    "and the target velocity eps - x.")
        out.backward()
    return _run("Exercise E · flow-matching loss", test)


def euler_sample(fn):
    def test():
        eps = torch.randn(5, 2)
        v = lambda x, t: -x + t  # some time-dependent field
        out = fn(v, eps, 10)
        ref = solutions.euler_sample(v, eps, 10)
        if out.shape != ref.shape:
            return f"return the whole trajectory, shape (n_steps + 1, N, 2) = {tuple(ref.shape)}; got {tuple(out.shape)}."
        if not _close(out, ref):
            return "each step is x <- x - h * velocity(x, t) with h = 1 / n_steps, starting at t = 1 and going down to t = 0."
    return _run("Exercise F · Euler sampler", test)


def noise_levels(fn):
    def test():
        B, T = 64, 8
        for regime in ["full sequence", "teacher forcing", "diffusion forcing"]:
            k, kc = fn(regime, B, T)
            if k.shape != (B, T) or kc.shape != (B, T):
                return f"{regime}: both outputs should have shape (B, T) = ({B}, {T})."
            if k.min() < 0 or k.max() > 1:
                return f"{regime}: levels should lie in [0, 1]."
            spread = k.std(1).mean().item()
            if regime == "full sequence" and (spread > 1e-6 or not torch.equal(k, kc)):
                return "full sequence: every frame of a clip shares one level, and the RNN reads the same noisy frames."
            if regime == "teacher forcing" and (spread < 0.1 or not torch.all(kc == 0)):
                return "teacher forcing: independent level per target frame, but the RNN reads clean frames (k_ctx = 0)."
            if regime == "diffusion forcing" and (spread < 0.1 or not torch.equal(k, kc)):
                return "diffusion forcing: an independent level per frame, and the RNN reads those same noisy frames."
    return _run("Exercise 1 · noise levels", test)


def renoise(fn):
    def test():
        x = torch.randn(200, 3, 16, 16)
        out = fn(x, 0.2)
        if out.shape != x.shape:
            return "output should have the shape of x."
        r = out - 0.8 * x
        if abs(r.std().item() - 0.2) > 0.02 or abs(r.mean().item()) > 0.02:
            return "x^k = (1 - k) x + k eps with fresh Gaussian noise eps."
        if not torch.allclose(fn(x, 0.0), x):
            return "at k_ctx = 0 the frame should come back unchanged."
    return _run("Exercise 2 · re-noising the context", test)
