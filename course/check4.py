"""Exercise checks for notebook 4. Each returns (passed, marimo element to display)."""
import torch

from . import solutions4
from .check import _close, _run, solutions_switch  # noqa: F401


def pick(student_fn, name, use_solutions):
    return getattr(solutions4, name) if use_solutions else student_fn


def drop(fn):
    def test():
        g = torch.Generator().manual_seed(0)
        a = torch.randint(3, (4000, 4))
        null = torch.tensor(3)
        if not torch.equal(fn(a, null, 0.0, g), a):
            return "with p = 0 nothing should change."
        if not (fn(a, null, 1.0, g) == 3).all():
            return "with p = 1 every row should be replaced by `null`."
        out = fn(a, null, 0.3, g)
        if out.shape != a.shape:
            return f"the output should have the same shape as `cond`, got {tuple(out.shape)}."
        rows = (out == 3).all(1)
        partial = ((out == 3) & (a != 3)).any(1) & ~rows
        if partial.any():
            return "replace whole rows (one decision per sample), not single entries."
        frac = (rows & ~(a == 3).all(1)).float().mean().item()
        if not 0.25 < frac < 0.35:
            return f"with p = 0.3 about 30% of the rows should be dropped, got {100 * frac:.0f}%."
        p = torch.cat([torch.zeros(2000), torch.ones(2000)])
        out = fn(a, null, p, g)
        if (out[:2000] != a[:2000]).any() or not (out[2000:] == 3).all():
            return "p can also be a tensor with one probability per row."
        m = torch.randn(10, 4, 17, 17)
        out = fn(m, torch.tensor(float("nan")), 1.0, g)
        if out.shape != m.shape or not torch.isnan(out).all():
            return "it should also work for maps of shape (B, 4, 17, 17) (null = NaN)."
    return _run("Exercise 1 · dropping the conditioning", test)


def guide(fn):
    def test():
        xc, xu = torch.randn(5, 3, 16, 16), torch.randn(5, 3, 16, 16)
        if not _close(fn(xc, xu, 1.0), xc):
            return "with w = 1 you should get the conditional prediction x_cond."
        if not _close(fn(xc, xu, 0.0), xu):
            return "with w = 0 you should get the unconditional prediction x_uncond."
        if not _close(fn(xc, xu, 3.0), solutions4.guide(xc, xu, 3.0)):
            return "for w = 3 the result is off. Start at x_uncond and go w times the difference toward x_cond."
    return _run("Exercise 3 · classifier-free guidance", test)


def idm_loss(fn):
    def test():
        from .policy import IDM
        torch.manual_seed(0)
        idm = IDM(16)
        a, b = torch.randn(8, 3, 16, 16), torch.randn(8, 3, 16, 16)
        y = torch.randint(3, (8,))
        out = fn(idm, a, b, y)
        if out.ndim != 0:
            return f"should return a single number (a 0-dim tensor), got shape {tuple(out.shape)}."
        ref = solutions4.idm_loss(idm, a, b, y)
        if not _close(out, ref):
            return f"got {out.item():.4f}, expected {ref.item():.4f} (the mean cross-entropy of the logits idm(a, b) and the actions)."
        out.backward()
    return _run("Exercise 2 · inverse dynamics loss", test)


def read_actions(fn):
    def test():
        from .policy import IDM
        torch.manual_seed(0)
        idm = IDM(16).eval()
        fr = torch.randn(3, 9, 3, 16, 16)
        with torch.no_grad():
            out, ref = fn(idm, fr), solutions4.read_actions(idm, fr)
        if tuple(out.shape) != (3, 8):
            return f"expected one action per pair of consecutive frames, shape (3, 8); got {tuple(out.shape)}."
        if not torch.equal(out.long(), ref):
            return "the actions don't match. For every j, run the IDM on (frames[:, j], frames[:, j + 1]) and take the most likely action."
    return _run("Exercise 4 · reading actions off imagined frames", test)
