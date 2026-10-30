"""2D toy distributions (all roughly inside [-2.5, 2.5]^2) and ways to hide them in R^D."""
import math

import torch

TOYS = ["8 gaussians", "spiral", "two moons", "checkerboard", "rings", "pinwheel"]


def sample(name, n, seed=None, generator=None):
    """n points from a 2D toy distribution, as a float tensor (n, 2)."""
    g = generator if generator is not None else torch.Generator().manual_seed(
        seed if seed is not None else torch.randint(1 << 30, ()).item())
    rand = lambda *s: torch.rand(*s, generator=g)
    randn = lambda *s: torch.randn(*s, generator=g)
    if name == "8 gaussians":
        a = torch.randint(0, 8, (n,), generator=g) * (2 * math.pi / 8)
        return 2 * torch.stack([a.cos(), a.sin()], 1) + 0.15 * randn(n, 2)
    if name == "spiral":
        t = rand(n).sqrt() * 3 * math.pi + 0.5
        x = torch.stack([t * t.cos(), t * t.sin()], 1) / (3 * math.pi + 0.5) * 2.2
        return x + 0.05 * randn(n, 2)
    if name == "two moons":
        k = torch.randint(0, 2, (n,), generator=g)
        t = rand(n) * math.pi
        x = torch.where((k == 0)[:, None], torch.stack([t.cos(), t.sin()], 1),
                        torch.stack([1 - t.cos(), 0.5 - t.sin()], 1))
        return (x - torch.tensor([0.5, 0.25])) * 1.7 + 0.07 * randn(n, 2)
    if name == "checkerboard":
        x = rand(n, 2) * 4 - 2
        flip = ((x[:, 0].floor() + x[:, 1].floor()) % 2) != 0
        x[flip, 0] = x[flip, 0] + torch.where(x[flip, 0] < 1, 1.0, -3.0)  # move to a black square
        return x
    if name == "rings":
        r = torch.tensor([0.7, 1.4, 2.1])[torch.randint(0, 3, (n,), generator=g)]
        a = rand(n) * 2 * math.pi
        return r[:, None] * torch.stack([a.cos(), a.sin()], 1) + 0.06 * randn(n, 2)
    if name == "pinwheel":
        k = torch.randint(0, 5, (n,), generator=g)
        r = 0.4 + 1.8 * rand(n)
        a = k * (2 * math.pi / 5) + 0.6 * r + 0.08 * randn(n)
        return r[:, None] * torch.stack([a.cos(), a.sin()], 1)
    raise ValueError(f"unknown toy {name!r}; choose from {TOYS}")


class Embedding:
    """Hide 2D data u in R^D.

    - "identity": x = u (D = 2).
    - "linear":   x = s * P u, P a random D x 2 matrix with orthonormal columns (the JiT toy).
    - "crumpled": x = cos(W u + b), W ~ N(0, omega^2): a curled-up 2D sheet. Nearby points stay
      nearby, but all far-apart points end up roughly equally far apart, like images.
    Both high-D versions are scaled so that each coordinate has variance ~1/2.
    """

    def __init__(self, kind="identity", D=64, omega=4.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.kind, self.omega = kind, omega
        self.D = 2 if kind == "identity" else D
        if kind == "linear":
            self.P = torch.linalg.qr(torch.randn(D, 2, generator=g))[0] * math.sqrt(D / 2) / 2
        elif kind == "crumpled":
            self.W = omega * torch.randn(D, 2, generator=g)
            self.b = 2 * math.pi * torch.rand(D, generator=g)
        elif kind != "identity":
            raise ValueError(kind)
        self._grid = None

    def __call__(self, u):
        if self.kind == "identity":
            return u
        if self.kind == "linear":
            return u @ self.P.T
        return torch.cos(u @ self.W.T + self.b)

    def decode(self, x, lim=2.8, res=160):
        """Back to 2D: the nearest point of the embedded sheet (searched on a res x res grid).
        Returns (u, off) where off is the RMS distance per coordinate from x to the sheet:
        0 means 'exactly on the data manifold', large means 'blurry / noisy'."""
        if self.kind == "identity":
            return x, torch.zeros(len(x))
        if self._grid is None:
            lin = torch.linspace(-lim, lim, res)
            self._grid = torch.stack(torch.meshgrid(lin, lin, indexing="xy"), -1).reshape(-1, 2)
            self._fgrid = self(self._grid)
        u, off = [], []
        for chunk in x.split(512):
            d = torch.cdist(chunk, self._fgrid)
            m, i = d.min(1)
            u.append(self._grid[i])
            off.append(m / math.sqrt(self.D))
        return torch.cat(u), torch.cat(off)
