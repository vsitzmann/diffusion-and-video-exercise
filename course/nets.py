"""Small networks for the 2D / high-D toys."""
import math

import torch
import torch.nn as nn


def mlp(d_in, d_out, width=256, depth=3):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class Generator(nn.Module):
    """One-step generator G_theta: noise z in R^d_noise -> sample in R^D."""

    def __init__(self, D=2, d_noise=2, width=256):
        super().__init__()
        self.d_noise = d_noise
        self.net = mlp(d_noise, D, width)

    def forward(self, z):
        return self.net(z)


class FlowNet(nn.Module):
    """Flow-matching network on R^D, conditioned on the time t in [0, 1].

    pred="x": the network outputs a guess of the clean data point, z_hat = E[z | X_t = x], and the
              velocity is computed from it: u = (z_hat - x) / (1 - t). This is JiT's choice
              (Li & He, 2025) and works much better when the data lives on a thin sheet in high D.
    pred="v": the network outputs the velocity directly.
    """

    def __init__(self, D=2, width=256, depth=3, pred="x", t_clip=0.05):
        super().__init__()
        self.pred, self.t_clip = pred, t_clip
        self.net = mlp(D + 16, D, width, depth)

    def _t_features(self, t):
        f = torch.exp(torch.linspace(0, math.log(100), 8))
        a = t * f
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self._t_features(t)], -1))

    def velocity(self, x, t):
        out = self(x, t)
        if self.pred == "v":
            return out
        return (out - x) / (1 - t).clamp(min=self.t_clip)

    def denoise(self, x, t):
        """The network's guess of the clean data point."""
        out = self(x, t)
        if self.pred == "x":
            return out
        return x + (1 - t) * out
