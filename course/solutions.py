"""Reference solutions for all exercises. Try the exercise first; peek here when stuck."""
import torch

# ---------------------------------------------------------------- notebook 1


def regression_loss(G, z, x):
    return ((G(z) - x) ** 2).sum(1).mean()


def nearest_neighbor(y, x):
    return x[torch.cdist(y, x).argmin(1)]


def repulsion(y, tau):
    d = torch.cdist(y, y)
    logits = -d / tau
    logits.fill_diagonal_(float("-inf"))  # a sample does not repel itself
    w = torch.softmax(logits, dim=1)
    return y - w @ y  # = sum_j w_ij (y_i - y_j), since the weights sum to 1


def drift_target(y, x, lam, tau):
    V = nearest_neighbor(y, x) - y + lam * repulsion(y, tau)
    return y + V


# ---------------------------------------------------------------- notebook 2


def conditional_path(z, t, eps):
    return t * z + (1 - t) * eps


def conditional_velocity(x, z, t):
    return (z - x) / (1 - t)


def posterior_weights(x, t, Z):
    # p(z_i | x) ∝ N(x; t z_i, (1-t)^2 I): a softmax over the training points
    sq = torch.cdist(x, t * Z) ** 2
    return torch.softmax(-sq / (2 * (1 - t) ** 2), dim=1)


def marginal_velocity(x, t, Z):
    w = posterior_weights(x, t, Z)
    z_hat = w @ Z  # posterior mean E[z | X_t = x]
    return (z_hat - x) / (1 - t)


def flow_matching_loss(net, z, t_min=0.0):
    B = z.shape[0]
    t = torch.rand(B, 1) * (1 - t_min) + t_min
    eps = torch.randn_like(z)
    x = t * z + (1 - t) * eps
    v_pred = net.velocity(x, t)
    v_target = z - eps
    return ((v_pred - v_target) ** 2).mean()


def euler_sample(velocity, x0, n_steps):
    ts = torch.linspace(0, 1, n_steps + 1)
    xs = [x0]
    x = x0
    for i in range(n_steps):
        h = ts[i + 1] - ts[i]
        x = x + h * velocity(x, ts[i].expand(len(x), 1))
        xs.append(x)
    return torch.stack(xs)


# ---------------------------------------------------------------- notebook 3


def noise_levels(regime, B, T, g=None):
    if regime == "full sequence":
        k = torch.rand(B, 1, generator=g).expand(B, T)
        return k, k
    if regime == "teacher forcing":
        return torch.rand(B, T, generator=g), torch.ones(B, T)
    if regime == "diffusion forcing":
        k = torch.rand(B, T, generator=g)
        return k, k
    raise ValueError(regime)


def renoise(x, k_ctx, g=None):
    eps = torch.randn(x.shape, generator=g)
    return k_ctx * x + (1 - k_ctx) * eps
