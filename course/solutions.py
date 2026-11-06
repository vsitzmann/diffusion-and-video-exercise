"""Reference solutions for all exercises. Try the exercise first; peek here when stuck."""
import torch

# ---------------------------------------------------------------- notebook 1


def regression_loss(G, eps, x):
    return ((G(eps) - x) ** 2).sum(1).mean()


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


def conditional_path(x, t, eps):
    return (1 - t) * x + t * eps


def conditional_velocity(x_t, x, t):
    # d/dt x_t = eps - x, with eps = (x_t - (1 - t) x) / t
    return (x_t - x) / t


def posterior_weights(x_t, t, X):
    # p(x_i | x_t) ∝ N(x_t; (1-t) x_i, t^2 I): a softmax over the training points
    sq = torch.cdist(x_t, (1 - t) * X) ** 2
    return torch.softmax(-sq / (2 * t ** 2), dim=1)


def marginal_velocity(x_t, t, X):
    w = posterior_weights(x_t, t, X)
    x_hat = w @ X  # posterior mean E[x | x_t]
    return (x_t - x_hat) / t


def flow_matching_loss(net, x):
    B = x.shape[0]
    t = torch.rand(B, 1)
    eps = torch.randn_like(x)
    x_t = (1 - t) * x + t * eps
    v_pred = net.velocity(x_t, t)
    v_target = eps - x
    return ((v_pred - v_target) ** 2).mean()


def euler_sample(velocity, eps, n_steps):
    # backward in time, from noise at t = 1 to data at t = 0
    ts = torch.linspace(1, 0, n_steps + 1)
    xs = [eps]
    x = eps
    for i in range(n_steps):
        h = ts[i] - ts[i + 1]
        x = x - h * velocity(x, ts[i].expand(len(x), 1))
        xs.append(x)
    return torch.stack(xs)


# ---------------------------------------------------------------- notebook 3


def noise_levels(regime, B, T, g=None):
    if regime == "full sequence":
        k = torch.rand(B, 1, generator=g).expand(B, T)
        return k, k
    if regime == "teacher forcing":
        return torch.rand(B, T, generator=g), torch.zeros(B, T)
    if regime == "diffusion forcing":
        k = torch.rand(B, T, generator=g)
        return k, k
    raise ValueError(regime)


def renoise(x, k_ctx, g=None):
    eps = torch.randn(x.shape, generator=g)
    return (1 - k_ctx) * x + k_ctx * eps
