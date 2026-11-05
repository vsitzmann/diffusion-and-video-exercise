"""Reference solutions for notebook 4."""
import torch
import torch.nn.functional as F


def drop(cond, null, p, g=None):
    """Replace each row of cond (B, ...) by null with probability p (a number or a (B,) tensor)."""
    B = cond.shape[0]
    mask = torch.rand(B, generator=g, device=cond.device) < p
    return torch.where(mask.view(B, *[1] * (cond.ndim - 1)), null, cond)


def guide(x_cond, x_uncond, w):
    """Classifier-free guidance on the clean-frame predictions."""
    return x_uncond + w * (x_cond - x_uncond)


def idm_loss(idm, frame_a, frame_b, actions):
    """Cross-entropy between the IDM's logits for (frame_a -> frame_b) and the true actions."""
    return F.cross_entropy(idm(frame_a, frame_b), actions)


def read_actions(idm, frames):
    """frames (B, H + 1, 3, r, r): the last real frame and H imagined ones -> actions (B, H)."""
    B, T = frames.shape[:2]
    logits = idm(frames[:, :-1].flatten(0, 1), frames[:, 1:].flatten(0, 1))
    return logits.argmax(-1).view(B, T - 1)
