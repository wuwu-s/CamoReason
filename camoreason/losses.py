"""Paper losses: Eq.7 (InfoNCE contrastive), Eq.23 (gate regularization),
Eq.29-31 (oracle selection, focal+dice segmentation, MSE IoU regression).

All functions are eps-guarded: they must stay finite on empty-GT samples.
"""
import torch
import torch.nn.functional as F


def focal_loss(logits, gt, alpha=0.25, gamma=2.0):
    """Focal loss on binary logits. logits/gt: [B,K,H,W]."""
    p = torch.sigmoid(logits)
    pt = p * gt + (1 - p) * (1 - gt)
    w = alpha * gt + (1 - alpha) * (1 - gt)
    bce = gt * torch.log(p.clamp_min(1e-6)) + (1 - gt) * torch.log((1 - p).clamp_min(1e-6))
    return -(w * (1 - pt) ** gamma * bce)


def dice_loss(logits, gt, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * gt).sum(dim=(-2, -1))
    union = p.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
    return 1 - (2 * inter + eps) / (union + eps)


def seg_loss_per_mask(logits, gt):
    """Per-mask L_seg = focal + dice (Eq.29 input). Returns [B,K]."""
    return focal_loss(logits, gt).mean(dim=(-2, -1)) + dice_loss(logits, gt)


def iou_of(logits, gt):
    """Soft IoU between probability mask and GT (q_gt, patent Eq.43).
    logits [B,K,H,W] -> [B,K]."""
    p = torch.sigmoid(logits)
    inter = (p * gt).sum(dim=(-2, -1))
    union = p.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) - inter
    return inter / union.clamp_min(1e-6)


def info_nce(z_pos, z_neg, batch_t_pos, batch_t_neg, w_n, tau=0.07):
    """Eq.7 (paper): positive = z+_i vs t+_i; negatives N_i = {z-_i} U in-batch
    NEGATIVE text embeddings t-_j (j != i). In-batch targets do NOT act as
    embedding-level negatives (in the paper they enter the negative prompt
    text via Eq.4 instead). Text enters the contrastive through the paper's
    own negative projection W_n; cosine sim = dot on L2-normalized vectors."""
    z_pos = F.normalize(z_pos, dim=-1)
    z_neg = F.normalize(z_neg, dim=-1)
    B = z_pos.shape[0]
    t_pos_n = F.normalize(w_n(batch_t_pos), dim=-1)  # [B,512]
    t_neg_n = F.normalize(w_n(batch_t_neg), dim=-1)  # [B,512]
    pos = (z_pos * t_pos_n).sum(-1, keepdim=True) / tau  # own t+ [B,1]
    own_neg = (z_pos * z_neg).sum(-1, keepdim=True) / tau  # z-_i [B,1]
    sim = z_pos @ t_neg_n.t() / tau  # [B,B] vs in-batch negative text embeddings
    # own t-_i is already counted once via own_neg (z-_i), so exclude it here
    idx = torch.arange(B, device=z_pos.device)
    sim = sim.masked_fill(torch.eye(B, dtype=torch.bool, device=z_pos.device), float("-inf"))
    logits = torch.cat([pos, own_neg, sim], dim=1)  # [B, 1+1+B]
    labels = torch.zeros(B, dtype=torch.long, device=z_pos.device)
    return F.cross_entropy(logits, labels)


def gate_reg(alphas, lambda_tv=0.1, u=None):
    """Eq.23: L_gate = sum_s ||mean(alpha_s) - u||^2 + lambda_tv * sum TV(alpha).
    alphas: list of [B,M,H,W] per level; u defaults to the uniform prior 1/M."""
    if u is None:
        M = alphas[0].shape[1]
        u = torch.full((M,), 1.0 / M, device=alphas[0].device, dtype=alphas[0].dtype)
    loss = alphas[0].new_zeros(())
    for a in alphas:
        mean_a = a.mean(dim=(0, 2, 3))
        loss = loss + ((mean_a - u) ** 2).sum()
        dx = (a[..., 1:, :] - a[..., :-1, :]).abs().mean()
        dy = (a[..., :, 1:] - a[..., :, :-1]).abs().mean()
        loss = loss + lambda_tv * (dx + dy)
    return loss / len(alphas)
