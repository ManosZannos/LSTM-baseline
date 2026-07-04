"""
visualize_attention.py — Attention weight visualization for CPA-GRN.

Shows which neighbors each vessel attends to at each timestep,
overlaid on the actual vessel positions. Useful as qualitative
evidence that the model learns to focus on vessels in convergence.

Usage:
    python visualize_attention.py --gpu_num 0 --out attention.pdf
"""
import os, argparse, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_dataloaders, denorm


# ── Patched model that captures attention weights ─────────────────────────────

class NeighborAggregationViz(nn.Module):
    """Same as NeighborAggregation but stores attention weights."""

    def __init__(self, d_model, edge_dim=7, top_k=10):
        super().__init__()
        self.top_k = top_k
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.msg_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.last_weights = None   # captured here

    def forward(self, h, edges, mask):
        B, N, D = h.shape
        h_j    = h.unsqueeze(1).expand(B, N, N, D)
        scores = self.attn_mlp(torch.cat([h_j, edges], dim=-1)).squeeze(-1)

        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        dist = edges[..., 2]
        if mask is not None:
            dist_masked = dist.masked_fill(~mask_j, float('inf'))
        else:
            dist_masked = dist

        if self.top_k < N:
            k = min(self.top_k, N)
            kth, _ = dist_masked.topk(k, dim=-1, largest=False)
            threshold = kth[..., -1].unsqueeze(-1)
            scores = scores.masked_fill(dist_masked > threshold, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        self.last_weights = weights.detach().cpu()   # ← captured

        msgs = self.msg_proj(h)
        agg  = torch.einsum('bij,bjd->bid', weights, msgs)
        return self.norm(self.out_proj(agg))


def build_viz_model(original_ckpt_path, device):
    """
    Load a CPAGRN with NeighborAggregationViz instead of NeighborAggregation.
    Weights are copied from the original checkpoint.
    """
    from model_cpagrn import CPAGRN, CPAFeatures

    ckpt  = torch.load(original_ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    d_model  = saved.get('d_model', 64)
    pred_len = saved.get('pred_len', 10)

    # Build standard model and load weights
    model = CPAGRN(feature_size=4, d_model=d_model,
                   gru_layers=1, pred_len=pred_len).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Replace NeighborAggregation with the visualization version
    def swap(mod, edge_dim):
        viz = NeighborAggregationViz(d_model=d_model,
                                     edge_dim=edge_dim, top_k=10).to(device)
        # Copy weights
        viz.attn_mlp.load_state_dict(mod.attn_mlp.state_dict())
        viz.msg_proj.load_state_dict(mod.msg_proj.state_dict())
        viz.out_proj.load_state_dict(mod.out_proj.state_dict())
        viz.norm.load_state_dict(mod.norm.state_dict())
        return viz

    model.neighbor_agg  = swap(model.neighbor_agg,  edge_dim=7)
    model.final_spatial = swap(model.final_spatial, edge_dim=7)

    print(f'Loaded {original_ckpt_path}  (epoch {ckpt["epoch"]})')
    return model, ckpt.get('stats')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_num',    type=int, default=0)
    p.add_argument('--obs_len',    type=int, default=10)
    p.add_argument('--pred_len',   type=int, default=10)
    p.add_argument('--focus_vessel', type=int, default=0,
                   help='Which vessel (index 0..N-1) to show attention FROM')
    p.add_argument('--out', type=str, default='attention.pdf')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = args.obs_len

    ckpt_path = f'checkpoints/CPAGRN_obs{T}_pred{T}_s42/val_best.pth'
    model, stats = build_viz_model(ckpt_path, device)

    if stats is None:
        with open('dataset/noaa_dec2021_1min/global_stats.json') as f:
            stats = json.load(f)

    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    _, _, test_loader, _ = get_dataloaders(
        'dataset/noaa_dec2021_1min', args.obs_len, args.pred_len, batch_size=1
    )

    # Find a scene with a reasonable number of vessels
    print('Searching for a good scene...')
    chosen = None
    for obs, _, mask, _ in test_loader:
        n_real = mask[0].sum().item()
        if n_real >= 5:
            chosen = (obs.to(device), mask.to(device))
            break
    if chosen is None:
        raise RuntimeError('No scene with ≥5 vessels found in test set.')

    obs_d, mask_d = chosen
    obs_cpu = obs_d.cpu()
    mask_np = mask_d.cpu().numpy()[0].astype(bool)  # [N]

    # Forward pass — captures attention at each per-step aggregation
    with torch.no_grad():
        _ = model(obs_d, mask=mask_d)

    # Collect attention weights per timestep from the per-step aggregator
    # model.neighbor_agg.last_weights is from the LAST t in the loop
    # For full per-timestep capture we need a custom forward; here we use the
    # last-timestep weights as a proxy (the final spatial refinement)
    attn = model.final_spatial.last_weights[0]  # [N, N] from B=1

    # Denormalize positions (last observed timestep)
    pos_z = obs_cpu[0, :, -1, :2].numpy()  # [N, 2] z-score
    pos_lon = denorm(pos_z[:, 0], lon_mean, lon_std)
    pos_lat = denorm(pos_z[:, 1], lat_mean, lat_std)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax_map, ax_hm) = plt.subplots(1, 2, figsize=(12, 5),
                                         constrained_layout=True)

    # Left: spatial attention arrows
    focus = args.focus_vessel
    real_idx = np.where(mask_np)[0]
    if focus >= len(real_idx):
        focus = 0
        print(f'focus_vessel adjusted to 0 (only {len(real_idx)} real vessels)')
    i = real_idx[focus]

    attn_row = attn[i].numpy()   # [N] attention FROM vessel i TO all j

    # All vessel positions
    ax_map.scatter(pos_lon[mask_np], pos_lat[mask_np],
                   c='lightgrey', s=40, zorder=2, label='Other vessels')

    # Color by attention weight
    top_k_idx = np.argsort(attn_row)[::-1][:10]   # top-10 neighbors
    top_k_idx = [j for j in top_k_idx if mask_np[j] and j != i][:10]

    if top_k_idx:
        w = attn_row[top_k_idx]
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-9)
        cmap = cm.get_cmap('Reds')
        for j, wn in zip(top_k_idx, w_norm):
            ax_map.annotate('',
                xy=(pos_lon[j], pos_lat[j]),
                xytext=(pos_lon[i], pos_lat[i]),
                arrowprops=dict(arrowstyle='->', color=cmap(0.3 + 0.7 * wn),
                                lw=1.5 + 2.5 * wn))
            sc = ax_map.scatter(pos_lon[j], pos_lat[j],
                                c=[[cmap(0.3 + 0.7 * wn)]],
                                s=60 + 80 * wn, zorder=3)

    # Focus vessel
    ax_map.scatter(pos_lon[i], pos_lat[i],
                   c='blue', s=120, marker='*', zorder=5, label=f'Vessel {i} (query)')

    ax_map.set_xlabel('Longitude (°)', fontsize=10)
    ax_map.set_ylabel('Latitude (°)',  fontsize=10)
    ax_map.set_title(f'Attention FROM Vessel {i}\n'
                     f'(arrow thickness ∝ attention weight)', fontsize=10)
    ax_map.legend(fontsize=8)
    ax_map.grid(True, alpha=0.3)

    # Right: attention heatmap (N_real × N_real)
    real_attn = attn[np.ix_(real_idx, real_idx)].numpy()
    im = ax_hm.imshow(real_attn, cmap='hot_r', aspect='auto',
                      vmin=0, vmax=real_attn.max())
    ax_hm.set_xlabel('Source vessel j', fontsize=10)
    ax_hm.set_ylabel('Query vessel i',  fontsize=10)
    ax_hm.set_title('Full Attention Matrix\n(real vessels only)', fontsize=10)
    plt.colorbar(im, ax=ax_hm, label='Attention weight')

    # Mark focus vessel
    ax_hm.axhline(focus, color='blue', lw=1.5, ls='--', alpha=0.7)
    ax_hm.axvline(focus, color='blue', lw=1.5, ls='--', alpha=0.7)

    fig.suptitle('CPA-GRN v4 — Spatial Attention Weights (Final Spatial Refinement)\n'
                 f'10-min Horizon | San Diego Test Set',
                 fontsize=11, fontweight='bold')

    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
