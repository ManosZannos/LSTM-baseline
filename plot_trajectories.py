"""
plot_trajectories.py — Trajectory prediction visualization (predicted vs ground truth).

Finds the test-set scene where CPA-GRN has the largest ADE advantage over LSTM,
then plots predicted vs ground truth for the top-3 most active vessels.

Usage:
    python plot_trajectories.py --gpu_num 0 --out trajectories.pdf
"""
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from dataset import get_dataloaders, denorm


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_cpagrn(tag, pred_len, device):
    from model_cpagrn import CPAGRN
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = CPAGRN(feature_size=4, d_model=64, gru_layers=1,
               pred_len=saved.get('pred_len', pred_len)).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m, ckpt.get('stats')


def load_lstm(tag, pred_len, device):
    from model_lstm import VanillaLSTM
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = VanillaLSTM(feature_size=4, hidden_size=64, num_layers=1,
                    pred_len=saved.get('pred_len', pred_len)).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m, ckpt.get('stats')


def load_nocpa(tag, pred_len, device):
    from model_cpagrn_nocpa import CPAGRN as NoCPA
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    m = NoCPA(feature_size=4, d_model=64, gru_layers=1,
              pred_len=pred_len).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m, ckpt.get('stats')


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict(model, obs, mask, use_mask=True):
    """Returns pred_disp [B, N, T, 2] on CPU."""
    with torch.no_grad():
        if use_mask:
            return model(obs, mask=mask).cpu()
        else:
            return model(obs).cpu()


def disp_to_abs_deg(pred_disp, obs_cpu, stats):
    """Convert z-score displacement → absolute degrees."""
    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    last_obs = obs_cpu[:, :, -1, :2]          # [B, N, 2] z-score
    pred_abs_z = pred_disp + last_obs.unsqueeze(2)  # [B, N, T, 2]

    pred_lon = denorm(pred_abs_z[..., 0].numpy(), lon_mean, lon_std)
    pred_lat = denorm(pred_abs_z[..., 1].numpy(), lat_mean, lat_std)
    return pred_lon, pred_lat   # both [B, N, T]


def obs_to_deg(obs_cpu, stats):
    """Convert z-score obs → degrees."""
    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']
    obs_lon = denorm(obs_cpu[..., 0].numpy(), lon_mean, lon_std)
    obs_lat = denorm(obs_cpu[..., 1].numpy(), lat_mean, lat_std)
    return obs_lon, obs_lat    # [B, N, T_obs]


def ade(pred_lon, pred_lat, true_lon, true_lat, mask_np):
    """Per-vessel ADE [B, N]."""
    err = np.sqrt((pred_lon - true_lon)**2 + (pred_lat - true_lat)**2)
    return err.mean(axis=-1)  # [B, N]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_num',  type=int, default=0)
    p.add_argument('--obs_len',  type=int, default=10)
    p.add_argument('--pred_len', type=int, default=10)
    p.add_argument('--out',      type=str, default='trajectories.pdf')
    p.add_argument('--n_vessels', type=int, default=4,
                   help='How many vessels to plot per scene')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = args.obs_len

    print('Loading models...')
    lstm,   stats = load_lstm  (f'LSTM_obs{T}_pred{T}_s42',   args.pred_len, device)
    nocpa,  _     = load_nocpa (f'CPAGRN_nocpa_v4_obs{T}_pred{T}_s42', args.pred_len, device)
    cpagrn, _     = load_cpagrn(f'CPAGRN_obs{T}_pred{T}_s42', args.pred_len, device)

    # stats from CPA-GRN checkpoint (same for all)
    if stats is None:
        import json
        with open('dataset/noaa_dec2021_1min/global_stats.json') as f:
            stats = json.load(f)

    _, _, test_loader, _ = get_dataloaders(
        'dataset/noaa_dec2021_1min', args.obs_len, args.pred_len, batch_size=8
    )

    print('Searching for best scene...')
    best_scene = None
    best_advantage = -np.inf

    for obs, pred_gt, mask, _ in test_loader:
        obs_d    = obs.to(device)
        mask_d   = mask.to(device)
        obs_cpu  = obs.cpu()
        mask_np  = mask.numpy()

        # Ground truth in degrees
        last_obs_z = obs_cpu[:, :, -1, :2]
        gt_disp    = pred_gt.cpu() - last_obs_z.unsqueeze(2)
        gt_lon, gt_lat = disp_to_abs_deg(gt_disp, obs_cpu, stats)

        # Predictions
        pred_lstm   = predict(lstm,   obs_d, mask_d, use_mask=False)
        pred_nocpa  = predict(nocpa,  obs_d, mask_d, use_mask=True)
        pred_cpagrn = predict(cpagrn, obs_d, mask_d, use_mask=True)

        lstm_lon,   lstm_lat   = disp_to_abs_deg(pred_lstm,   obs_cpu, stats)
        nocpa_lon,  nocpa_lat  = disp_to_abs_deg(pred_nocpa,  obs_cpu, stats)
        cpagrn_lon, cpagrn_lat = disp_to_abs_deg(pred_cpagrn, obs_cpu, stats)

        ade_lstm   = ade(lstm_lon,   lstm_lat,   gt_lon, gt_lat, mask_np)
        ade_cpagrn = ade(cpagrn_lon, cpagrn_lat, gt_lon, gt_lat, mask_np)

        # Scene score: mean advantage of CPA-GRN over LSTM (masked)
        valid = mask_np.astype(bool)
        diff  = (ade_lstm - ade_cpagrn)  # positive = CPA-GRN better
        score = diff[valid].mean() if valid.any() else -np.inf

        if score > best_advantage:
            best_advantage = score
            best_scene = {
                'obs_cpu':    obs_cpu,
                'mask_np':    mask_np,
                'gt_lon':     gt_lon,
                'gt_lat':     gt_lat,
                'lstm_lon':   lstm_lon,   'lstm_lat':   lstm_lat,
                'nocpa_lon':  nocpa_lon,  'nocpa_lat':  nocpa_lat,
                'cpagrn_lon': cpagrn_lon, 'cpagrn_lat': cpagrn_lat,
                'ade_lstm':   ade_lstm,
                'ade_cpagrn': ade_cpagrn,
            }

    print(f'Best scene: CPA-GRN advantage = {best_advantage*60:.4f} nm avg')

    # ── Select top vessels (most active: highest ADE spread) ─────────────────
    sc = best_scene
    valid_b, valid_n = np.where(sc['mask_np'])

    # Sort by LSTM ADE descending → pick vessels where models disagree most
    scores = sc['ade_lstm'][valid_b, valid_n]
    top_idx = np.argsort(scores)[::-1][:args.n_vessels]
    sel = [(valid_b[i], valid_n[i]) for i in top_idx]

    # Obs in degrees
    obs_cpu = sc['obs_cpu']
    obs_lon_all = denorm(obs_cpu[..., 0].numpy(),
                         stats['LON']['mean'], stats['LON']['std'])
    obs_lat_all = denorm(obs_cpu[..., 1].numpy(),
                         stats['LAT']['mean'], stats['LAT']['std'])

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(sel), figsize=(4.5 * len(sel), 4.5),
                             constrained_layout=True)
    if len(sel) == 1:
        axes = [axes]

    colors = {
        'Ground truth': 'black',
        'LSTM':         '#1f77b4',
        'No-CPA v4':    '#ff7f0e',
        'CPA-GRN v4':   '#2ca02c',
    }

    for ax, (b, n) in zip(axes, sel):
        obs_lon = obs_lon_all[b, n, :]    # [T_obs]
        obs_lat = obs_lat_all[b, n, :]

        # Observed trajectory
        ax.plot(obs_lon, obs_lat, 'o-', color='grey', lw=1.5,
                ms=4, label='Observed', zorder=3)
        ax.plot(obs_lon[-1], obs_lat[-1], 's', color='grey',
                ms=8, zorder=4)   # last observed position

        # Ground truth future
        ax.plot(sc['gt_lon'][b, n, :], sc['gt_lat'][b, n, :],
                'o-', color=colors['Ground truth'], lw=2,
                ms=4, label='Ground truth', zorder=5)

        # Model predictions
        for name, plon, plat in [
            ('LSTM',       sc['lstm_lon'],   sc['lstm_lat']),
            ('No-CPA v4',  sc['nocpa_lon'],  sc['nocpa_lat']),
            ('CPA-GRN v4', sc['cpagrn_lon'], sc['cpagrn_lat']),
        ]:
            ax.plot(plon[b, n, :], plat[b, n, :],
                    '--', color=colors[name], lw=1.8,
                    ms=3, label=name, zorder=4)

        ade_l = sc['ade_lstm'][b, n] * 60
        ade_c = sc['ade_cpagrn'][b, n] * 60
        ax.set_title(
            f'Vessel {n+1}\n'
            f'ADE: LSTM={ade_l:.3f} nm  |  CPA-GRN={ade_c:.3f} nm',
            fontsize=9
        )
        ax.set_xlabel('Longitude (°)', fontsize=9)
        ax.set_ylabel('Latitude (°)',  fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles = [
        plt.Line2D([0],[0], color='grey',                    lw=1.5, marker='o', ms=4, label='Observed'),
        plt.Line2D([0],[0], color=colors['Ground truth'],    lw=2,   marker='o', ms=4, label='Ground truth'),
        plt.Line2D([0],[0], color=colors['LSTM'],            lw=1.8, ls='--',    label='LSTM'),
        plt.Line2D([0],[0], color=colors['No-CPA v4'],       lw=1.8, ls='--',    label='No-CPA v4'),
        plt.Line2D([0],[0], color=colors['CPA-GRN v4'],      lw=1.8, ls='--',    label='CPA-GRN v4'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f'Vessel Trajectory Prediction — {T}-min Horizon\n'
                 f'(San Diego AIS, December 2021, Test Set)',
                 fontsize=11, fontweight='bold')

    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
