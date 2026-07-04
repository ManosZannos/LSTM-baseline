"""
plot_trajectories.py — Ποιοτική ανάλυση: εντοπισμός σεναρίου σύγκλισης
(convergence scenario) στο test set και σχεδίαση τροχιών παρατήρησης /
πραγματικής συνέχειας / πρόβλεψης, για LSTM vs CPA-GRN v4 (10min task).

Γιατί "σενάριο σύγκλισης";
    Η κεντρική υπόθεση της εργασίας είναι ότι το CPA-GRN υπερτερεί ακριβώς
    σε καταστάσεις όπου δύο πλοία έχουν μικρό DCPA / θετικό, εύλογο TCPA
    (πραγματικός κίνδυνος μελλοντικής σύγκρουσης), όχι απλώς σε τυχαίες
    σκηνές. Ένα τυχαίο σενάριο σχεδόν σίγουρα δεν θα δείξει οπτικά καμία
    διαφορά, αφού και τα δύο μοντέλα κάνουν καλή δουλειά σε "εύκολα",
    μη-αλληλεπιδρώντα σενάρια.

Μέθοδος εντοπισμού:
    Υπολογίζεται DCPA/TCPA (ίδιος τύπος με CPAFeatures του model_cpagrn.py)
    για κάθε ζεύγος πλοίων, στο τελευταίο παρατηρούμενο timestep κάθε
    σκηνής του test set, ΧΩΡΙΣ να χρειαστεί να φορτωθεί κανένα μοντέλο
    (καθαρά γεωμετρικός υπολογισμός πάνω στα raw δεδομένα). Κρατάμε τα
    ζεύγη με:
      - θετικό TCPA εντός εύλογου εύρους (πραγματική μελλοντική σύγκλιση,
        όχι ήδη περασμένη προσέγγιση ή εξωπραγματικά μακρινή),
      - επαρκή ταχύτητα και για τα δύο πλοία (αποφυγή artifacts από σχεδόν
        ακίνητα πλοία όπου το TCPA γίνεται αριθμητικά ασταθές παρά το clamp),
    και ταξινομούμε κατά αύξον DCPA (πιο επικίνδυνο σενάριο πρώτο).

Usage:
    python plot_trajectories.py --gpu_num 0
    python plot_trajectories.py --gpu_num 0 --include_smchn
    python plot_trajectories.py --gpu_num 0 --top_n 5   # αποθηκεύει τα 5 καλύτερα υποψήφια σενάρια
"""
from __future__ import annotations
import os
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import AISDataset, denorm


# ── CPA γεωμετρία (ίδιος τύπος με model_cpagrn.CPAFeatures, σε numpy) ─────────

def pairwise_dcpa_tcpa(pos: np.ndarray, vel: np.ndarray, eps: float = 1e-6):
    """
    pos, vel: [N, 2]  (z-score space, ίδιο convention με το μοντέλο)
    Returns: dcpa [N,N], tcpa [N,N]  (ίδιο clamping με CPAFeatures)
    """
    N = pos.shape[0]
    pos_i = pos[:, None, :]
    pos_j = pos[None, :, :]
    vel_i = vel[:, None, :]
    vel_j = vel[None, :, :]

    r = pos_j - pos_i
    v = vel_j - vel_i

    v_sq = (v * v).sum(-1) + eps
    tcpa = np.clip(-(r * v).sum(-1) / v_sq, -5.0, 5.0)
    dcpa = np.clip(np.linalg.norm(r + tcpa[..., None] * v, axis=-1), 0.0, 10.0)
    return dcpa, tcpa


def find_convergence_scenes(test_ds, min_tcpa=0.5, max_tcpa=8.0,
                             min_speed=0.05, top_n=5):
    """
    Σαρώνει όλο το test set (χωρίς μοντέλα) και επιστρέφει τα top_n σενάρια
    με μικρότερο DCPA μεταξύ ενός ζεύγους πλοίων, υπό τους περιορισμούς
    πραγματικής μελλοντικής σύγκλισης.

    Returns: list of dict {scene_idx, i, j, dcpa, tcpa}
    """
    candidates = []
    for idx in range(len(test_ds)):
        obs = test_ds.obs_list[idx]           # [N, T_obs, 4] numpy, z-score
        N = obs.shape[0]
        if N < 2:
            continue

        pos_last = obs[:, -1, :2]
        pos_prev = obs[:, -2, :2]
        vel_last = pos_last - pos_prev

        speed = np.linalg.norm(vel_last, axis=-1)          # [N]
        dcpa, tcpa = pairwise_dcpa_tcpa(pos_last, vel_last)  # [N,N] each

        np.fill_diagonal(dcpa, np.inf)

        speed_ok = (speed[:, None] > min_speed) & (speed[None, :] > min_speed)
        tcpa_ok  = (tcpa > min_tcpa) & (tcpa < max_tcpa)
        valid    = speed_ok & tcpa_ok

        dcpa_masked = np.where(valid, dcpa, np.inf)
        if not np.isfinite(dcpa_masked).any():
            continue

        i, j = np.unravel_index(np.argmin(dcpa_masked), dcpa_masked.shape)
        candidates.append({
            'scene_idx': idx, 'i': int(i), 'j': int(j),
            'dcpa': float(dcpa_masked[i, j]), 'tcpa': float(tcpa[i, j]),
        })

    candidates.sort(key=lambda c: c['dcpa'])
    return candidates[:top_n]


# ── Model loaders (ίδιο pattern με measure_inference.py) ────────────────────

def load_lstm(tag, pred_len, device):
    from model_lstm import VanillaLSTM
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = VanillaLSTM(feature_size=4,
                    hidden_size=saved.get('hidden_size', 64),
                    num_layers=saved.get('num_layers', 1),
                    pred_len=saved.get('pred_len', pred_len)).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


def load_cpagrn(tag, pred_len, device):
    from model_cpagrn import CPAGRN
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = CPAGRN(feature_size=4,
               d_model=saved.get('d_model', 64),
               gru_layers=saved.get('gru_layers', 1),
               pred_len=saved.get('pred_len', pred_len)).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


def make_identity(T: int, N: int, device):
    identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
    identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
    return [identity_spatial, identity_temporal]


def load_smchn(tag, obs_len, pred_len, device):
    from model_smchn import TrajectoryModel
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    m = TrajectoryModel(
        number_asymmetric_conv_layer = saved.get('number_asymmetric_conv_layer', 2),
        embedding_dims               = saved.get('embedding_dims', 64),
        number_gcn_layers            = saved.get('number_gcn_layers', 1),
        dropout                      = 0.0,
        obs_len                      = saved.get('obs_len', obs_len),
        pred_len                     = saved.get('pred_len', pred_len),
        out_dims                     = 5,
        num_heads                    = saved.get('num_heads', 4),
    ).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


def smchn_predict_abs(model, obs_t, device):
    """
    obs_t: [1, N, T_obs, 4] tensor (όλη η σκηνή, ήδη στο device)
    Returns: pred_abs [T_pred, N, 2]  (z-score absolute LON/LAT), ίδιο
    convention με evaluate_smchn.py.
    """
    N = obs_t.shape[1]
    T_obs = obs_t.shape[2]

    abs_obs = obs_t[0].permute(1, 0, 2)          # [T_obs, N, 4]
    rel_obs = torch.zeros_like(abs_obs)
    rel_obs[1:] = abs_obs[1:] - abs_obs[:-1]

    pos_idx = torch.arange(1, T_obs + 1, device=device, dtype=torch.float32)
    pos_idx = pos_idx.view(T_obs, 1, 1).expand(T_obs, N, 1)

    V_obs = torch.cat([pos_idx, rel_obs], dim=-1).unsqueeze(0)  # [1,T_obs,N,5]
    identity = make_identity(T_obs, N, device)

    V_pred = model(V_obs, identity)              # [T_pred, N, 5] Gaussian params over VELOCITY
    mu_vel = V_pred[:, :, :2]
    last_obs_pos = abs_obs[-1, :, :2]
    mu_abs = torch.cumsum(mu_vel, dim=0) + last_obs_pos.unsqueeze(0)  # [T_pred,N,2]
    return mu_abs


# ── Πρόβλεψη + denormalization για ένα σενάριο ──────────────────────────────

def predict_scene(test_ds, scene_idx, lstm, cpagrn, smchn, device, stats):
    obs_np  = test_ds.obs_list[scene_idx]    # [N, T_obs, 4]
    pred_np = test_ds.pred_list[scene_idx]   # [N, T_pred, 2]
    N = obs_np.shape[0]

    obs_t  = torch.from_numpy(obs_np).unsqueeze(0).float().to(device)   # [1,N,T_obs,4]
    mask_t = torch.ones(1, N, dtype=torch.bool, device=device)

    last_obs = obs_t[:, :, -1, :2]           # [1,N,2]

    with torch.no_grad():
        disp_lstm   = lstm(obs_t, mask=mask_t)                       # [1,N,T_pred,2]
        disp_cpagrn = cpagrn(obs_t, mask=mask_t)                     # [1,N,T_pred,2]
        abs_lstm    = (disp_lstm   + last_obs.unsqueeze(2))[0].cpu().numpy()   # [N,T_pred,2]
        abs_cpagrn  = (disp_cpagrn + last_obs.unsqueeze(2))[0].cpu().numpy()

        abs_smchn = None
        if smchn is not None:
            mu_abs = smchn_predict_abs(smchn, obs_t, device)         # [T_pred,N,2]
            abs_smchn = mu_abs.permute(1, 0, 2).cpu().numpy()        # [N,T_pred,2]

    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    def to_degrees(arr_zscore):
        lon = denorm(arr_zscore[..., 0], lon_mean, lon_std)
        lat = denorm(arr_zscore[..., 1], lat_mean, lat_std)
        return lon, lat

    result = {
        'obs_lonlat':    to_degrees(obs_np[..., :2]),     # ([N,T_obs], [N,T_obs])
        'gt_lonlat':     to_degrees(pred_np),              # ([N,T_pred], [N,T_pred])
        'lstm_lonlat':   to_degrees(abs_lstm),
        'cpagrn_lonlat': to_degrees(abs_cpagrn),
    }
    if abs_smchn is not None:
        result['smchn_lonlat'] = to_degrees(abs_smchn)
    return result


def ade_degrees(pred_lon, pred_lat, true_lon, true_lat, vessel_idx):
    err = np.sqrt((pred_lon[vessel_idx] - true_lon[vessel_idx])**2 +
                  (pred_lat[vessel_idx] - true_lat[vessel_idx])**2)
    return float(err.mean())


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_scene(result, i, j, dcpa, tcpa, out_path, include_smchn):
    obs_lon, obs_lat   = result['obs_lonlat']
    gt_lon,  gt_lat    = result['gt_lonlat']
    lstm_lon, lstm_lat = result['lstm_lonlat']
    cpa_lon,  cpa_lat  = result['cpagrn_lonlat']

    fig, ax = plt.subplots(figsize=(8, 7))

    # Context: όλα τα άλλα πλοία της σκηνής, μόνο η τελευταία παρατηρούμενη
    # θέση τους, ως αχνά γκρι σημεία (δείχνει την πυκνότητα του σκηνικού
    # χωρίς να γεμίζει το plot με τροχιές που δεν μας ενδιαφέρουν).
    N = obs_lon.shape[0]
    other = [k for k in range(N) if k not in (i, j)]
    ax.scatter(obs_lon[other, -1], obs_lat[other, -1],
               c='lightgray', s=12, zorder=1, label='Άλλα πλοία σκηνής (τελ. θέση)')

    colors = {'i': 'tab:blue', 'j': 'tab:orange'}
    for vessel, color, label in [(i, colors['i'], 'Πλοίο A'), (j, colors['j'], 'Πλοίο B')]:
        # Παρατήρηση (obs)
        ax.plot(obs_lon[vessel], obs_lat[vessel], '-o', color=color,
                linewidth=2, markersize=4, zorder=3,
                label=f'{label} — παρατήρηση')
        # Πραγματική συνέχεια (ground truth)
        full_gt_lon = np.concatenate([[obs_lon[vessel, -1]], gt_lon[vessel]])
        full_gt_lat = np.concatenate([[obs_lat[vessel, -1]], gt_lat[vessel]])
        ax.plot(full_gt_lon, full_gt_lat, '-', color=color, linewidth=2,
                alpha=0.9, zorder=3, label=f'{label} — πραγματική τροχιά')
        # LSTM πρόβλεψη
        full_lstm_lon = np.concatenate([[obs_lon[vessel, -1]], lstm_lon[vessel]])
        full_lstm_lat = np.concatenate([[obs_lat[vessel, -1]], lstm_lat[vessel]])
        ax.plot(full_lstm_lon, full_lstm_lat, '--', color=color, linewidth=1.6,
                alpha=0.7, zorder=2, label=f'{label} — LSTM πρόβλεψη')
        # CPA-GRN πρόβλεψη
        full_cpa_lon = np.concatenate([[obs_lon[vessel, -1]], cpa_lon[vessel]])
        full_cpa_lat = np.concatenate([[obs_lat[vessel, -1]], cpa_lat[vessel]])
        ax.plot(full_cpa_lon, full_cpa_lat, ':', color=color, linewidth=2.2,
                alpha=0.9, zorder=4, label=f'{label} — CPA-GRN πρόβλεψη')

        if include_smchn and 'smchn_lonlat' in result:
            sm_lon, sm_lat = result['smchn_lonlat']
            full_sm_lon = np.concatenate([[obs_lon[vessel, -1]], sm_lon[vessel]])
            full_sm_lat = np.concatenate([[obs_lat[vessel, -1]], sm_lat[vessel]])
            ax.plot(full_sm_lon, full_sm_lat, '-.', color=color, linewidth=1.4,
                    alpha=0.6, zorder=2, label=f'{label} — SMCHN πρόβλεψη')

        # Marker στο τελευταίο σημείο παρατήρησης (σημείο σύγκλισης)
        ax.scatter([obs_lon[vessel, -1]], [obs_lat[vessel, -1]],
                   c=color, s=90, marker='*', zorder=5, edgecolors='black')

    ade_lstm_i = ade_degrees(lstm_lon, lstm_lat, gt_lon, gt_lat, i)
    ade_cpa_i  = ade_degrees(cpa_lon,  cpa_lat,  gt_lon, gt_lat, i)
    ade_lstm_j = ade_degrees(lstm_lon, lstm_lat, gt_lon, gt_lat, j)
    ade_cpa_j  = ade_degrees(cpa_lon,  cpa_lat,  gt_lon, gt_lat, j)

    title = (f'Σενάριο σύγκλισης — DCPA={dcpa:.4f}° TCPA={tcpa:.1f} min '
             f'(z-score timesteps)\n'
             f'ADE Πλοίο A: LSTM={ade_lstm_i:.5f}° CPA-GRN={ade_cpa_i:.5f}°   |   '
             f'ADE Πλοίο B: LSTM={ade_lstm_j:.5f}° CPA-GRN={ade_cpa_j:.5f}°')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Γεωγραφικό μήκος (LON, °)')
    ax.set_ylabel('Γεωγραφικό πλάτος (LAT, °)')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()

    fig.savefig(out_path + '.png', dpi=300)
    fig.savefig(out_path + '.pdf')
    plt.close(fig)

    return {
        'ade_lstm_i': ade_lstm_i, 'ade_cpa_i': ade_cpa_i,
        'ade_lstm_j': ade_lstm_j, 'ade_cpa_j': ade_cpa_j,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_num',   type=int, default=0)
    p.add_argument('--obs_len',   type=int, default=10)
    p.add_argument('--pred_len',  type=int, default=10)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--data_dir',  type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--out_dir',   type=str, default='figures/convergence')
    p.add_argument('--top_n',     type=int, default=3,
                   help='Πόσα υποψήφια σενάρια σύγκλισης να αποθηκευτούν (ταξινομημένα κατά DCPA)')
    p.add_argument('--min_tcpa',  type=float, default=0.5)
    p.add_argument('--max_tcpa',  type=float, default=8.0)
    p.add_argument('--min_speed', type=float, default=0.05)
    p.add_argument('--include_smchn', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, 'global_stats.json')) as f:
        stats = json.load(f)

    print('Φόρτωση test dataset...')
    test_ds = AISDataset(os.path.join(args.data_dir, 'test'),
                          args.obs_len, args.pred_len,
                          stride=args.obs_len + args.pred_len)
    print(f'  Test set: {len(test_ds):,} σκηνές\n')

    print('Αναζήτηση σεναρίων σύγκλισης (καθαρά γεωμετρικά, χωρίς μοντέλα)...')
    candidates = find_convergence_scenes(
        test_ds, min_tcpa=args.min_tcpa, max_tcpa=args.max_tcpa,
        min_speed=args.min_speed, top_n=args.top_n,
    )
    if not candidates:
        raise RuntimeError(
            'Δεν βρέθηκε κανένα σενάριο σύγκλισης με τους δεδομένους '
            'περιορισμούς (--min_tcpa/--max_tcpa/--min_speed). Δοκίμασε να '
            'χαλαρώσεις τα thresholds.'
        )
    print(f'  Βρέθηκαν {len(candidates)} υποψήφια σενάρια (ταξινομημένα κατά DCPA):')
    for c in candidates:
        print(f"    scene={c['scene_idx']:>4}  i={c['i']:>3} j={c['j']:>3}  "
              f"DCPA={c['dcpa']:.4f}°  TCPA={c['tcpa']:.2f}")

    lstm_tag   = f'LSTM_obs{args.obs_len}_pred{args.pred_len}'
    cpagrn_tag = f'CPAGRN_obs{args.obs_len}_pred{args.pred_len}_s{args.seed}'
    smchn_tag  = f'SMCHN_obs{args.obs_len}_pred{args.pred_len}_s{args.seed}'

    print('\nΦόρτωση μοντέλων...')
    lstm   = load_lstm(lstm_tag, args.pred_len, device)
    cpagrn = load_cpagrn(cpagrn_tag, args.pred_len, device)
    smchn  = None
    if args.include_smchn:
        try:
            smchn = load_smchn(smchn_tag, args.obs_len, args.pred_len, device)
        except Exception as e:
            print(f'  ⚠ SMCHN δεν φορτώθηκε ({e}) — συνεχίζω χωρίς αυτό.')

    print('\nΠαραγωγή plots...')
    for rank, c in enumerate(candidates, 1):
        result = predict_scene(test_ds, c['scene_idx'], lstm, cpagrn, smchn, device, stats)
        out_path = os.path.join(args.out_dir, f'convergence_{rank}_scene{c["scene_idx"]}')
        ades = plot_scene(result, c['i'], c['j'], c['dcpa'], c['tcpa'],
                           out_path, include_smchn=args.include_smchn)
        print(f"  [{rank}] scene={c['scene_idx']} → {out_path}.png")
        print(f"       ADE A: LSTM={ades['ade_lstm_i']:.5f}° CPA-GRN={ades['ade_cpa_i']:.5f}°  "
              f"({'CPA-GRN καλύτερο' if ades['ade_cpa_i']<ades['ade_lstm_i'] else 'LSTM καλύτερο'})")
        print(f"       ADE B: LSTM={ades['ade_lstm_j']:.5f}° CPA-GRN={ades['ade_cpa_j']:.5f}°  "
              f"({'CPA-GRN καλύτερο' if ades['ade_cpa_j']<ades['ade_lstm_j'] else 'LSTM καλύτερο'})")

    print(f'\nΈτοιμο. Έλεγξε τα {args.top_n} candidate plots στο {args.out_dir}/ '
          f'και διάλεξε το πιο καθαρό οπτικά για την εργασία.')


if __name__ == '__main__':
    main()