"""
measure_inference.py — Inference time per scene για τα 4 μοντέλα (10min task).

Usage:
    python measure_inference.py --gpu_num 0
    python measure_inference.py --gpu_num 0 --obs_len 5 --pred_len 5
"""
import os, time, argparse
import numpy as np
import torch
from dataset import get_dataloaders


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    print(f'  Loaded {tag}  (epoch {ckpt["epoch"]})')
    return m


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
    print(f'  Loaded {tag}  (epoch {ckpt["epoch"]})')
    return m


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def measure(model, batches, device, use_mask=True,
            n_warmup=10, n_measure=50):
    """Returns mean ± std of wall-clock ms per sample (scene)."""
    with torch.no_grad():
        # Warmup
        for obs, mask in batches[:n_warmup]:
            _ = model(obs, mask=mask) if use_mask else model(obs)
        sync(device)

        # Timed runs
        times = []
        for obs, mask in batches[n_warmup : n_warmup + n_measure]:
            sync(device)
            t0 = time.perf_counter()
            _ = model(obs, mask=mask) if use_mask else model(obs)
            sync(device)
            times.append((time.perf_counter() - t0) * 1000)  # ms per scene

    return float(np.mean(times)), float(np.std(times))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_num',  type=int, default=0)
    p.add_argument('--obs_len',  type=int, default=10)
    p.add_argument('--pred_len', type=int, default=10)
    p.add_argument('--n_warmup',   type=int, default=10)
    p.add_argument('--n_measure',  type=int, default=50)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    T = args.obs_len

    # Load data with batch_size=1 for per-scene timing
    _, _, test_loader, _ = get_dataloaders(
        'dataset/noaa_dec2021_1min', args.obs_len, args.pred_len, batch_size=1
    )

    # Pre-fetch batches to GPU (avoids measuring data-transfer)
    needed = args.n_warmup + args.n_measure
    batches = []
    for obs, _, mask, _ in test_loader:
        batches.append((obs.to(device), mask.to(device)))
        if len(batches) >= needed:
            break
    print(f'Pre-fetched {len(batches)} batches to {device}\n')

    results = {}

    # 1. LSTM
    print('Timing LSTM...')
    lstm = load_lstm('LSTM_obs10_pred10', args.pred_len, device)
    ms, sd = measure(lstm, batches, device, use_mask=False,
                     n_warmup=args.n_warmup, n_measure=args.n_measure)
    results['LSTM'] = (ms, sd)
    del lstm

    # 2. No-CPA v4 — checkpoint on DGX, skip on RTX
    print('\nNo-CPA v4: checkpoint on DGX — skipped')

    # 3. CPA-GRN v4
    print('\nTiming CPA-GRN v4...')
    cpagrn = load_cpagrn('CPAGRN_obs10_pred10_s42', args.pred_len, device)
    ms, sd = measure(smchn, batches, device, use_mask=False,
                         n_warmup=args.n_warmup, n_measure=args.n_measure)
    
    results['CPA-GRN v4'] = (ms, sd)
    del cpagrn

   # 4. SMCHN
    print('\nTiming SMCHN...')
    try:
        from model_smchn import TrajectoryModel
        ckpt = torch.load('checkpoints/SMCHN_obs10_pred10_s42/val_best.pth',
                          map_location=device, weights_only=False)
        smchn = TrajectoryModel(
            obs_len=10, pred_len=10,
            embedding_dims=64, number_gcn_layers=1,
            dropout=0, num_heads=4
        ).to(device)
        smchn.load_state_dict(ckpt['model'])
        smchn.eval()
        print(f'  Loaded SMCHN_obs10_pred10_s42  (epoch {ckpt["epoch"]})')
        ms, sd = measure(smchn, batches, device, use_mask=True,
                         n_warmup=args.n_warmup, n_measure=args.n_measure)
        results['SMCHN'] = (ms, sd)
        del smchn
    except Exception as e:
        print(f'  Skipped: {e}')

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(f'  Inference Time per Scene  |  {T}min task  |  device={device}')
    print(f'  (batch=1, {args.n_measure} runs after {args.n_warmup} warmup)')
    print('=' * 60)
    print(f'  {"Model":<15}  {"Mean (ms)":>10}  {"±Std (ms)":>10}')
    print('-' * 60)
    for name, (ms, sd) in results.items():
        print(f'  {name:<15}  {ms:>10.2f}  {sd:>10.2f}')
    print('=' * 60)
    print('\nAdd "Inference time (ms/scene)" column to Table 4.4 in §4.1.2')


if __name__ == '__main__':
    main()