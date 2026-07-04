"""
measure_inference.py — Inference time per scene για τα μοντέλα (Πίνακας 4.4).

Διορθώσεις έναντι της πρώτης έκδοσης:
  1. Τα checkpoint tags χτίζονται δυναμικά από obs_len/pred_len/seed
     (πριν ήταν hardcoded σε "obs10_pred10" ό,τι κι αν έβαζες στο CLI).
  2. Οι σκηνές δειγματοληπτούνται ΤΥΧΑΙΑ (fixed seed) από ΟΛΟ το test set,
     όχι τα πρώτα N χρονολογικά batches — αποφεύγεται bias λόγω
     μεταβλητού πλήθους πλοίων (N) ανά σκηνή, που επηρεάζει άμεσα το
     κόστος του CPA-GRN (O(N^2) στο NeighborAggregation).
  3. Αναφέρεται το μέσο/std πλήθος πλοίων (N) των δειγματισμένων σκηνών,
     ώστε το ms/scene να είναι ερμηνεύσιμο.
  4. Το SMCHN διαβάζει saved['args'] από το checkpoint αν υπάρχουν
     (όπως ήδη κάνουν CPA-GRN/LSTM) αντί για hardcoded υπερπαραμέτρους,
     με σαφή προειδοποίηση αν πέσει σε fallback.
  5. Mask περνιέται με συνέπεια σε όλα τα μοντέλα.
  6. Φορτώνεται απευθείας μόνο το test split (όχι ολόκληρο το train/val)
     ώστε το script να τρέχει γρήγορα.

Usage:
    python measure_inference.py --gpu_num 0
    python measure_inference.py --gpu_num 0 --obs_len 5 --pred_len 5
    python measure_inference.py --gpu_num 0 --obs_len 10 --pred_len 10 --seed 42
"""
import os, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import AISDataset, collate_fn


# ── Data loading (test split only) ─────────────────────────────────────────────

def load_test_only(data_dir, obs_len, pred_len):
    """Φτιάχνει DataLoader μόνο για το test split (χωρίς train/val)."""
    test_dir = os.path.join(data_dir, 'test')
    eval_stride = obs_len + pred_len  # non-overlapping, όπως στο get_dataloaders
    test_ds = AISDataset(test_dir, obs_len, pred_len, stride=eval_stride)
    return test_ds


def sample_scenes(test_ds, n_needed, seed, batch_size=1, num_workers=0):
    """Τυχαία δειγματοληψία n_needed σκηνών από ΟΛΟ το test set."""
    rng = np.random.default_rng(seed)
    n_total = len(test_ds)
    if n_total < n_needed:
        raise ValueError(
            f'Το test set έχει μόνο {n_total} σκηνές, χρειάζονται {n_needed}. '
            f'Μείωσε --n_warmup/--n_measure.'
        )
    idx = rng.choice(n_total, size=n_needed, replace=False)
    subset = Subset(test_ds, idx.tolist())
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    return loader


# ── Model loaders ───────────────────────────────────────────────────────────────

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


def load_smchn(tag, obs_len, pred_len, device):
    from model_smchn import TrajectoryModel
    ckpt = torch.load(f'checkpoints/{tag}/val_best.pth',
                      map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    if not saved:
        print('  ⚠ Το checkpoint δεν έχει saved["args"] — χρησιμοποιούνται '
              'προεπιλογές (embedding_dims=64, number_gcn_layers=1, '
              'num_heads=4). Επιβεβαίωσε ότι ταιριάζουν με το training config.')
    m = TrajectoryModel(
        obs_len=obs_len, pred_len=pred_len,
        embedding_dims=saved.get('embedding_dims', 64),
        number_gcn_layers=saved.get('number_gcn_layers', 1),
        dropout=0,
        num_heads=saved.get('num_heads', 4),
    ).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()
    print(f'  Loaded {tag}  (epoch {ckpt["epoch"]})')
    return m


def obs_to_smchn_graph(obs, mask):
    """
    ΠΡΟΣΩΡΙΝΟ PLACEHOLDER — ΔΕΝ είναι έτοιμο για χρήση.

    model_smchn.TrajectoryModel.forward(graph, identity) περιμένει:
      graph:    [1, obs_len, N, 5]  = [pos_enc, LON_rel, LAT_rel, SOG_rel, Heading_rel]
      identity: (spatial_identity [T,N,N], temporal_identity [N,T,T])

    Το AISDataset δίνει obs [B, N, obs_len, 4] = [LON, LAT, SOG, Heading] (z-score,
    ΟΧΙ relative), χωρίς pos_enc και χωρίς τα identity tensors.

    Η ακριβής μετατροπή (τύπος pos_enc, τρόπος υπολογισμού *_rel, κατασκευή
    identity matrices) καθορίζεται στο evaluate_smchn.py / train_smchn.py, τα
    οποία δεν είναι διαθέσιμα σε αυτό το project. ΜΗΝ μαντέψεις εδώ — καλύτερα
    να σκάσει καθαρά παρά να μετρήσει λάθος tensors σιωπηλά.

    Δώσε το evaluate_smchn.py για να συμπληρωθεί σωστά αυτή η συνάρτηση.
    """
    raise NotImplementedError(
        'obs_to_smchn_graph() δεν έχει υλοποιηθεί ακόμα — χρειάζεται το '
        'evaluate_smchn.py / train_smchn.py για τη σωστή μετατροπή '
        '(obs,mask) -> (graph,identity). Μέχρι τότε τρέξε με --skip_smchn.'
    )


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def default_call(model, obs, mask):
    return model(obs, mask=mask)


def measure(model, batches, device, n_warmup=10, n_measure=50, call_fn=default_call):
    """Returns mean ± std of wall-clock ms per sample (scene), και μέσο N.

    call_fn(model, obs, mask) -> προαιρετικό adapter για μοντέλα με άλλο
    forward signature (π.χ. SMCHN: model(graph, identity)).
    """
    with torch.no_grad():
        for obs, mask in batches[:n_warmup]:
            _ = call_fn(model, obs, mask)
        sync(device)

        times = []
        ns = []
        for obs, mask in batches[n_warmup: n_warmup + n_measure]:
            sync(device)
            t0 = time.perf_counter()
            _ = call_fn(model, obs, mask)
            sync(device)
            times.append((time.perf_counter() - t0) * 1000)  # ms per scene
            ns.append(obs.shape[1])  # N πλοία σε αυτή τη σκηνή

    return (float(np.mean(times)), float(np.std(times)),
            float(np.mean(ns)), float(np.std(ns)))


def smchn_call(model, obs, mask):
    graph, identity = obs_to_smchn_graph(obs, mask)
    return model(graph, identity)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_num',   type=int, default=0)
    p.add_argument('--obs_len',   type=int, default=10)
    p.add_argument('--pred_len',  type=int, default=10)
    p.add_argument('--seed',      type=int, default=42,
                   help='seed suffix του CPA-GRN checkpoint tag (π.χ. s42)')
    p.add_argument('--sample_seed', type=int, default=0,
                   help='seed για την τυχαία δειγματοληψία σκηνών (αναπαραγωγιμότητα)')
    p.add_argument('--data_dir',  type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--n_warmup',  type=int, default=10)
    p.add_argument('--n_measure', type=int, default=50)
    p.add_argument('--skip_smchn', action='store_true',
                   help='αγνόησε το SMCHN (π.χ. αν το checkpoint δεν είναι σε αυτόν τον server)')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Task: obs={args.obs_len} pred={args.pred_len}\n')

    # Δυναμικά tags — ΟΧΙ πλέον hardcoded σε obs10_pred10
    lstm_tag   = f'LSTM_obs{args.obs_len}_pred{args.pred_len}'
    cpagrn_tag = f'CPAGRN_obs{args.obs_len}_pred{args.pred_len}_s{args.seed}'
    smchn_tag  = f'SMCHN_obs{args.obs_len}_pred{args.pred_len}_s{args.seed}'

    needed = args.n_warmup + args.n_measure

    print('Building test dataset (μόνο test split)...')
    test_ds = load_test_only(args.data_dir, args.obs_len, args.pred_len)
    print(f'  Test set: {len(test_ds):,} σκηνές\n')

    print(f'Τυχαία δειγματοληψία {needed} σκηνών (sample_seed={args.sample_seed})...')
    loader = sample_scenes(test_ds, needed, seed=args.sample_seed)

    batches = []
    for obs, _, mask, _ in loader:
        batches.append((obs.to(device), mask.to(device)))
    print(f'  Pre-fetched {len(batches)} σκηνές στη {device}\n')

    results = {}

    print('Timing LSTM...')
    lstm = load_lstm(lstm_tag, args.pred_len, device)
    ms, sd, n_mean, n_sd = measure(lstm, batches, device,
                                    n_warmup=args.n_warmup, n_measure=args.n_measure)
    results['LSTM'] = (ms, sd, n_mean, n_sd)
    del lstm

    print('\nTiming CPA-GRN v4...')
    cpagrn = load_cpagrn(cpagrn_tag, args.pred_len, device)
    ms, sd, n_mean, n_sd = measure(cpagrn, batches, device,
                                    n_warmup=args.n_warmup, n_measure=args.n_measure)
    results['CPA-GRN v4'] = (ms, sd, n_mean, n_sd)
    del cpagrn

    if not args.skip_smchn:
        print('\nTiming SMCHN...')
        try:
            smchn = load_smchn(smchn_tag, args.obs_len, args.pred_len, device)
            ms, sd, n_mean, n_sd = measure(smchn, batches, device,
                                            n_warmup=args.n_warmup, n_measure=args.n_measure,
                                            call_fn=smchn_call)
            results['SMCHN'] = (ms, sd, n_mean, n_sd)
            del smchn
        except FileNotFoundError as e:
            print(f'  Skipped (checkpoint not found on this server): {e}')
        except Exception as e:
            print(f'  Skipped: {e}')
    else:
        print('\nSMCHN: --skip_smchn set, παραλείπεται')

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print(f'  Inference Time per Scene  |  {args.pred_len}min task  |  device={device}')
    print(f'  (batch=1, {args.n_measure} runs after {args.n_warmup} warmup, '
          f'random sample seed={args.sample_seed})')
    print('=' * 72)
    print(f'  {"Model":<15}  {"Mean (ms)":>10}  {"±Std (ms)":>10}  {"Mean N":>8}  {"±Std N":>8}')
    print('-' * 72)
    for name, (ms, sd, n_mean, n_sd) in results.items():
        print(f'  {name:<15}  {ms:>10.3f}  {sd:>10.3f}  {n_mean:>8.1f}  {n_sd:>8.1f}')
    print('=' * 72)
    print('\nΣημείωση: όλα τα μοντέλα μετρήθηκαν πάνω ΣΤΙΣ ΙΔΙΕΣ τυχαία')
    print('δειγματισμένες σκηνές (ίδιο sample_seed) — η στήλη Mean N πρέπει')
    print('να είναι ίδια σε όλες τις γραμμές, ως έλεγχος συνέπειας.')
    print('\nΠρόσθεσε τη στήλη "Inference time (ms/scene)" στον Πίνακα 4.4 (§4.1.2).')


if __name__ == '__main__':
    main()