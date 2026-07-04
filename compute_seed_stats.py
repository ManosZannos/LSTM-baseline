"""
compute_seed_stats.py — Συγκεντρώνει ADE/FDE από τα 3 seeds και εκτυπώνει
mean ± std για LSTM και CPA-GRN στο 10min task.

Τρέξε ΜΕΤΑ από evaluate_*.py για seeds 42, 123, 456.

Usage:
    python compute_seed_stats.py
"""
import numpy as np

# ── Συμπλήρωσε εδώ τα αποτελέσματα από τα evaluate scripts ──────────────────
#    ADE (avg) σε μοίρες, ακριβώς όπως εκτυπώνεται

results = {
    'LSTM': {
        42:  {'ade': 0.001486, 'fde': 0.002295},
        123: {'ade': None,     'fde': None},      # ← συμπλήρωσε
        456: {'ade': None,     'fde': None},      # ← συμπλήρωσε
    },
    'CPA-GRN v4': {
        42:  {'ade': 0.001137, 'fde': 0.001628},
        123: {'ade': None,     'fde': None},      # ← συμπλήρωσε
        456: {'ade': None,     'fde': None},      # ← συμπλήρωσε
    },
}

# ── Υπολογισμός ───────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('  Seed Robustness — 10min task (seeds 42, 123, 456)')
print('=' * 65)
print(f'  {"Model":<15} {"ADE mean±std (°)":>20} {"FDE mean±std (°)":>20}')
print('-' * 65)

for model, seeds in results.items():
    ade_vals = [v['ade'] for v in seeds.values() if v['ade'] is not None]
    fde_vals = [v['fde'] for v in seeds.values() if v['fde'] is not None]

    if len(ade_vals) == 3:
        ade_m, ade_s = np.mean(ade_vals), np.std(ade_vals)
        fde_m, fde_s = np.mean(fde_vals), np.std(fde_vals)
        print(f'  {model:<15} {ade_m:.6f} ± {ade_s:.6f}   {fde_m:.6f} ± {fde_s:.6f}')
    else:
        print(f'  {model:<15} [πλήρωσε τιμές — έχεις {len(ade_vals)}/3 seeds]')

print('=' * 65)
print('\nΠρόσθεσε ως νέα γραμμή στον Πίνακα 4.4 ή ως footnote:')
print('"All results are mean±std over seeds {42, 123, 456}."')
