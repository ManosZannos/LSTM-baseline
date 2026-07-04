#!/bin/bash
# run_extra_seeds.sh
# Τρέχει LSTM + CPA-GRN για το 10min task με seeds 123 και 456
# Σκοπός: mean ± std για το κύριο αποτέλεσμα (Πίνακας 4.4)
#
# Εκτέλεση:
#   bash run_extra_seeds.sh 0      # RTX, GPU 0 (LSTM, γρήγορο)
#   bash run_extra_seeds.sh X      # DGX, GPU X (CPA-GRN, --batch_size 16)
#
# Μετά: python compute_seed_stats.py

GPU=${1:-0}

echo "=== LSTM seed 123 ==="
python train_lstm.py \
    --obs_len 10 --pred_len 10 \
    --hidden_size 64 --epochs 200 --lr 0.001 \
    --tag LSTM_obs10_pred10_s123 --seed 123 --gpu_num $GPU

echo "=== LSTM seed 456 ==="
python train_lstm.py \
    --obs_len 10 --pred_len 10 \
    --hidden_size 64 --epochs 200 --lr 0.001 \
    --tag LSTM_obs10_pred10_s456 --seed 456 --gpu_num $GPU

echo "=== CPA-GRN seed 123 (batch=16, DGX) ==="
python train_cpagrn.py \
    --obs_len 10 --pred_len 10 \
    --d_model 64 --epochs 200 --lr 0.001 --batch_size 16 \
    --tag CPAGRN_obs10_pred10_s123 --seed 123 --gpu_num $GPU

echo "=== CPA-GRN seed 456 (batch=16, DGX) ==="
python train_cpagrn.py \
    --obs_len 10 --pred_len 10 \
    --d_model 64 --epochs 200 --lr 0.001 --batch_size 16 \
    --tag CPAGRN_obs10_pred10_s456 --seed 456 --gpu_num $GPU

echo ""
echo "Training done. Now evaluate:"
echo "  python evaluate_lstm.py   --tag LSTM_obs10_pred10_s123   --obs_len 10 --pred_len 10 --split test --gpu_num $GPU"
echo "  python evaluate_lstm.py   --tag LSTM_obs10_pred10_s456   --obs_len 10 --pred_len 10 --split test --gpu_num $GPU"
echo "  python evaluate_cpagrn.py --tag CPAGRN_obs10_pred10_s123 --obs_len 10 --pred_len 10 --split test --gpu_num $GPU"
echo "  python evaluate_cpagrn.py --tag CPAGRN_obs10_pred10_s456 --obs_len 10 --pred_len 10 --split test --gpu_num $GPU"
echo ""
echo "Then run: python compute_seed_stats.py"
