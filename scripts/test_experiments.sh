#!/bin/bash
set -e

# Experiment 1: Shallow LRU on MNIST98 (matches mnist98_lru_comparison sweep)
# 1 layer, hidden=32, LRU, cumulative_mean pooling, all 4 training modes
for mode in normal spatial forward forward_forward; do
  python src/train.py data=mnist98_test model.cell=lru model.n_layers=1 \
    model.hidden_dim=32 model.lru_r_min=0.9 model.pooling=cumulative_mean \
    model.training_mode=$mode training.wandb_log=false
done

# Experiment 2: Deep LRU on MNIST98 (matches mnist98_deep_lru_comparison sweep)
# 4 layers, hidden=32, LRU, all 4 training modes
for mode in normal spatial forward forward_forward; do
  python src/train.py data=mnist98_test model.cell=lru model.n_layers=4 \
    model.hidden_dim=32 model.lru_r_min=0.99 model.pooling=cumulative_mean \
    model.training_mode=$mode training.wandb_log=false
done

# Experiment 3: Simulation passes ablation (matches mnist98_deep_lru_simulation_passes sweep)
# 4 layers, hidden=32, forward_forward only, varying simulation_passes
for passes in 1 3 5; do
  python src/train.py data=mnist98_test model.cell=lru model.n_layers=4 \
    model.hidden_dim=32 model.lru_r_min=0.99 model.pooling=cumulative_mean \
    model.training_mode=forward_forward model.forward_simulation_passes=$passes \
    training.wandb_log=false
done

# Experiment 4: Deep LRU on full MNIST with frozen recurrence (matches mnist_deep_lru_comparison sweep)
# 2 layers, hidden=128, LRU, r_min=0.999, freeze_recurrence=true
for mode in normal spatial forward forward_forward; do
  python src/train.py data=mnist_test model.cell=lru model.n_layers=2 \
    model.hidden_dim=128 model.lru_r_min=0.999 model.pooling=cumulative_mean \
    model.freeze_recurrence=true model.training_mode=$mode training.wandb_log=false
done

# Bonus: Copy task baseline
python src/train.py data=copy_test training.wandb_log=false
