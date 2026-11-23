Project layout

run.py is the main entry point for training, evaluation and analysis.

srpaper/models/
Model definitions and the global-input intervention.

srpaper/data/
Data augmentation and paired cropping.

srpaper/utils/
Image conversion, metrics, logging and MAC counting.

configs/
Training settings, dataset paths and checkpoint information.

scripts/train/
Training scripts.

scripts/eval/
PSNR and SSIM evaluation.

scripts/analysis/
Statistical analysis, interventions and MAC measurement.

scripts/benchmark/
Inference time and memory measurement.

scripts/figures/
Plots made from the saved results.

tools/
Checkpoint and model-output checks.

checkpoints/
The nine pretrained checkpoints.

results/reference/
Per-image results and statistics used in the paper.

experiments/
New outputs, saved in a separate folder for each task and timestamp. CPU check reports are saved under experiments/tests/.
