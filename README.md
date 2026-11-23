Local–Global Routing and Residual Weighting

Code and checkpoints for the paper:
Local–Global Routing and Residual Weighting for Lightweight Single-Image Super-Resolution: A Controlled Study.

We compare serial and parallel local–global refinement on an RFDN-based network. P0.5 and P1 use residual coefficients of 0.5 and 1.0. The main comparison is at ×4, with two training seeds for each model.


Installation

The experiments used Python 3.12.3, PyTorch 2.5.1+cu124 and torchvision 0.20.1+cu124. Install the dependencies in requirements.txt:

    pip install -r requirements.txt


Data

Download the datasets separately and set the data location:

    python run.py configure --data-root /path/to/data

Dataset links are listed in docs/DATA.md. Edit configs/datasets.json if your folders use a different layout.


Evaluation

Evaluate Serial on Set5 with and without TLC:

    python run.py evaluate --models serial_x4_seed0 --datasets Set5 --inference Standard TLC

Compare Serial, P0.5 and P1 on Urban100:

    python run.py evaluate --models serial_x4_seed0 p05_x4_seed0 p1_x4_seed0 --datasets Urban100 --inference Standard TLC

Replace seed0 with seed1 for the second training instance. The supported datasets are Set5, Set14, BSD100, Urban100, Manga109 and DIV2K-valid. Add --save-images to save reconstructed images, or --device cpu for CPU inference.

Checkpoint names:

Serial at ×4: serial_x4_seed0 and serial_x4_seed1.

P0.5 at ×4: p05_x4_seed0 and p05_x4_seed1.

P1 at ×4: p1_x4_seed0 and p1_x4_seed1.

Four-block RFDN reference at ×4: fair_rfdn_x4.

Serial at ×2 and ×3: serial_x2 and serial_x3.

All nine checkpoints are from training iteration 885,000. The ×2 and ×3 results use one Serial model per scale.

PSNR and SSIM are measured on floating-point Y after the RGB prediction is clipped and rounded to uint8. Y is not rounded again. The border crop equals the scale. Evaluation uses FP32 and batch size 1; TLC uses 48×48 feature grids.


Training

Choose a model and start training:

    python run.py train --model serial_x4_seed0 --stop-after 885000

Use p05_x4_seed0 or p1_x4_seed0 for the parallel models. Training settings are in configs/training/. The learning-rate schedule spans 1,000,000 iterations; --stop-after sets when training stops.

Training logs use the original rounded-Y metric. Use the evaluation command above for the float-Y results reported in the paper.


Analysis

Check the saved comparisons and statistics:

    python run.py analyze

Add --resample to rerun the bootstrap and permutation tests. To plot the saved ECDF and intervention results:

    python run.py figures

Run the global-input intervention:

    python run.py conditioning --seeds 0 1

Count MACs:

    python run.py macs --models serial_x4_seed0 p05_x4_seed0 p1_x4_seed0

Measure inference time and memory on a CUDA GPU:

    python run.py benchmark

Check checkpoint loading and model outputs on CPU:

    python run.py verify


Files

srpaper/ contains the models, data transforms and shared functions.

configs/ contains training settings, data paths and checkpoint information.

scripts/ contains the training, evaluation, analysis and plotting scripts.

tools/ contains the repository checks.

checkpoints/ contains the nine pretrained checkpoints.

results/reference/ contains the per-image results and statistics used in the paper.

New outputs are saved in experiments/, grouped by task and timestamp.

Upstream code credits and license information are in THIRD_PARTY_NOTICES.md.
