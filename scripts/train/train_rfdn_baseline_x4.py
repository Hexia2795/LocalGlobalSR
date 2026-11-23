from __future__ import annotations

import argparse
import logging
import math
import random
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

from srpaper.models.rfdn_baseline_arch import RFDN, count_parameters
from srpaper.utils import image as util


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def timestamp() -> str:
    return datetime.now().strftime("%y-%m-%d %H:%M:%S.%f")[:-3]


def log(message: str) -> None:
    print(f"{timestamp()} : {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_key(path: Path, scale: int) -> str:
    stem = path.stem.lower()
    stem = re.sub(rf"x{scale}w?$", "", stem)
    stem = re.sub(rf"_x{scale}w?$", "", stem)
    return stem


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in: {folder}")
    return paths


def pair_images(hr_dir: Path, lr_dir: Path, scale: int) -> List[Tuple[Path, Path]]:
    hr_paths = list_images(hr_dir)
    lr_paths = list_images(lr_dir)

    lr_map: Dict[str, Path] = {}
    for path in lr_paths:
        key = image_key(path, scale)
        if key in lr_map:
            raise RuntimeError(f"Duplicate LR key '{key}': {lr_map[key]} and {path}")
        lr_map[key] = path

    pairs: List[Tuple[Path, Path]] = []
    missing: List[Path] = []
    for hr_path in hr_paths:
        key = image_key(hr_path, scale)
        lr_path = lr_map.get(key)
        if lr_path is None:
            missing.append(hr_path)
        else:
            pairs.append((hr_path, lr_path))

    if missing:
        examples = "\n".join(str(path) for path in missing[:10])
        raise RuntimeError(
            f"{len(missing)} HR images have no matching LR image. Examples:\n{examples}"
        )
    if not pairs:
        raise RuntimeError("No HR/LR image pairs found.")
    return pairs


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class PairedSRDataset(Dataset):
    def __init__(
        self,
        hr_dir: Path,
        lr_dir: Path,
        scale: int,
        gt_size: int,
        cache_images: bool,
    ) -> None:
        super().__init__()
        if gt_size % scale != 0:
            raise ValueError(f"gt_size={gt_size} must be divisible by scale={scale}")

        self.scale = scale
        self.gt_size = gt_size
        self.lq_size = gt_size // scale
        self.pairs = pair_images(hr_dir, lr_dir, scale)
        self.cache_images = cache_images
        self.cache: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

        log(f"Training pairs: {len(self.pairs)}")
        if cache_images:
            self.cache = []
            progress = tqdm(self.pairs, desc="Caching DF2K images", dynamic_ncols=True)
            for hr_path, lr_path in progress:
                self.cache.append((read_rgb(hr_path), read_rgb(lr_path)))
            log(f"Cached {len(self.cache)} HR/LR pairs in RAM.")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache is not None:
            return self.cache[index]
        hr_path, lr_path = self.pairs[index]
        return read_rgb(hr_path), read_rgb(lr_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr, lr = self._load(index)

        h_lr, w_lr = lr.shape[:2]
        max_h = min(h_lr, hr.shape[0] // self.scale)
        max_w = min(w_lr, hr.shape[1] // self.scale)

        if max_h < self.lq_size or max_w < self.lq_size:
            raise RuntimeError(
                f"Image is too small for crop. LR shape={lr.shape}, "
                f"HR shape={hr.shape}, required LR crop={self.lq_size}"
            )

        top = random.randint(0, max_h - self.lq_size)
        left = random.randint(0, max_w - self.lq_size)

        lr_patch = lr[
            top : top + self.lq_size,
            left : left + self.lq_size,
            :,
        ]
        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_patch = hr[
            hr_top : hr_top + self.gt_size,
            hr_left : hr_left + self.gt_size,
            :,
        ]


        if random.random() < 0.5:
            lr_patch = np.flip(lr_patch, axis=1)
            hr_patch = np.flip(hr_patch, axis=1)
        if random.random() < 0.5:
            lr_patch = np.flip(lr_patch, axis=0)
            hr_patch = np.flip(hr_patch, axis=0)
        if random.random() < 0.5:
            lr_patch = np.transpose(lr_patch, (1, 0, 2))
            hr_patch = np.transpose(hr_patch, (1, 0, 2))

        lr_patch = np.ascontiguousarray(lr_patch.transpose(2, 0, 1))
        hr_patch = np.ascontiguousarray(hr_patch.transpose(2, 0, 1))

        lr_tensor = torch.from_numpy(lr_patch).float().div_(255.0)
        hr_tensor = torch.from_numpy(hr_patch).float().div_(255.0)
        return lr_tensor, hr_tensor


def build_validation_pairs(config: dict) -> List[Tuple[Path, Path]]:
    return pair_images(
        Path(config["datasets"]["val"]["hr"]),
        Path(config["datasets"]["val"]["lr"]),
        int(config["scale"]),
    )


def build_model(config: dict) -> RFDN:
    model_cfg = config["model"]
    return RFDN(
        in_nc=int(model_cfg.get("in_nc", 3)),
        nf=int(model_cfg.get("nf", 48)),
        num_modules=int(model_cfg.get("num_modules", 4)),
        out_nc=int(model_cfg.get("out_nc", 3)),
        upscale=int(config["scale"]),
        distillation_rate=float(model_cfg.get("distillation_rate", 0.5)),
    )


def get_raw_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=torch.float16)


def checkpoint_payload(
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    current_iter: int,
    best_psnr: float,
    config: dict,
) -> dict:
    return {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "iter": current_iter,
        "best_psnr": best_psnr,
        "config": config,
    }


def save_checkpoint(
    path: Path,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    current_iter: int,
    best_psnr: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        checkpoint_payload(
            raw_model,
            optimizer,
            scheduler,
            current_iter,
            best_psnr,
            config,
        ),
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)

    current_iter = 0
    best_psnr = float("-inf")
    if isinstance(checkpoint, dict):
        current_iter = int(checkpoint.get("iter", 0))
        best_psnr = float(checkpoint.get("best_psnr", float("-inf")))

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return current_iter, best_psnr


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    latest_path = checkpoint_dir / "latest.pth"
    if latest_path.is_file():
        return latest_path

    candidates: List[Tuple[int, Path]] = []
    for path in checkpoint_dir.glob("model_*.pth"):
        match = re.fullmatch(r"model_(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


@torch.inference_mode()
def validate(
    model: nn.Module,
    pairs: Sequence[Tuple[Path, Path]],
    scale: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    psnr_values: List[float] = []
    ssim_values: List[float] = []

    progress = tqdm(pairs, desc="Validating DIV2K", dynamic_ncols=True, leave=False)
    for hr_path, lr_path in progress:
        hr = read_rgb(hr_path)
        lr = read_rgb(lr_path)

        h_lr, w_lr = lr.shape[:2]
        target_h = h_lr * scale
        target_w = w_lr * scale
        hr = hr[:target_h, :target_w, :]

        lr_array = np.ascontiguousarray(lr.transpose(2, 0, 1))
        lr_tensor = (
            torch.from_numpy(lr_array)
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .to(device, non_blocking=True)
        )


        sr_tensor = model(lr_tensor)
        sr = util.tensor2uint(sr_tensor)
        sr = sr[:target_h, :target_w, :]

        psnr = util.calculate_psnr(
            sr,
            hr,
            border=scale,
            test_y_channel=True,
        )
        ssim = util.calculate_ssim(
            sr,
            hr,
            border=scale,
            test_y_channel=True,
        )
        psnr_values.append(float(psnr))
        ssim_values.append(float(ssim))

    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controlled original-RFDN baseline")
    parser.add_argument("-opt", "--opt", required=True, type=Path, help="YAML config")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--stop_after", type=int, default=None)
    parser.add_argument("--no_compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.opt.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = int(config.get("seed", 123))
    seed_everything(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training command.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    scale = int(config["scale"])
    train_cfg = config["train"]
    total_iter = int(train_cfg["total_iter"])
    target_iter = (
        min(int(args.stop_after), total_iter)
        if args.stop_after is not None
        else total_iter
    )
    if target_iter < 1:
        raise ValueError("Target iteration must be positive.")

    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log(f"Config: {args.opt}")
    log(f"Output directory: {output_dir}")
    log(f"Target iteration: {target_iter}; scheduler T_max remains {total_iter}")

    dataset = PairedSRDataset(
        hr_dir=Path(config["datasets"]["train"]["hr"]),
        lr_dir=Path(config["datasets"]["train"]["lr"]),
        scale=scale,
        gt_size=int(train_cfg["gt_size"]),
        cache_images=bool(train_cfg.get("cache_images", True)),
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    validation_pairs = build_validation_pairs(config)
    log(f"Validation pairs: {len(validation_pairs)}")

    raw_model = build_model(config).to(device)
    log(f"Model: original RFDB/ESA RFDN")
    log(f"Trainable parameters: {count_parameters(raw_model):,}")

    optimizer = Adam(
        raw_model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(float(value) for value in train_cfg.get("betas", [0.9, 0.999])),
        eps=float(train_cfg.get("eps", 1e-8)),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_iter,
        eta_min=float(train_cfg.get("eta_min", 1e-7)),
    )
    criterion = nn.L1Loss()

    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = find_latest_checkpoint(checkpoint_dir)

    current_iter = 0
    best_psnr = float("-inf")
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        current_iter, best_psnr = load_checkpoint(
            resume_path,
            raw_model,
            optimizer,
            scheduler,
        )
        log(
            f"Resumed from {resume_path}; iter={current_iter}, "
            f"best_psnr={best_psnr:.4f}"
        )

    if current_iter >= target_iter:
        log(f"Checkpoint is already at iter {current_iter}, target is {target_iter}.")
        return

    compile_enabled = bool(train_cfg.get("compile", True)) and not args.no_compile
    model: nn.Module = raw_model
    if compile_enabled and hasattr(torch, "compile"):
        log("Compiling training model with torch.compile(mode='default')...")
        model = torch.compile(raw_model, mode="default")
    else:
        log("torch.compile disabled.")

    amp_enabled = bool(train_cfg.get("amp", True))
    scaler = make_grad_scaler(amp_enabled)

    val_freq = int(train_cfg.get("val_freq", 5000))
    checkpoint_freq = int(train_cfg.get("checkpoint_freq", 100000))
    log_freq = int(train_cfg.get("log_freq", 100))

    model.train()
    progress = tqdm(
        total=target_iter,
        initial=current_iter,
        desc="Training RFDN baseline",
        dynamic_ncols=True,
    )

    running_loss = 0.0
    running_count = 0
    start_time = time.time()

    while current_iter < target_iter:
        for lr_batch, hr_batch in loader:
            if current_iter >= target_iter:
                break

            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                prediction = model(lr_batch)
                loss = criterion(prediction, hr_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_iter += 1
            loss_value = float(loss.detach().item())
            running_loss += loss_value
            running_count += 1

            progress.update(1)
            progress.set_postfix(
                iter=current_iter,
                loss=f"{loss_value:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if current_iter % log_freq == 0:
                average_loss = running_loss / max(running_count, 1)
                elapsed = time.time() - start_time
                speed = current_iter / max(elapsed, 1e-9)
                log(
                    f"Iter {current_iter}/{target_iter} | "
                    f"loss={average_loss:.6f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.8e} | "
                    f"average speed={speed:.3f} it/s"
                )
                running_loss = 0.0
                running_count = 0

            needs_validation = current_iter % val_freq == 0
            needs_checkpoint = (
                current_iter % checkpoint_freq == 0
                or current_iter == target_iter
            )

            if needs_validation:
                psnr, ssim = validate(
                    raw_model,
                    validation_pairs,
                    scale,
                    device,
                )
                log(
                    f"--- Iter {current_iter} | DIV2K-valid x{scale} "
                    f"Y-Channel PSNR/SSIM: {psnr:.4f} dB / {ssim:.4f} ---"
                )

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_path = checkpoint_dir / "best.pth"
                    save_checkpoint(
                        best_path,
                        raw_model,
                        optimizer,
                        scheduler,
                        current_iter,
                        best_psnr,
                        config,
                    )
                    log(
                        f"New best DIV2K-valid PSNR: {best_psnr:.4f} dB. "
                        f"Saved to {best_path}"
                    )
                model.train()


            if needs_validation:
                latest_path = checkpoint_dir / "latest.pth"
                save_checkpoint(
                    latest_path,
                    raw_model,
                    optimizer,
                    scheduler,
                    current_iter,
                    best_psnr,
                    config,
                )
                log(f"Latest checkpoint updated: {latest_path}")

            if needs_checkpoint:
                checkpoint_path = checkpoint_dir / f"model_{current_iter}.pth"
                save_checkpoint(
                    checkpoint_path,
                    raw_model,
                    optimizer,
                    scheduler,
                    current_iter,
                    best_psnr,
                    config,
                )
                log(f"Checkpoint saved: {checkpoint_path}")

    progress.close()

    last_path = checkpoint_dir / "latest.pth"
    save_checkpoint(
        last_path,
        raw_model,
        optimizer,
        scheduler,
        current_iter,
        best_psnr,
        config,
    )
    elapsed = time.time() - start_time
    log(
        f"Training finished at iter {current_iter}. "
        f"Elapsed: {elapsed / 3600:.2f} hours. Final checkpoint: {last_path}"
    )


if __name__ == "__main__":
    main()
