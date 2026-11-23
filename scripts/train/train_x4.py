from __future__ import annotations

import os


os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_OK', 'TRUE')

import argparse
import gc
import glob
import logging
import random
import re
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from srpaper.models.RFDN import RFDN
from srpaper.data.transforms import augment, paired_random_crop
from srpaper.utils import image as util
from srpaper.utils.logger import logger_info


cv2.setNumThreads(0)
torch.set_float32_matmul_precision('high')

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
CHECKPOINT_DIR = 'experiments/checkpoints_x4'
LOG_PATH = 'experiments/train_x4.log'


def set_random_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_rgb(path: str) -> np.ndarray:

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'无法读取图像: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def find_lq_path(lq_dir: str, basename: str, extension: str, scale: int) -> Optional[str]:

    candidates = (
        f'{basename}x{scale}w{extension}',
        f'{basename}x{scale}{extension}',
        f'{basename}{extension}',
    )
    for name in candidates:
        path = os.path.join(lq_dir, name)
        if os.path.isfile(path):
            return path
    return None


def build_image_pairs(gt_dir: str, lq_dir: str, scale: int) -> Tuple[list, list]:

    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f'GT 文件夹不存在: {gt_dir}')
    if not os.path.isdir(lq_dir):
        raise FileNotFoundError(f'LQ 文件夹不存在: {lq_dir}')

    gt_paths = []
    lq_paths = []

    for gt_name in sorted(os.listdir(gt_dir)):
        if not gt_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        basename, extension = os.path.splitext(gt_name)
        lq_path = find_lq_path(lq_dir, basename, extension, scale)
        if lq_path is None:
            continue

        gt_paths.append(os.path.join(gt_dir, gt_name))
        lq_paths.append(lq_path)

    if not gt_paths:
        raise RuntimeError(
            '没有找到任何 GT/LQ 配对。\n'
            f'GT: {gt_dir}\n'
            f'LQ: {lq_dir}\n'
            f'已尝试文件名格式: <basename>x{scale}w、<basename>x{scale}、<basename>'
        )

    return gt_paths, lq_paths


class SRDataset(Dataset):


    def __init__(self, gt_path: str, lq_path: str, gt_size: int, scale: int):
        self.gt_size = int(gt_size)
        self.scale = int(scale)
        self.to_tensor = ToTensor()
        self.gt_paths, self.lq_paths = build_image_pairs(gt_path, lq_path, self.scale)
        self.data_cache = []

        print(f'\n[TrainDataset] 正在解码并缓存 {len(self.gt_paths)} 对训练图像...')
        for gt_file, lq_file in tqdm(
            zip(self.gt_paths, self.lq_paths),
            total=len(self.gt_paths),
            desc='Loading training images to RAM',
        ):
            img_gt = read_rgb(gt_file)
            img_lq = read_rgb(lq_file)

            h_lq, w_lq = img_lq.shape[:2]
            required_h = h_lq * self.scale
            required_w = w_lq * self.scale
            if img_gt.shape[0] < required_h or img_gt.shape[1] < required_w:
                raise RuntimeError(
                    'GT 尺寸小于对应 LQ 的放大尺寸:\n'
                    f'GT={gt_file}, shape={img_gt.shape}\n'
                    f'LQ={lq_file}, shape={img_lq.shape}, scale={self.scale}'
                )


            img_gt = img_gt[:required_h, :required_w, :]
            self.data_cache.append((img_gt, img_lq))

        gc.collect()
        print('[TrainDataset] 训练图像缓存完成。')

    def __len__(self) -> int:
        return len(self.data_cache)

    def __getitem__(self, idx: int):
        img_gt, img_lq = self.data_cache[idx]

        img_gt_crop, img_lq_crop = paired_random_crop(
            img_gt,
            img_lq,
            self.gt_size,
            self.scale,
        )


        img_gt_crop = img_gt_crop.copy()
        img_lq_crop = img_lq_crop.copy()

        img_gt_aug, img_lq_aug = augment(
            [img_gt_crop, img_lq_crop],
            hflip=True,
            rotation=True,
        )

        return self.to_tensor(img_lq_aug), self.to_tensor(img_gt_aug)


class ValDataset(Dataset):


    def __init__(self, gt_path: str, lq_path: str, scale: int = 4):
        self.scale = int(scale)
        self.to_tensor = ToTensor()
        self.gt_paths, self.lq_paths = build_image_pairs(gt_path, lq_path, self.scale)

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, idx: int):
        img_gt = read_rgb(self.gt_paths[idx])
        img_lq = read_rgb(self.lq_paths[idx])

        h_lq, w_lq = img_lq.shape[:2]
        required_h = h_lq * self.scale
        required_w = w_lq * self.scale
        if img_gt.shape[0] < required_h or img_gt.shape[1] < required_w:
            raise RuntimeError(
                '验证图像尺寸不匹配:\n'
                f'GT={self.gt_paths[idx]}, shape={img_gt.shape}\n'
                f'LQ={self.lq_paths[idx]}, shape={img_lq.shape}, scale={self.scale}'
            )

        img_gt = img_gt[:required_h, :required_w, :]
        return self.to_tensor(img_lq.copy()), self.to_tensor(img_gt.copy())


def checkpoint_iteration(path: str) -> int:

    match = re.search(r'model_(\d+)\.pth$', os.path.basename(path))
    return int(match.group(1)) if match else -1


def save_checkpoint(
    path: str,
    current_iter: int,
    best_psnr: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    val_psnr: Optional[float] = None,
    val_ssim: Optional[float] = None,
) -> None:

    state: Dict = {
        'iter': int(current_iter),
        'best_psnr': float(best_psnr),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    if val_psnr is not None:
        state['val_psnr'] = float(val_psnr)
    if val_ssim is not None:
        state['val_ssim'] = float(val_ssim)

    torch.save(state, path)


def load_checkpoint(
    path: str,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> Tuple[int, float]:

    if not os.path.isfile(path):
        raise FileNotFoundError(f'断点文件不存在: {path}')

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict']


    clean_state_dict = {
        key.removeprefix('_orig_mod.'): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(clean_state_dict, strict=True)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_iter = int(checkpoint['iter'])
    best_psnr = float(checkpoint.get('best_psnr', -1.0))
    return start_iter, best_psnr


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    scale: int,
) -> Tuple[float, float]:

    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0

    with torch.inference_mode():
        for val_lq, val_gt in tqdm(val_loader, desc='Validating', leave=False):
            val_lq = val_lq.to(device, non_blocking=True)


            out_val = model(val_lq)

            out_val_uint = util.tensor2uint(out_val)
            gt_val_uint = util.tensor2uint(val_gt)

            psnr_sum += util.calculate_psnr(
                out_val_uint,
                gt_val_uint,
                border=scale,
                test_y_channel=True,
            )
            ssim_sum += util.calculate_ssim(
                out_val_uint,
                gt_val_uint,
                border=scale,
                test_y_channel=True,
            )

    num_images = len(val_loader.dataset)
    if num_images == 0:
        raise RuntimeError('验证集为空，无法计算 PSNR/SSIM。')

    return psnr_sum / num_images, ssim_sum / num_images


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    device: torch.device,
) -> torch.optim.Optimizer:

    if device.type == 'cuda':
        try:
            return torch.optim.Adam(model.parameters(), lr=learning_rate, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='train_RFDN_x4.yml')
    parser.add_argument('--resume', type=str, default=None, help='手动指定断点路径')
    parser.add_argument('--auto_resume', action='store_true', help='自动恢复迭代数最大的 model_*.pth')
    parser.add_argument('--no_compile', action='store_true', help='禁用 torch.compile')
    parser.add_argument(
        '--stop_after',
        type=int,
        default=None,
        help='仅用于短跑测试：在指定绝对迭代数停止，例如 --stop_after 5000',
    )
    parser.add_argument(
        '--compile_mode',
        type=str,
        default=None,
        help='覆盖 YAML 中的 compile_mode，例如 default 或 reduce-overhead',
    )
    args = parser.parse_args()

    with open(args.opt, mode='r', encoding='utf-8') as file:
        opt = yaml.safe_load(file)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger_info('Train_x4', log_path=LOG_PATH)
    logger = logging.getLogger('Train_x4')

    seed = int(opt.get('manual_seed', opt.get('train', {}).get('manual_seed', 0)))
    set_random_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = device.type == 'cuda'
    pin_memory = device.type == 'cuda'


    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_opt = opt['train']
    compile_enabled = bool(train_opt.get('compile', True)) and not args.no_compile
    compile_mode = args.compile_mode or str(train_opt.get('compile_mode', 'default'))

    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(device)}')
    logger.info(f'PyTorch: {torch.__version__}')
    logger.info(f'Random seed: {seed}')
    logger.info(f'Training AMP enabled: {amp_enabled}')
    logger.info('Validation precision: FP32')
    logger.info('cuDNN benchmark: enabled; deterministic CUDA algorithms: disabled')


    raw_model = RFDN(
        in_nc=opt['network_g']['num_in_ch'],
        nf=opt['network_g']['num_feat'],
        num_modules=opt['network_g']['num_block'],
        out_nc=opt['network_g']['num_out_ch'],
        upscale=opt['network_g']['upscale'],
    ).to(device)

    num_params = sum(
        parameter.numel()
        for parameter in raw_model.parameters()
        if parameter.requires_grad
    )
    logger.info(f'Model Parameters: {num_params / 1000:.2f} K')

    total_iter = int(train_opt['total_iter'])
    val_freq = int(train_opt['val_freq'])
    save_freq = int(train_opt.get('save_freq', val_freq))
    log_freq = int(train_opt.get('log_freq', 1000))

    for name, value in {
        'total_iter': total_iter,
        'val_freq': val_freq,
        'save_freq': save_freq,
        'log_freq': log_freq,
    }.items():
        if value <= 0:
            raise ValueError(f'{name} 必须大于 0，当前值: {value}')

    optimizer = create_optimizer(
        raw_model,
        learning_rate=float(train_opt['lr']),
        device=device,
    )
    criterion = nn.L1Loss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_iter,
        eta_min=float(train_opt.get('eta_min', 1e-7)),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    start_iter = 0
    best_psnr = -1.0

    if args.auto_resume and not args.resume:
        checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, 'model_*.pth'))
        if checkpoints:
            args.resume = max(checkpoints, key=checkpoint_iteration)
            logger.info(f'Auto-resume checkpoint: {args.resume}')
        else:
            logger.info('Auto-resume enabled, but no model_*.pth checkpoint was found.')


    if args.resume:
        start_iter, best_psnr = load_checkpoint(
            args.resume,
            device,
            raw_model,
            optimizer,
            scheduler,
            scaler,
        )
        logger.info(
            f'Successfully resumed from iter {start_iter}; '
            f'best DIV2K-valid PSNR={best_psnr:.4f} dB.'
        )

    if start_iter > total_iter:
        raise ValueError(
            f'断点迭代数 {start_iter} 大于 total_iter {total_iter}，请检查配置或断点。'
        )

    run_total_iter = total_iter
    if args.stop_after is not None:
        if args.stop_after <= 0:
            raise ValueError('--stop_after 必须大于 0。')
        run_total_iter = min(total_iter, args.stop_after)
        if start_iter >= run_total_iter:
            raise ValueError(
                f'起始 iter={start_iter} 已达到本次停止 iter={run_total_iter}。'
            )
        logger.info(
            f'Short-run mode: this run will stop at iter {run_total_iter}; '
            f'configured total_iter remains {total_iter}.'
        )

    train_model: nn.Module = raw_model
    if compile_enabled and device.type == 'cuda':
        if hasattr(torch, 'compile'):
            logger.info(f'Enabling torch.compile(mode={compile_mode}).')
            logger.info('首次训练迭代会进行图编译，可能暂时停留在 0 iter，这是正常现象。')
            train_model = torch.compile(
                raw_model,
                mode=compile_mode,
                fullgraph=False,
            )
        else:
            logger.warning('当前 PyTorch 不支持 torch.compile，自动使用 eager 模式。')
    else:
        logger.info('torch.compile disabled.')

    train_workers = int(opt['datasets']['train'].get('num_worker_per_gpu', 4))
    val_workers = int(opt['datasets']['val'].get('num_worker_per_gpu', 2))
    batch_size = int(opt['datasets']['train']['batch_size_per_gpu'])

    train_dataset = SRDataset(
        opt['datasets']['train']['dataroot_gt'],
        opt['datasets']['train']['dataroot_lq'],
        opt['datasets']['train']['gt_size'],
        opt['scale'],
    )
    val_dataset = ValDataset(
        opt['datasets']['val']['dataroot_gt'],
        opt['datasets']['val']['dataroot_lq'],
        opt['scale'],
    )

    if len(train_dataset) < batch_size:
        raise ValueError(
            f'训练图片数量 {len(train_dataset)} 小于 batch_size {batch_size}。'
        )

    logger.info(f'Train pairs: {len(train_dataset)}')
    logger.info(f'Validation pairs: {len(val_dataset)}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Train workers: {train_workers}; validation workers: {val_workers}')

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader_kwargs = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'shuffle': True,
        'drop_last': True,
        'num_workers': train_workers,
        'pin_memory': pin_memory,
        'worker_init_fn': seed_worker,
        'generator': loader_generator,
    }
    if train_workers > 0:
        train_loader_kwargs.update({
            'persistent_workers': True,
            'prefetch_factor': 2,
        })

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
        persistent_workers=val_workers > 0,
    )

    current_iter = start_iter
    last_val_psnr: Optional[float] = None
    last_val_ssim: Optional[float] = None
    progress = tqdm(total=run_total_iter, initial=start_iter, desc='Training')

    speed_start_time = time.perf_counter()
    speed_start_iter = current_iter

    raw_model.train()
    train_model.train()

    while current_iter < run_total_iter:
        for lq, gt in train_loader:
            if current_iter >= run_total_iter:
                break

            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = train_model(lq)
                loss = criterion(output, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_iter += 1
            progress.update(1)
            progress.set_postfix(loss=f'{loss.item():.4f}', iter=current_iter)

            if current_iter % log_freq == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = max(time.perf_counter() - speed_start_time, 1e-9)
                interval_iters = current_iter - speed_start_iter
                speed = interval_iters / elapsed
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f'Iter: {current_iter}/{total_iter} | '
                    f'Loss: {loss.item():.6f} | LR: {current_lr:.8e} | '
                    f'Speed: {speed:.3f} it/s'
                )
                speed_start_time = time.perf_counter()
                speed_start_iter = current_iter

            if current_iter % val_freq == 0:

                raw_model.eval()
                avg_psnr, avg_ssim = validate(
                    raw_model,
                    val_loader,
                    device,
                    int(opt['scale']),
                )
                last_val_psnr = avg_psnr
                last_val_ssim = avg_ssim

                logger.info(
                    f'--- Iter {current_iter} | DIV2K-valid x4 Y-Channel '
                    f'PSNR/SSIM: {avg_psnr:.4f} dB / {avg_ssim:.4f} ---'
                )

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_path = os.path.join(CHECKPOINT_DIR, 'best.pth')
                    save_checkpoint(
                        best_path,
                        current_iter,
                        best_psnr,
                        raw_model,
                        optimizer,
                        scheduler,
                        scaler,
                        avg_psnr,
                        avg_ssim,
                    )
                    logger.info(
                        f'New best DIV2K-valid PSNR: {best_psnr:.4f} dB. '
                        f'Saved to {best_path}'
                    )

                raw_model.train()
                train_model.train()
                speed_start_time = time.perf_counter()
                speed_start_iter = current_iter

            if current_iter % save_freq == 0:
                iteration_path = os.path.join(
                    CHECKPOINT_DIR,
                    f'model_{current_iter}.pth',
                )
                save_checkpoint(
                    iteration_path,
                    current_iter,
                    best_psnr,
                    raw_model,
                    optimizer,
                    scheduler,
                    scaler,
                    last_val_psnr,
                    last_val_ssim,
                )

                latest_path = os.path.join(CHECKPOINT_DIR, 'latest.pth')
                save_checkpoint(
                    latest_path,
                    current_iter,
                    best_psnr,
                    raw_model,
                    optimizer,
                    scheduler,
                    scaler,
                    last_val_psnr,
                    last_val_ssim,
                )
                logger.info(f'Checkpoint saved: {iteration_path}')

    final_name = 'speed_test_last.pth' if args.stop_after is not None else 'last.pth'
    final_path = os.path.join(CHECKPOINT_DIR, final_name)
    save_checkpoint(
        final_path,
        current_iter,
        best_psnr,
        raw_model,
        optimizer,
        scheduler,
        scaler,
        last_val_psnr,
        last_val_ssim,
    )

    progress.close()
    logger.info(f'Training finished. Final checkpoint: {final_path}')


if __name__ == '__main__':
    main()
