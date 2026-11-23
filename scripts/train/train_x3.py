import os


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_OK'] = 'TRUE'

import yaml
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import cv2


cv2.setNumThreads(0)

import numpy as np
from tqdm import tqdm
import logging
import gc

from srpaper.models.RFDN import RFDN
from srpaper.data.transforms import paired_random_crop, augment
from srpaper.utils import image as util
from srpaper.utils.logger import logger_info


torch.set_float32_matmul_precision('high')

class SRDataset(Dataset):
    def __init__(self, gt_path, lq_path, gt_size, scale):
        self.gt_size = gt_size
        self.scale = scale
        self.to_tensor = ToTensor()
        self.gt_paths = []
        self.lq_paths = []
        self.data_cache = []

        gt_files = sorted(os.listdir(gt_path))
        for gt_name in gt_files:
            if not gt_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            basename, ext = os.path.splitext(gt_name)
            lq_name_x3 = f"{basename}x{scale}{ext}"
            lq_name_same = f"{basename}{ext}"
            path_x3 = os.path.join(lq_path, lq_name_x3)
            path_same = os.path.join(lq_path, lq_name_same)

            if os.path.exists(path_x3):
                self.gt_paths.append(os.path.join(gt_path, gt_name))
                self.lq_paths.append(path_x3)
            elif os.path.exists(path_same):
                self.gt_paths.append(os.path.join(gt_path, gt_name))
                self.lq_paths.append(path_same)

        print(f"\n🚀 [极速模式] 正在将 {len(self.gt_paths)} 张图片完全解码并载入 120GB 内存池...")
        for i in tqdm(range(len(self.gt_paths)), desc="Loading decoded images to RAM"):
            img_gt = cv2.imread(self.gt_paths[i], cv2.IMREAD_COLOR)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

            img_lq = cv2.imread(self.lq_paths[i], cv2.IMREAD_COLOR)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

            self.data_cache.append((img_gt, img_lq))

        print("✅ 内存加载完成！正在清理多余缓存...")
        gc.collect()

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):

        img_gt, img_lq = self.data_cache[idx]

        h_lq, w_lq, _ = img_lq.shape
        img_gt_view = img_gt[:h_lq * self.scale, :w_lq * self.scale, :]


        img_gt_crop, img_lq_crop = paired_random_crop(img_gt_view, img_lq, self.gt_size, self.scale)


        img_gt_crop = img_gt_crop.copy()
        img_lq_crop = img_lq_crop.copy()


        img_gt_aug, img_lq_aug = augment([img_gt_crop, img_lq_crop], hflip=True, rotation=True)

        img_gt_t = self.to_tensor(img_gt_aug)
        img_lq_t = self.to_tensor(img_lq_aug)

        return img_lq_t, img_gt_t

class ValDataset(Dataset):
    def __init__(self, gt_path, lq_path, scale=3):
        self.scale = scale
        self.to_tensor = ToTensor()
        self.gt_paths = []
        self.lq_paths = []

        gt_files = sorted(os.listdir(gt_path))
        for gt_name in gt_files:
            if not gt_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            basename, ext = os.path.splitext(gt_name)
            lq_name_x3 = f"{basename}x{scale}{ext}"
            lq_name_same = f"{basename}{ext}"
            path_x3 = os.path.join(lq_path, lq_name_x3)
            path_same = os.path.join(lq_path, lq_name_same)

            if os.path.exists(path_x3):
                self.gt_paths.append(os.path.join(gt_path, gt_name))
                self.lq_paths.append(path_x3)
            elif os.path.exists(path_same):
                self.gt_paths.append(os.path.join(gt_path, gt_name))
                self.lq_paths.append(path_same)

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        img_gt = cv2.imread(self.gt_paths[idx], cv2.IMREAD_COLOR)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        img_lq = cv2.imread(self.lq_paths[idx], cv2.IMREAD_COLOR)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        h_lq, w_lq, _ = img_lq.shape
        img_gt = img_gt[:h_lq * self.scale, :w_lq * self.scale, :]
        img_gt = self.to_tensor(img_gt.copy())
        img_lq = self.to_tensor(img_lq.copy())
        return img_lq, img_gt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='train_RFDN_x3.yml')
    parser.add_argument('--resume', type=str, default=None, help='手动指定具体要恢复的权重路径')
    parser.add_argument('--auto_resume', action='store_true', help='自动寻找最新权重断点续训')
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.safe_load(f)


    os.makedirs('experiments/checkpoints_x3', exist_ok=True)

    logger_info('Train_x3', log_path='experiments/train_x3.log')
    logger = logging.getLogger('Train_x3')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = RFDN(
        in_nc=opt['network_g']['num_in_ch'],
        nf=opt['network_g']['num_feat'],
        num_modules=opt['network_g']['num_block'],
        out_nc=opt['network_g']['num_out_ch'],
        upscale=opt['network_g']['upscale']
    ).to(device)

    logger.info("Compiling model with torch.compile...")
    model = torch.compile(model, mode="reduce-overhead")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {num_params / 1000:.2f} K")

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['train']['lr'])
    criterion = nn.L1Loss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['train']['total_iter'], eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda', enabled=True)
    start_iter = 0

    if args.auto_resume and not args.resume:

        ckpt_dir = 'experiments/checkpoints_x3'
        if os.path.exists(ckpt_dir):
            ckpts = glob.glob(os.path.join(ckpt_dir, 'model_*.pth'))
            if ckpts:
                latest_ckpt = max(ckpts, key=os.path.getmtime)
                args.resume = latest_ckpt
                logger.info(f"Auto-resume triggered. Found latest checkpoint: {latest_ckpt}")

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_iter = checkpoint['iter']
        logger.info(f"Successfully resumed from iter {start_iter}!")

    safe_workers = 4
    logger.info(f"Setting num_workers to {safe_workers} (CoW RAM explosion fixed).")

    train_loader = DataLoader(
        SRDataset(opt['datasets']['train']['dataroot_gt'], opt['datasets']['train']['dataroot_lq'],
                  opt['datasets']['train']['gt_size'], opt['scale']),
        batch_size=opt['datasets']['train']['batch_size_per_gpu'],
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        ValDataset(opt['datasets']['val']['dataroot_gt'], opt['datasets']['val']['dataroot_lq'], opt['scale']),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    current_iter = start_iter
    total_iter = opt['train']['total_iter']
    val_freq = opt['train']['val_freq']

    pbar = tqdm(total=total_iter, initial=start_iter, desc="Training")

    model.train()
    while current_iter < total_iter:
        for lq, gt in train_loader:
            if current_iter >= total_iter:
                break

            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=True):
                output = model(lq)
                loss = criterion(output, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            current_iter += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", iter=current_iter)

            if current_iter % 1000 == 0:
                logger.info(f"Iter: {current_iter}/{total_iter} | Loss: {loss.item():.4f}")

            if current_iter % val_freq == 0:
                model.eval()
                psnr_sum = 0.0
                ssim_sum = 0.0

                with torch.no_grad():

                    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    for val_lq, val_gt in tqdm(val_loader, desc="Validating", leave=False):
                        val_lq = val_lq.to(device, non_blocking=True)

                        with torch.amp.autocast('cuda', enabled=True):

                            out_val = eval_model(val_lq)

                        out_val_uint = util.tensor2uint(out_val)
                        gt_val_uint = util.tensor2uint(val_gt)

                        psnr_sum += util.calculate_psnr(out_val_uint, gt_val_uint, border=opt['scale'],
                                                        test_y_channel=True)
                        ssim_sum += util.calculate_ssim(out_val_uint, gt_val_uint, border=opt['scale'],
                                                        test_y_channel=True)

                avg_psnr = psnr_sum / len(val_loader)
                avg_ssim = ssim_sum / len(val_loader)
                logger.info(
                    f"--- Iter {current_iter} | Baseline x3 Y-Channel PSNR/SSIM: {avg_psnr:.4f} dB / {avg_ssim:.4f} ---")


                torch.save({
                    'iter': current_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, f"experiments/checkpoints_x3/model_{current_iter}.pth")
                model.train()

    pbar.close()
    logger.info("Training finished.")

if __name__ == '__main__':
    main()
