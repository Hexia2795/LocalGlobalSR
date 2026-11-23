from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def new_run(task, test=False):
    parent = ROOT / "experiments" / "tests" if test else ROOT / "experiments"
    path = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + task)
    path.mkdir(parents=True, exist_ok=False)
    return path


def settings(data_root=None, checkpoint_dir=None):
    local = ROOT / "configs/paths.local.json"
    result = read_json(local) if local.exists() else {}
    if data_root is not None:
        result["data_root"] = str(Path(data_root).expanduser().resolve())
    if checkpoint_dir is not None:
        result["checkpoint_dir"] = str(Path(checkpoint_dir).expanduser().resolve())
    result.setdefault("checkpoint_dir", "checkpoints")
    for key in ("data_root", "checkpoint_dir"):
        if result.get(key):
            path = Path(result[key]).expanduser()
            result[key] = str(path if path.is_absolute() else ROOT / path)
    return result


def model_specs():
    return read_json(ROOT / "configs/models.json")


def constructor(composition, package="srpaper.models"):
    names = {"serial": "RFDN", "p05": "RFDN_parallel_fixed", "p1": "RFDN_parallel_unit", "fair": "rfdn_baseline_arch"}
    return importlib.import_module(package + "." + names[composition]).RFDN


def instantiate(spec, package="srpaper.models"):
    options = dict(in_nc=3, nf=48, num_modules=4, out_nc=3, upscale=spec["scale"])
    if spec["composition"] == "fair":
        options["distillation_rate"] = 0.5
    return constructor(spec["composition"], package)(**options)


def extract_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "params_ema", "params"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    return {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in checkpoint.items()}


def load_model(label, device="cpu", checkpoint_dir=None, package="srpaper.models"):
    import torch
    spec = model_specs()[label]
    path = Path(settings(checkpoint_dir=checkpoint_dir)["checkpoint_dir"]) / spec["file"]
    digest = sha256(path)
    if digest != spec["sha256"]:
        raise ValueError("Checkpoint hash differs: " + str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    iteration = checkpoint.get("iter", checkpoint.get("iteration", checkpoint.get("current_iter")))
    if iteration is not None and int(iteration) != spec["iteration"]:
        raise ValueError("Unexpected checkpoint iteration: " + str(iteration))
    model = instantiate(spec, package)
    model.load_state_dict(extract_state(checkpoint), strict=True)
    params = sum(p.numel() for p in model.parameters())
    if params != spec["params"]:
        raise ValueError("Unexpected parameter count: " + str(params))
    return model.float().to(device).eval(), {**spec, "label": label, "sha256": digest, "strict_load": True, "checkpoint_iteration": iteration}


def set_tlc(model, enabled):
    count = 0
    for module in model.modules():
        if hasattr(module, "tlc_flag"):
            module.tlc_flag = bool(enabled)
            count += 1
    return count


def inference_precision():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def read_rgb(path):
    import cv2
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError("Cannot read image: " + str(path))
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def image_tensor(image, device):
    import numpy as np
    import torch
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0).to(device)


def pairs(dataset, scale, data_root):
    if not data_root:
        raise ValueError("Set --data-root or run configure first")
    info = read_json(ROOT / "configs/datasets.json")[str(scale)][dataset]
    hr_dir, lr_dir = (Path(data_root) / info[k] for k in ("hr", "lr"))
    if not hr_dir.is_dir() or not lr_dir.is_dir():
        raise FileNotFoundError("Dataset folders: " + str(hr_dir) + " ; " + str(lr_dir))
    files = sorted(p for p in hr_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"})
    result = []
    for hr in files:
        candidates = [lr_dir/hr.name, lr_dir/(hr.stem+"x"+str(scale)+hr.suffix), lr_dir/(hr.stem+"X"+str(scale)+hr.suffix)]
        lr = next((p for p in candidates if p.is_file()), None)
        if lr is None:
            raise FileNotFoundError("Missing LR image for " + str(hr))
        result.append((hr, lr))
    if len(result) != info["images"]:
        raise ValueError(f"{dataset}: expected {info['images']} pairs, found {len(result)}")
    return result


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_file(label, dataset, mode):
    spec = model_specs()[label]
    base = ROOT / "results/reference"
    if spec["scale"] != 4:
        return base / "scales23" / f"x{spec['scale']}_{dataset}_{mode}_per_image.csv"
    seed = spec["seed"]
    token = mode.lower()
    if spec["composition"] == "fair":
        return base / "serial_p05_seed0" / f"fair_baseline_{dataset}_per_image.csv"
    if spec["composition"] == "p1":
        return base / f"p1_seed{seed}" / f"parallel_unit_seed{seed}_{token}_{dataset}_per_image.csv"
    prefix = "serial" if spec["composition"] == "serial" else "parallel"
    return base / f"serial_p05_seed{seed}" / f"{prefix}_seed{seed}_{token}_{dataset}_per_image.csv"


def mean(rows, field):
    return statistics.mean(float(row[field]) for row in rows)
