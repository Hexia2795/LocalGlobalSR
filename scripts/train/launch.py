from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
from srpaper import runtime as rt

RECIPES = {
    "serial_x4_seed0": ("train_x4", "serial_x4_seed0.yml"),
    "serial_x4_seed1": ("train_x4_seed1", "serial_x4_seed1.yml"),
    "p05_x4_seed0": ("train_x4_parallel_fixed", "parallel_x4_seed0.yml"),
    "p05_x4_seed1": ("train_x4_parallel_fixed_seed1", "parallel_x4_seed1.yml"),
    "p1_x4_seed0": ("train_x4_parallel_fixed", "parallel_unit_x4_seed0.yml"),
    "p1_x4_seed1": ("train_x4_parallel_fixed", "parallel_unit_x4_seed1.yml"),
    "fair_rfdn_x4": ("train_rfdn_baseline_x4", "fair_x4.yml"),
    "serial_x2": ("train_x2", "serial_x2.yml"),
    "serial_x3": ("train_x3", "serial_x3.yml"),
}


def replace_paths(value, data_root):
    if isinstance(value, str):
        return value.replace("${DATA_ROOT}", str(data_root))
    if isinstance(value, dict):
        return {k:replace_paths(v,data_root) for k,v in value.items()}
    if isinstance(value, list):
        return [replace_paths(v,data_root) for v in value]
    return value


def run(args):
    import torch
    import yaml
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; verification and analysis can use CPU")
    config = rt.settings(args.data_root)
    if not config.get("data_root"):
        raise ValueError("Set --data-root or run configure first")
    name, recipe = RECIPES[args.model]
    out = rt.new_run("train_"+args.model)
    module = importlib.import_module("scripts.train."+name)
    options = yaml.safe_load((rt.ROOT/"configs/training"/recipe).read_text())
    options = replace_paths(options, config["data_root"])
    if args.model.startswith("p1_"):
        module.RFDN = rt.constructor("p1")
    if args.model == "fair_rfdn_x4":
        options["paths"]["output_dir"] = str(out/"training")
    else:
        module.CHECKPOINT_DIR = str(out/"checkpoints")
        module.LOG_PATH = str(out/"train.log")
    used = out/"used_config.yml"
    used.write_text(yaml.safe_dump(options, sort_keys=False))
    rt.write_json(out/"release_training_manifest.json", {"model":args.model,"recipe":recipe,
                  "stop_after":args.stop_after,"schedule_iterations":options["train"]["total_iter"],"fresh_run":True})
    sys.argv = [name, "-opt", str(used), "--stop_after", str(args.stop_after)]
    print("Training output:", out, flush=True)
    module.main()
