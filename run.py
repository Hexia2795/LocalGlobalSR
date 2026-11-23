from __future__ import annotations
import argparse
import importlib
from pathlib import Path
from srpaper import runtime as rt


def main():
    parser = argparse.ArgumentParser(description="Single-image super-resolution experiments")
    commands = parser.add_subparsers(dest="task", required=True)
    configure = commands.add_parser("configure", help="Set data and checkpoint paths")
    configure.add_argument("--data-root", required=True, type=Path)
    configure.add_argument("--checkpoint-dir", type=Path)
    verify = commands.add_parser("verify", help="Check checkpoints and model outputs on CPU")
    verify.add_argument("--reference-root", type=Path)
    verify.add_argument("--checkpoint-dir", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate PSNR and SSIM")
    evaluate.add_argument("--models", nargs="+", choices=list(rt.model_specs()), default=["serial_x4_seed0"])
    evaluate.add_argument("--datasets", nargs="+", default=["Set5"])
    evaluate.add_argument("--inference", nargs="+", choices=["Standard","TLC"], default=["Standard"])
    evaluate.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    evaluate.add_argument("--data-root", type=Path)
    evaluate.add_argument("--checkpoint-dir", type=Path)
    evaluate.add_argument("--save-images", action="store_true")
    conditioning = commands.add_parser("conditioning", help="Run the global-input intervention")
    conditioning.add_argument("--seeds", nargs="+", type=int, choices=[0,1], default=[0,1])
    conditioning.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    conditioning.add_argument("--data-root", type=Path)
    conditioning.add_argument("--checkpoint-dir", type=Path)
    analyze = commands.add_parser("analyze", help="Analyze saved results")
    analyze.add_argument("--resample", action="store_true")
    commands.add_parser("figures", help="Plot ECDF and intervention results")
    train = commands.add_parser("train", help="Train a model")
    train.add_argument("--model", required=True, choices=list(rt.model_specs()))
    train.add_argument("--data-root", type=Path)
    train.add_argument("--stop-after", type=int, default=885000)
    benchmark = commands.add_parser("benchmark", help="Measure inference time and memory on GPU")
    benchmark.add_argument("--checkpoint-dir", type=Path)
    macs = commands.add_parser("macs", help="Count model MACs")
    macs.add_argument("--models", nargs="+", choices=list(rt.model_specs()), default=["serial_x4_seed0"])
    macs.add_argument("--inference", nargs="+", choices=["Standard","TLC"], default=["Standard","TLC"])
    macs.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    macs.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    if args.task == "configure":
        config = rt.settings(args.data_root, args.checkpoint_dir)
        rt.write_json(rt.ROOT/"configs/paths.local.json", config)
        print(config)
        return
    modules = {"verify":"tools.verify_release", "evaluate":"scripts.eval.evaluate", "conditioning":"scripts.analysis.conditioning",
               "analyze":"scripts.analysis.analyze", "figures":"scripts.figures.plot", "train":"scripts.train.launch", "benchmark":"scripts.benchmark.launch", "macs":"scripts.analysis.macs"}
    importlib.import_module(modules[args.task]).run(args)


if __name__ == "__main__":
    main()
