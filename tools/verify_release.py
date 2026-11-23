from __future__ import annotations
import ast
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import traceback
from srpaper import runtime as rt


def computational_ast(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    class RemoveNotes(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)
    return ast.dump(RemoveNotes().visit(tree))


def install_reference(root):
    name = "_sr_release_reference"
    spec = importlib.util.spec_from_file_location(name, Path(root)/"srpaper/__init__.py",
                                                 submodule_search_locations=[str(Path(root)/"srpaper")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name + ".models"


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MPLBACKEND", "Agg")
    out = rt.new_run("verify_release", test=True)
    report = {"task": "verify_release", "status": "started", "device": "cpu", "models": [],
              "source_checks": [], "training_performed": False, "dataset_evaluation_performed": False}
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        report["torch"] = torch.__version__
        report["numpy"] = np.__version__
        sources = rt.read_json(rt.ROOT/"docs/source_manifest.json")
        for item in sources:
            path = rt.ROOT/item["path"]
            record = {"path": item["path"], "release_hash_matches": rt.sha256(path) == item["release_sha256"]}
            if args.reference_root:
                reference = Path(args.reference_root)/item["path"]
                record["reference_hash_matches"] = rt.sha256(reference) == item["frozen_sha256"]
                record["computational_ast_matches"] = computational_ast(path) == computational_ast(reference)
            report["source_checks"].append(record)
            if not all(v for k,v in record.items() if k != "path"):
                raise ValueError("Source identity differs: " + item["path"])
        package = install_reference(args.reference_root) if args.reference_root else None
        if args.reference_root:
            source = Path(args.reference_root)/"scripts/analysis/audit_smfanet_metric_protocol.py"
            spec = importlib.util.spec_from_file_location("_release_metric_reference", source)
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            native, _ = helper.native_metrics(Path(args.reference_root)/"third_party/SMFANet/a05e0bb554bbbb80cd27d8fd2b78d4228c657170")
            from srpaper.utils import metrics
            rng = np.random.default_rng(20261123)
            first = rng.integers(0, 256, (31, 37, 3), dtype=np.uint8)
            second = np.clip(first.astype(float)+rng.integers(-4, 5, first.shape), 0, 255).astype(np.uint8)
            report["native_metric_parity"] = []
            for border in (2, 3, 4):
                for name in ("calculate_psnr", "calculate_ssim"):
                    a = float(getattr(metrics,name)(first,second,crop_border=border,test_y_channel=True))
                    b = float(getattr(native,name)(first,second,crop_border=border,test_y_channel=True))
                    if a != b:
                        raise ValueError("Native metric differs: " + name)
                    report["native_metric_parity"].append({"function":name,"border":border,"exact_match":True})
        generator = torch.Generator().manual_seed(20261123)
        sample = torch.rand(1, 3, 52, 60, generator=generator)
        with torch.inference_mode():
            for label, spec in rt.model_specs().items():
                model, record = rt.load_model(label, checkpoint_dir=args.checkpoint_dir)
                reference = rt.load_model(label, checkpoint_dir=args.checkpoint_dir, package=package)[0] if package else None
                record["forward_checks"] = []
                for mode in ([False] if spec["composition"] == "fair" else [False, True]):
                    count = rt.set_tlc(model, mode)
                    if count != (0 if spec["composition"] == "fair" else 4):
                        raise ValueError("Unexpected TLC module count")
                    predicted = model(sample.clone())
                    expected_shape = (1, 3, 52*spec["scale"], 60*spec["scale"])
                    if tuple(predicted.shape) != expected_shape or not torch.isfinite(predicted).all():
                        raise ValueError("Invalid output: " + label)
                    check = {"inference": "TLC" if mode else "Standard", "shape": list(predicted.shape), "finite": True}
                    if reference is not None:
                        rt.set_tlc(reference, mode)
                        other = reference(sample.clone())
                        check["exact_reference_parity"] = bool(torch.equal(predicted, other))
                        check["max_abs_difference"] = float((predicted-other).abs().max())
                        if not check["exact_reference_parity"]:
                            raise ValueError("CPU forward differs from frozen source: " + label)
                    record["forward_checks"].append(check)
                report["models"].append(record)
                print("PASS", label, "strict load / parameters / CPU forward", flush=True)
                del model, reference
        recipes = ["train_x4", "train_x4_seed1", "train_x4_parallel_fixed", "train_x4_parallel_fixed_seed1", "train_x2", "train_x3", "train_rfdn_baseline_x4"]
        for name in recipes:
            importlib.import_module("scripts.train."+name)
        report["training_modules_imported"] = recipes
        from scripts.analysis.analyze import audit
        report["saved_results_audit"] = audit()
        report["status"] = "passed"
        print("PASS: all nine checkpoints and saved-result checks", flush=True)
    except Exception as error:
        report.update(status="failed", error=type(error).__name__+": "+str(error), traceback=traceback.format_exc())
        raise
    finally:
        rt.write_json(out/"verification_report.json", report)
        print("REPORT:", out/"verification_report.json", flush=True)
