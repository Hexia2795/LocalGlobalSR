from srpaper import runtime as rt


def run(args):
    import torch
    from scripts.benchmark import benchmark_reference as benchmark
    if not torch.cuda.is_available():
        raise RuntimeError("The execution benchmark requires CUDA")
    out = rt.new_run("benchmark")
    directory = rt.settings(checkpoint_dir=args.checkpoint_dir)["checkpoint_dir"]
    specs = rt.model_specs()
    from pathlib import Path
    benchmark.FAIR_CKPT = Path(directory)/specs["fair_rfdn_x4"]["file"]
    benchmark.SERIAL_CKPT = Path(directory)/specs["serial_x4_seed0"]["file"]
    benchmark.PARALLEL_CKPT = Path(directory)/specs["p05_x4_seed0"]["file"]
    for key in ("fair_rfdn_x4","serial_x4_seed0","p05_x4_seed0"):
        if rt.sha256(Path(directory)/specs[key]["file"]) != specs[key]["sha256"]:
            raise ValueError("Checkpoint hash differs: "+key)
    benchmark.OUTPUT_DIR = out
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("highest")
    results = []
    for variant in ["Fair RFDN", "Parallel Fixed", "Serial", "Parallel Fixed + TLC", "Serial + TLC"]:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        results.append(benchmark.benchmark_one(variant, torch.device("cuda:0")))
    benchmark.write_results(results)
    print("Results:", out)
