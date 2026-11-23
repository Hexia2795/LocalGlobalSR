from srpaper import runtime as rt


def run(args):
    import torch
    from scripts.eval.evaluate import evaluate_model
    from srpaper.models.conditioning import ConditioningIsolationBlock
    rt.inference_precision()
    config = rt.settings(args.data_root, args.checkpoint_dir)
    images = rt.pairs("Urban100", 4, config.get("data_root"))
    out = rt.new_run("conditioning")
    summary = []
    for seed in args.seeds:
        label = "serial_x4_seed" + str(seed)
        model, _ = rt.load_model(label, args.device, config["checkpoint_dir"])
        model.requires_grad_(False)
        rt.set_tlc(model, False)
        baseline = evaluate_model(model, images, 4, args.device)
        rt.write_csv(out/f"seed{seed}_baseline_per_image.csv", baseline)
        for block_name in ("B1", "B2", "B3", "B4"):
            original = getattr(model, block_name)
            try:
                for beta in (1.0, 0.75, 0.5, 0.25, 0.0):
                    wrapped = ConditioningIsolationBlock(original, beta)
                    wrapped.eval()
                    setattr(model, block_name, wrapped)
                    rows = evaluate_model(model, images, 4, args.device)
                    rt.write_csv(out/f"seed{seed}_{block_name}_beta{beta:.2f}_per_image.csv", rows)
                    entry = {"training_seed": seed, "block": block_name, "beta": beta, "images": len(rows),
                             "psnr": rt.mean(rows,"psnr"), "ssim": rt.mean(rows,"ssim"),
                             "delta_psnr": rt.mean(rows,"psnr") - rt.mean(baseline,"psnr"),
                             "delta_ssim": rt.mean(rows,"ssim") - rt.mean(baseline,"ssim")}
                    summary.append(entry)
                    print(entry, flush=True)
            finally:
                setattr(model, block_name, original)
    rt.write_csv(out/"summary.csv", summary)
    print("Results:", out)
