from srpaper import runtime as rt


def evaluate_model(model, image_pairs, scale, device, output_images=None):
    import cv2
    import torch
    from srpaper.utils.image import tensor2uint
    from srpaper.utils.metrics import calculate_psnr, calculate_ssim
    rows = []
    with torch.inference_mode():
        for index, (hr_path, lr_path) in enumerate(image_pairs, 1):
            hr, lr = rt.read_rgb(hr_path), rt.read_rgb(lr_path)
            height, width = lr.shape[0]*scale, lr.shape[1]*scale
            bottom, right = hr.shape[0]-height, hr.shape[1]-width
            if bottom < 0 or right < 0 or bottom >= scale or right >= scale:
                raise ValueError("HR/LR dimensions differ: " + hr_path.name)
            if scale == 4 and (bottom != 0 or right != 0):
                raise ValueError("The controlled x4 study requires exact paired dimensions")
            hr = hr[:height, :width, :]
            sr = tensor2uint(model(rt.image_tensor(lr, device)))
            if sr.shape != hr.shape:
                raise ValueError("SR dimensions differ: " + hr_path.name)
            rows.append({"index": index, "image": hr_path.name,
                         "psnr": float(calculate_psnr(sr[..., ::-1], hr[..., ::-1], crop_border=scale, test_y_channel=True)),
                         "ssim": float(calculate_ssim(sr[..., ::-1], hr[..., ::-1], crop_border=scale, test_y_channel=True))})
            if output_images is not None:
                output_images.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_images/hr_path.name), sr[..., ::-1])
    return rows


def run(args):
    import torch
    rt.inference_precision()
    out = rt.new_run("evaluate")
    config = rt.settings(args.data_root, args.checkpoint_dir)
    report = {"task": "evaluate", "status": "started", "device": args.device, "torch": torch.__version__,
              "protocol": "BasicSR float-Y after RGB uint8 reconstruction; crop=scale; FP32; batch=1", "models": []}
    summary = []
    try:
        for label in args.models:
            spec = rt.model_specs()[label]
            for mode in args.inference:
                if spec["composition"] == "fair" and mode == "TLC":
                    continue
                model, record = rt.load_model(label, args.device, config["checkpoint_dir"])
                record["inference"] = mode
                record["tlc_modules"] = rt.set_tlc(model, mode == "TLC")
                report["models"].append(record)
                for dataset in args.datasets:
                    images = rt.pairs(dataset, spec["scale"], config.get("data_root"))
                    folder = out / label / mode
                    folder.mkdir(parents=True, exist_ok=True)
                    rows = evaluate_model(model, images, spec["scale"], args.device, folder/dataset if args.save_images else None)
                    rt.write_csv(folder/(dataset+"_per_image.csv"), rows)
                    entry = {"model": label, "scale": spec["scale"], "dataset": dataset, "images": len(rows),
                             "inference": mode, "psnr": rt.mean(rows, "psnr"), "ssim": rt.mean(rows, "ssim")}
                    reference = rt.reference_file(label, dataset, mode)
                    if reference.exists():
                        old = rt.read_csv(reference)
                        if [r["image"] for r in rows] != [r["image"] for r in old]:
                            raise ValueError("Reference image IDs differ")
                        entry["delta_psnr_from_archived_mean"] = entry["psnr"] - rt.mean(old, "psnr")
                        entry["delta_ssim_from_archived_mean"] = entry["ssim"] - rt.mean(old, "ssim")
                    summary.append(entry)
                    print(label, mode, dataset, "PSNR %.6f SSIM %.8f" % (entry["psnr"], entry["ssim"]), flush=True)
                del model
        rt.write_csv(out/"summary.csv", summary)
        report["status"] = "completed"
    finally:
        rt.write_json(out/"run_manifest.json", report)
        print("Results:", out, flush=True)
