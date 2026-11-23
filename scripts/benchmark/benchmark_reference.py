from __future__ import annotations

import csv
import gc
import statistics
from pathlib import Path

import torch

from srpaper.models.rfdn_baseline_arch import (
    RFDN as FairRFDN,
)

from srpaper.models.RFDN_parallel_fixed import (
    RFDN as ParallelFixedRFDN,
)

from srpaper.models.RFDN import (
    RFDN as SerialRFDN,
)


ROOT = Path(
    "/root/autodl-tmp/demo/RFDN-master-2"
)

OUTPUT_DIR = (
    ROOT
    / "experiments/"
      "final_efficiency_x4_720p"
)

FAIR_CKPT = (
    ROOT
    / "experiments/final_fair_885k/"
      "checkpoints/"
      "fair_rfdn_baseline_885000.pth"
)

SERIAL_CKPT = (
    ROOT
    / "experiments/final_fair_885k/"
      "checkpoints/"
      "improved_serial_885000.pth"
)

PARALLEL_CKPT = (
    ROOT
    / "experiments/final_parallel_fixed_885k/"
      "checkpoints/"
      "parallel_fixed_885000.pth"
)


EXPECTED_PARAMS = {
    "Fair RFDN": 402288,
    "Parallel Fixed": 361696,
    "Serial": 361696,
    "Parallel Fixed + TLC": 361696,
    "Serial + TLC": 361696,
}


LR_HEIGHT = 180
LR_WIDTH = 320
SCALE = 4

EXPECTED_OUTPUT_SHAPE = (
    1,
    3,
    LR_HEIGHT * SCALE,
    LR_WIDTH * SCALE,
)

WARMUP_RUNS = 30

ROUNDS = 5
RUNS_PER_ROUND = 50


def count_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def load_checkpoint_cpu(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    return checkpoint


def clean_state_dict(checkpoint):

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state = checkpoint[
            "state_dict"
        ]

    else:
        state = checkpoint

    cleaned = {}

    for key, value in state.items():

        key = key.removeprefix(
            "_orig_mod."
        )

        key = key.removeprefix(
            "module."
        )

        cleaned[key] = value

    return cleaned


def strict_load(
    model,
    checkpoint_path,
    name,
):

    checkpoint = load_checkpoint_cpu(
        checkpoint_path
    )

    state = clean_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    source_iter = None

    if isinstance(
        checkpoint,
        dict,
    ):
        source_iter = checkpoint.get(
            "iter",
            None,
        )

    print(
        f"{name}: strict checkpoint "
        f"load PASS"
    )

    print(
        f"{name}: checkpoint iter = "
        f"{source_iter}"
    )

    return model


def set_tlc(
    model,
    enabled,
):

    count = 0

    for module in model.modules():

        if hasattr(
            module,
            "tlc_flag",
        ):
            module.tlc_flag = bool(
                enabled
            )

            count += 1


            for attr in (
                "_cached_h_w",
                "_cached_ij",
                "_cached_count_mt",
            ):
                if hasattr(
                    module,
                    attr,
                ):
                    delattr(
                        module,
                        attr,
                    )

    return count


def build_model(
    variant,
):

    if variant == "Fair RFDN":

        model = FairRFDN(
            in_nc=3,
            nf=48,
            num_modules=4,
            out_nc=3,
            upscale=4,
        )

        model = strict_load(
            model,
            FAIR_CKPT,
            variant,
        )

        tlc_enabled = False
        tlc_modules = 0

    elif variant in (
        "Parallel Fixed",
        "Parallel Fixed + TLC",
    ):

        model = ParallelFixedRFDN(
            in_nc=3,
            nf=48,
            num_modules=4,
            out_nc=3,
            upscale=4,
        )

        model = strict_load(
            model,
            PARALLEL_CKPT,
            variant,
        )

        tlc_enabled = (
            variant
            == "Parallel Fixed + TLC"
        )

        tlc_modules = set_tlc(
            model,
            tlc_enabled,
        )

        if tlc_modules != 4:
            raise RuntimeError(
                f"{variant}: expected "
                f"4 TLC-capable global "
                f"attention modules, got "
                f"{tlc_modules}."
            )

    elif variant in (
        "Serial",
        "Serial + TLC",
    ):

        model = SerialRFDN(
            in_nc=3,
            nf=48,
            num_modules=4,
            out_nc=3,
            upscale=4,
        )

        model = strict_load(
            model,
            SERIAL_CKPT,
            variant,
        )

        tlc_enabled = (
            variant
            == "Serial + TLC"
        )

        tlc_modules = set_tlc(
            model,
            tlc_enabled,
        )

        if tlc_modules != 4:
            raise RuntimeError(
                f"{variant}: expected "
                f"4 TLC-capable global "
                f"attention modules, got "
                f"{tlc_modules}."
            )

    else:

        raise ValueError(
            f"Unknown variant: "
            f"{variant}"
        )

    params = count_params(
        model
    )

    expected = EXPECTED_PARAMS[
        variant
    ]

    if params != expected:

        raise RuntimeError(
            f"{variant}: parameter "
            f"count mismatch. "
            f"Expected {expected:,}, "
            f"got {params:,}."
        )

    return (
        model,
        tlc_enabled,
        tlc_modules,
    )


def benchmark_one(
    variant,
    device,
):

    print()
    print("=" * 100)
    print(variant)
    print("=" * 100)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    (
        model,
        tlc_enabled,
        tlc_modules,
    ) = build_model(
        variant
    )

    params = count_params(
        model
    )

    model = (
        model
        .to(device)
        .eval()
    )


    x = torch.randn(
        1,
        3,
        LR_HEIGHT,
        LR_WIDTH,
        dtype=torch.float32,
        device=device,
    )


    with torch.inference_mode():

        y = model(
            x
        )

    torch.cuda.synchronize()

    output_shape = tuple(
        y.shape
    )

    finite = bool(
        torch.isfinite(
            y
        ).all()
    )

    print(
        "Params:",
        f"{params:,}",
    )

    print(
        "TLC:",
        tlc_enabled,
    )

    print(
        "TLC-capable modules:",
        tlc_modules,
    )

    print(
        "Input:",
        tuple(
            x.shape
        ),
    )

    print(
        "Output:",
        output_shape,
    )

    print(
        "Finite:",
        finite,
    )

    if (
        output_shape
        != EXPECTED_OUTPUT_SHAPE
    ):
        raise RuntimeError(
            f"{variant}: output "
            f"shape mismatch."
        )

    if not finite:
        raise RuntimeError(
            f"{variant}: NaN/Inf "
            f"in output."
        )

    del y


    print(
        f"Warmup: {WARMUP_RUNS} runs"
    )

    with torch.inference_mode():

        for _ in range(
            WARMUP_RUNS
        ):
            _ = model(
                x
            )

    torch.cuda.synchronize()


    del _

    torch.cuda.reset_peak_memory_stats(
        device
    )

    with torch.inference_mode():

        y = model(
            x
        )

    torch.cuda.synchronize()

    peak_allocated_mb = (
        torch.cuda
        .max_memory_allocated(
            device
        )
        / (1024 ** 2)
    )

    peak_reserved_mb = (
        torch.cuda
        .max_memory_reserved(
            device
        )
        / (1024 ** 2)
    )

    del y


    round_latencies = []

    for round_idx in range(
        ROUNDS
    ):

        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        torch.cuda.synchronize()

        start_event.record()

        with torch.inference_mode():

            for _ in range(
                RUNS_PER_ROUND
            ):
                _ = model(
                    x
                )

        end_event.record()

        torch.cuda.synchronize()

        total_ms = (
            start_event
            .elapsed_time(
                end_event
            )
        )

        latency_ms = (
            total_ms
            / RUNS_PER_ROUND
        )

        round_latencies.append(
            float(
                latency_ms
            )
        )

        print(
            f"Round "
            f"{round_idx + 1}/{ROUNDS}: "
            f"{latency_ms:.4f} ms"
        )

    latency_mean = float(
        statistics.mean(
            round_latencies
        )
    )

    latency_std = float(
        statistics.stdev(
            round_latencies
        )
        if len(
            round_latencies
        ) > 1
        else 0.0
    )

    latency_min = float(
        min(
            round_latencies
        )
    )

    fps = float(
        1000.0
        / latency_mean
    )

    print()
    print(
        f"Latency mean : "
        f"{latency_mean:.4f} ms"
    )

    print(
        f"Latency std  : "
        f"{latency_std:.4f} ms"
    )

    print(
        f"Latency min  : "
        f"{latency_min:.4f} ms"
    )

    print(
        f"Throughput   : "
        f"{fps:.2f} image/s"
    )

    print(
        f"Peak allocated VRAM: "
        f"{peak_allocated_mb:.2f} MiB"
    )

    print(
        f"Peak reserved VRAM : "
        f"{peak_reserved_mb:.2f} MiB"
    )

    result = {
        "variant": variant,
        "params": params,
        "params_k": (
            params / 1000.0
        ),
        "tlc": (
            "ON"
            if tlc_enabled
            else "OFF"
        ),
        "input_h": LR_HEIGHT,
        "input_w": LR_WIDTH,
        "output_h": (
            LR_HEIGHT
            * SCALE
        ),
        "output_w": (
            LR_WIDTH
            * SCALE
        ),
        "precision": "FP32",
        "round_latencies_ms": list(round_latencies),
        "latency_mean_ms": (
            latency_mean
        ),
        "latency_std_ms": (
            latency_std
        ),
        "latency_min_ms": (
            latency_min
        ),
        "throughput_img_s": fps,
        "peak_allocated_vram_mb": (
            float(
                peak_allocated_mb
            )
        ),
        "peak_reserved_vram_mb": (
            float(
                peak_reserved_mb
            )
        ),
    }

    del model
    del x

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return result


def write_results(
    results,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    fieldnames = list(
        results[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    txt_path = (
        OUTPUT_DIR
        / "summary.txt"
    )

    lines = [
        (
            "Final x4 Network-forward "
            "Efficiency Benchmark"
        ),
        "=" * 110,
        "",
        (
            "Hardware: current CUDA GPU"
        ),
        (
            "Input LR: "
            f"1x3x{LR_HEIGHT}x{LR_WIDTH}"
        ),
        (
            "Output HR: "
            f"1x3x"
            f"{LR_HEIGHT * SCALE}x"
            f"{LR_WIDTH * SCALE}"
        ),
        "Scale: x4",
        "Precision: FP32",
        "TF32: OFF",
        "Batch size: 1",
        (
            f"Warmup runs: "
            f"{WARMUP_RUNS}"
        ),
        (
            f"Timing: "
            f"{ROUNDS} rounds x "
            f"{RUNS_PER_ROUND} runs"
        ),
        "",
    ]

    header = (
        f"{'Variant':<24}"
        f"{'Params(K)':>12}"
        f"{'TLC':>8}"
        f"{'Latency(ms)':>16}"
        f"{'Std(ms)':>12}"
        f"{'PeakAlloc(MiB)':>16}"
        f"{'PeakReserv(MiB)':>17}"
    )

    lines.append(
        header
    )

    lines.append(
        "-" * len(
            header
        )
    )

    for r in results:

        lines.append(
            f"{r['variant']:<24}"
            f"{r['params_k']:>12.3f}"
            f"{r['tlc']:>8}"
            f"{r['latency_mean_ms']:>16.4f}"
            f"{r['latency_std_ms']:>12.4f}"
            f"{r['peak_allocated_vram_mb']:>16.2f}"
            f"{r['peak_reserved_vram_mb']:>17.2f}"
        )


    lookup = {
        r["variant"]: r
        for r in results
    }

    p = lookup[
        "Parallel Fixed"
    ]

    s = lookup[
        "Serial"
    ]

    p_tlc = lookup[
        "Parallel Fixed + TLC"
    ]

    s_tlc = lookup[
        "Serial + TLC"
    ]

    lines.extend(
        [
            "",
            (
                "Matched topology "
                "comparisons"
            ),
            "-" * 110,
            (
                "TLC OFF | "
                "Serial - Parallel latency: "
                f"{s['latency_mean_ms'] - p['latency_mean_ms']:+.4f} ms"
            ),
            (
                "TLC OFF | "
                "Serial / Parallel latency ratio: "
                f"{s['latency_mean_ms'] / p['latency_mean_ms']:.4f}"
            ),
            (
                "TLC OFF | "
                "Serial - Parallel peak allocated VRAM: "
                f"{s['peak_allocated_vram_mb'] - p['peak_allocated_vram_mb']:+.2f} MiB"
            ),
            "",
            (
                "TLC ON  | "
                "Serial - Parallel latency: "
                f"{s_tlc['latency_mean_ms'] - p_tlc['latency_mean_ms']:+.4f} ms"
            ),
            (
                "TLC ON  | "
                "Serial / Parallel latency ratio: "
                f"{s_tlc['latency_mean_ms'] / p_tlc['latency_mean_ms']:.4f}"
            ),
            (
                "TLC ON  | "
                "Serial - Parallel peak allocated VRAM: "
                f"{s_tlc['peak_allocated_vram_mb'] - p_tlc['peak_allocated_vram_mb']:+.2f} MiB"
            ),
            "",
            (
                "Important:"
            ),
            (
                "This benchmark uses the "
                "actual frozen network-forward "
                "models and mature 885k "
                "checkpoints. It does not "
                "use the old DummyNetwork "
                "topology benchmark."
            ),
            (
                "FLOPs/MACs are deliberately "
                "not reported here; they will "
                "be audited separately so "
                "attention matmul operations "
                "are not silently omitted."
            ),
        ]
    )

    txt_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)

    print(
        "Saved:",
        csv_path,
    )

    print(
        "Saved:",
        txt_path,
    )

    print("=" * 100)


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
    )

    gpu_name = (
        torch.cuda.get_device_name(
            device
        )
    )


    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    torch.set_float32_matmul_precision(
        "highest"
    )

    print("=" * 100)

    print(
        "FINAL x4 NETWORK-FORWARD "
        "EFFICIENCY BENCHMARK"
    )

    print("=" * 100)

    print(
        "GPU:",
        gpu_name,
    )

    print(
        "Precision: FP32"
    )

    print(
        "TF32: OFF"
    )

    print(
        "LR input:",
        (
            1,
            3,
            LR_HEIGHT,
            LR_WIDTH,
        ),
    )

    print(
        "Expected SR output:",
        EXPECTED_OUTPUT_SHAPE,
    )

    print("=" * 100)

    variants = [
        "Fair RFDN",
        "Parallel Fixed",
        "Serial",
        "Parallel Fixed + TLC",
        "Serial + TLC",
    ]

    results = []

    for variant in variants:

        result = benchmark_one(
            variant,
            device,
        )

        results.append(
            result
        )

    write_results(
        results
    )

    print()
    print(
        "FINAL EFFICIENCY "
        "BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()
