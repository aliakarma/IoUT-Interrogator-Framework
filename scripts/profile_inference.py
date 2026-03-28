"""
Inference Profiler
==================
Measures per-sample inference latency and peak GPU/CPU memory usage for the
TrustTransformer model.

Usage:
    python scripts/profile_inference.py \
        --config model/configs/transformer_config.json \
        --seq-len 64 \
        --batch-size 1 \
        --num-samples 200

Output (printed and written to model/checkpoints/inference_profile.json):
    inference_latency_ms  – median latency per sample in milliseconds
    peak_memory_mb        – peak allocated memory in megabytes
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import ensure_supported_python
from model.inference.transformer_model import TrustTransformer


def profile_inference(
    config_path: str,
    seq_len: int,
    batch_size: int,
    num_samples: int,
    device: str,
) -> dict:
    """
    Run latency and memory profiling for a single batch-size=1 forward pass.

    Args:
        config_path:  path to transformer_config.json
        seq_len:      sequence length in time-steps
        batch_size:   batch size to use for each forward pass (default 1)
        num_samples:  number of forward passes to time
        device:       "cpu" or "cuda"

    Returns:
        dict with inference_latency_ms and peak_memory_mb
    """
    with open(config_path) as f:
        config = json.load(f)

    model = TrustTransformer(config)
    model.eval()
    model.to(device)

    n_features = config["architecture"]["input_dim"]

    # Warm-up passes to avoid cold-start bias
    with torch.no_grad():
        dummy = torch.randn(batch_size, seq_len, n_features, device=device)
        for _ in range(10):
            _ = model(dummy)

    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(batch_size, seq_len, n_features, device=device)

            if device == "cuda":
                torch.cuda.synchronize(device)

            t_start = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize(device)
            t_end = time.perf_counter()

            latencies_ms.append((t_end - t_start) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    median_latency_ms = float(np.median(latencies_arr))
    p95_latency_ms = float(np.percentile(latencies_arr, 95))
    mean_latency_ms = float(np.mean(latencies_arr))

    if device == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
    else:
        try:
            import tracemalloc
            tracemalloc.start()
            with torch.no_grad():
                x = torch.randn(batch_size, seq_len, n_features)
                _ = model(x)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        except Exception:
            peak_bytes = 0

    peak_memory_mb = float(peak_bytes) / (1024 * 1024)

    return {
        "inference_latency_ms": median_latency_ms,
        "inference_latency_mean_ms": mean_latency_ms,
        "inference_latency_p95_ms": p95_latency_ms,
        "peak_memory_mb": peak_memory_mb,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "device": device,
    }


def main():
    ensure_supported_python()

    parser = argparse.ArgumentParser(
        description="Profile TrustTransformer inference latency and memory usage"
    )
    parser.add_argument(
        "--config",
        default="model/configs/transformer_config.json",
        help="Path to transformer_config.json",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length in time-steps (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per forward pass (default: 1)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of forward passes to time (default: 200)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run profiling on (default: cpu)",
    )
    parser.add_argument(
        "--output",
        default="model/checkpoints/inference_profile.json",
        help="Path to write the JSON profile results",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[profile_inference] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    print(f"[profile_inference] Profiling on device={args.device}, "
          f"batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"num_samples={args.num_samples}")

    results = profile_inference(
        config_path=args.config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=args.device,
    )

    print("\n=== Inference Profile Results ===")
    print(f"  inference_latency_ms  (median): {results['inference_latency_ms']:.3f} ms")
    print(f"  inference_latency_ms  (mean):   {results['inference_latency_mean_ms']:.3f} ms")
    print(f"  inference_latency_ms  (p95):    {results['inference_latency_p95_ms']:.3f} ms")
    print(f"  peak_memory_mb:                 {results['peak_memory_mb']:.3f} MB")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
