"""
MoE Implementation Benchmark
=============================
Compares three MoE implementations across training and inference:
  1. Einsum (GatingFineGrainedMoE)       — batched matmul via einops
  2. Triton (TritonGroupedGEMMMoE)       — custom Triton grouped-GEMM kernel
  3. GroupedMM (TorchGroupedMMMoE)        — torch._grouped_mm (PyTorch 2.6+)

Metrics collected:
  - Training: step time, loss curve, peak GPU memory, throughput (tokens/sec)
  - Inference: latency per batch, throughput, peak memory

Usage:
  python -m cs336_basics.evaluate_moe [--device cuda] [--num_steps 50] [--batch_size 8]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, field

import torch

from cs336_basics.model import (
    TransformerLM_MoE,
    TransformerLM_TritonMoE,
    TransformerLM_GroupedMMMoE,
)
from cs336_basics.loss import crossEntropyLoss


# ─── Configuration ────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    # Model
    vocab_size: int = 512
    context_length: int = 128
    num_layers: int = 2
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 512
    rope_theta: float = 10000.0
    num_experts: int = 4
    num_experts_per_tok: int = 2
    # Training
    num_train_steps: int = 50
    batch_size: int = 8
    lr: float = 3e-4
    # Inference
    num_inference_steps: int = 100
    # Device
    device: str = "cuda"


# ─── Metrics Collector ────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    num_params: int = 0
    # Training
    train_step_times: list[float] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    train_peak_memory_mb: float = 0.0
    # Inference
    inference_times: list[float] = field(default_factory=list)
    inference_peak_memory_mb: float = 0.0

    @property
    def avg_train_step_ms(self) -> float:
        if not self.train_step_times:
            return 0.0
        # Skip first 5 steps (warmup)
        times = self.train_step_times[5:] if len(self.train_step_times) > 5 else self.train_step_times
        return sum(times) / len(times) * 1000

    @property
    def avg_inference_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        times = self.inference_times[5:] if len(self.inference_times) > 5 else self.inference_times
        return sum(times) / len(times) * 1000

    @property
    def train_tokens_per_sec(self) -> float:
        if self.avg_train_step_ms == 0:
            return 0.0
        return 0  # Will be computed externally

    @property
    def final_loss(self) -> float:
        return self.train_losses[-1] if self.train_losses else float("nan")


# ─── Utility ──────────────────────────────────────────────────────

def reset_memory_stats(device: torch.device):
    """Reset GPU memory tracking."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def get_peak_memory_mb(device: torch.device) -> float:
    """Get peak GPU memory in MB."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def generate_batch(cfg: BenchmarkConfig, device: torch.device):
    """Generate a random batch of token IDs."""
    input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
    return input_ids, targets


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─── Benchmark Routines ──────────────────────────────────────────

def benchmark_training(
    model: torch.nn.Module,
    cfg: BenchmarkConfig,
    device: torch.device,
    result: BenchmarkResult,
):
    """Run a short training loop and collect metrics."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    reset_memory_stats(device)

    for step in range(cfg.num_train_steps):
        input_ids, targets = generate_batch(cfg, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # Forward
        logits = model(input_ids)
        loss = crossEntropyLoss(logits[:, :-1, :], targets[:, 1:])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        result.train_step_times.append(t1 - t0)
        result.train_losses.append(loss.item())

        if step % 10 == 0:
            print(f"  [{result.name}] step {step:3d}/{cfg.num_train_steps}  loss={loss.item():.4f}  time={((t1-t0)*1000):.1f}ms")

    result.train_peak_memory_mb = get_peak_memory_mb(device)


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    cfg: BenchmarkConfig,
    device: torch.device,
    result: BenchmarkResult,
):
    """Run inference-only forward passes and collect metrics."""
    model.eval()
    reset_memory_stats(device)

    for step in range(cfg.num_inference_steps):
        input_ids, _ = generate_batch(cfg, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        _ = model(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        result.inference_times.append(t1 - t0)

    result.inference_peak_memory_mb = get_peak_memory_mb(device)


# ─── Model Factory ────────────────────────────────────────────────

MODEL_REGISTRY = {
    "Einsum MoE": TransformerLM_MoE,
    "Triton MoE": TransformerLM_TritonMoE,
    "GroupedMM MoE": TransformerLM_GroupedMMMoE,
}


def build_model(name: str, cls, cfg: BenchmarkConfig, device: torch.device) -> torch.nn.Module | None:
    """Try to build a model, returning None if the backend is unavailable."""
    try:
        model = cls(
            vocab_size=cfg.vocab_size,
            context_length=cfg.context_length,
            num_layers=cfg.num_layers,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            rope_theta=cfg.rope_theta,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            device=device,
        )
        return model
    except (ImportError, RuntimeError) as e:
        print(f"  ⚠ Skipping {name}: {e}")
        return None


# ─── Report ───────────────────────────────────────────────────────

def print_report(results: list[BenchmarkResult], cfg: BenchmarkConfig):
    """Print a formatted comparison table."""
    tokens_per_step = cfg.batch_size * (cfg.context_length - 1)

    sep = "─" * 90
    print(f"\n{'═' * 90}")
    print(f"  MoE BENCHMARK RESULTS")
    print(f"  Config: {cfg.num_layers}L / {cfg.d_model}d / {cfg.d_ff}ff / "
          f"{cfg.num_experts}E top-{cfg.num_experts_per_tok} / "
          f"batch={cfg.batch_size} / seq={cfg.context_length}")
    print(f"{'═' * 90}\n")

    # Header
    print(f"{'Metric':<30} ", end="")
    for r in results:
        print(f"{r.name:>18} ", end="")
    print()
    print(sep)

    # Parameters
    print(f"{'Parameters':30s} ", end="")
    for r in results:
        print(f"{r.num_params:>18,} ", end="")
    print()
    print(sep)

    # Training
    print(f"\n{'TRAINING':30s} ({cfg.num_train_steps} steps)")
    print(sep)

    print(f"{'Avg step time (ms)':30s} ", end="")
    for r in results:
        print(f"{r.avg_train_step_ms:>18.2f} ", end="")
    print()

    print(f"{'Final loss':30s} ", end="")
    for r in results:
        print(f"{r.final_loss:>18.4f} ", end="")
    print()

    print(f"{'Peak memory (MB)':30s} ", end="")
    for r in results:
        print(f"{r.train_peak_memory_mb:>18.1f} ", end="")
    print()

    print(f"{'Throughput (tok/s)':30s} ", end="")
    for r in results:
        tps = tokens_per_step / (r.avg_train_step_ms / 1000) if r.avg_train_step_ms > 0 else 0
        print(f"{tps:>18,.0f} ", end="")
    print()

    # Inference
    print(f"\n{'INFERENCE':30s} ({cfg.num_inference_steps} steps)")
    print(sep)

    print(f"{'Avg latency (ms)':30s} ", end="")
    for r in results:
        print(f"{r.avg_inference_ms:>18.2f} ", end="")
    print()

    print(f"{'Peak memory (MB)':30s} ", end="")
    for r in results:
        print(f"{r.inference_peak_memory_mb:>18.1f} ", end="")
    print()

    print(f"{'Throughput (tok/s)':30s} ", end="")
    for r in results:
        tps = (cfg.batch_size * cfg.context_length) / (r.avg_inference_ms / 1000) if r.avg_inference_ms > 0 else 0
        print(f"{tps:>18,.0f} ", end="")
    print()

    # Speedup relative to first
    if len(results) > 1 and results[0].avg_train_step_ms > 0:
        print(f"\n{'RELATIVE SPEEDUP (vs Einsum)':30s}")
        print(sep)
        base_train = results[0].avg_train_step_ms
        base_infer = results[0].avg_inference_ms

        print(f"{'Training speedup':30s} ", end="")
        for r in results:
            speedup = base_train / r.avg_train_step_ms if r.avg_train_step_ms > 0 else 0
            print(f"{speedup:>17.2f}x ", end="")
        print()

        print(f"{'Inference speedup':30s} ", end="")
        for r in results:
            speedup = base_infer / r.avg_inference_ms if r.avg_inference_ms > 0 else 0
            print(f"{speedup:>17.2f}x ", end="")
        print()

    print(f"\n{'═' * 90}\n")


def save_results(results: list[BenchmarkResult], cfg: BenchmarkConfig, path: str = "moe_benchmark_results.json"):
    """Save results to JSON for later analysis."""
    data = {
        "config": {
            "vocab_size": cfg.vocab_size,
            "context_length": cfg.context_length,
            "num_layers": cfg.num_layers,
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "d_ff": cfg.d_ff,
            "num_experts": cfg.num_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "batch_size": cfg.batch_size,
            "num_train_steps": cfg.num_train_steps,
            "num_inference_steps": cfg.num_inference_steps,
        },
        "results": [],
    }
    for r in results:
        data["results"].append({
            "name": r.name,
            "num_params": r.num_params,
            "avg_train_step_ms": r.avg_train_step_ms,
            "final_loss": r.final_loss,
            "train_peak_memory_mb": r.train_peak_memory_mb,
            "avg_inference_ms": r.avg_inference_ms,
            "inference_peak_memory_mb": r.inference_peak_memory_mb,
            "train_losses": r.train_losses,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoE Implementation Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_train_steps", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--save_results", type=str, default="moe_benchmark_results.json")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        num_layers=args.num_layers,
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
    )

    device = torch.device(cfg.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print()

    results: list[BenchmarkResult] = []

    for name, model_cls in MODEL_REGISTRY.items():
        print(f"{'─' * 50}")
        print(f"Benchmarking: {name}")
        print(f"{'─' * 50}")

        model = build_model(name, model_cls, cfg, device)
        if model is None:
            continue

        result = BenchmarkResult(name=name)
        result.num_params = count_parameters(model)
        print(f"  Parameters: {result.num_params:,}")

        # Training benchmark
        print(f"\n  ▶ Training ({cfg.num_train_steps} steps)...")
        benchmark_training(model, cfg, device, result)
        print(f"    Peak memory: {result.train_peak_memory_mb:.1f} MB")
        print(f"    Avg step:    {result.avg_train_step_ms:.2f} ms")

        # Inference benchmark
        print(f"\n  ▶ Inference ({cfg.num_inference_steps} steps)...")
        benchmark_inference(model, cfg, device, result)
        print(f"    Peak memory: {result.inference_peak_memory_mb:.1f} MB")
        print(f"    Avg latency: {result.avg_inference_ms:.2f} ms")

        results.append(result)

        # Free model before next run
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print()

    # Final report
    print_report(results, cfg)
    save_results(results, cfg, args.save_results)


if __name__ == "__main__":
    main()
