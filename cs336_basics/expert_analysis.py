"""
Expert Load Balancing Analysis
===============================
Instruments MoE routing layers to measure expert utilization and detect
load imbalance. Works with any MoE variant (Einsum, Triton, GroupedMM)
via forward hooks — no changes to existing model code required.

Metrics computed:
  - Per-expert token counts and utilization ratio
  - Load balance ratio (max / ideal)
  - Routing entropy vs ideal entropy
  - Gini coefficient (inequality measure)
  - Dead expert detection (below threshold)
  - Per-expert average gating weight

Usage:
  python -m cs336_basics.expert_analysis \
      --checkpoint ./checkpoints/moe/ckpt_step_10000.pt \
      --data_dir ./data/valid-bin/ \
      --model_type moe_triton \
      --num_batches 50 \
      --device cuda
"""

import os
import math
import json
import argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from cs336_basics.nn import GatingFineGrainedMoE, TritonGroupedGEMMMoE

# Try importing TorchGroupedMMMoE (requires torch >= 2.8)
try:
    from cs336_basics.nn import TorchGroupedMMMoE
    _HAS_GROUPED_MM = True
except (ImportError, RuntimeError):
    _HAS_GROUPED_MM = False

MOE_CLASSES = (GatingFineGrainedMoE, TritonGroupedGEMMMoE)
if _HAS_GROUPED_MM:
    MOE_CLASSES = (*MOE_CLASSES, TorchGroupedMMMoE)


# ═══════════════════════════════════════════════════════════════════
# Routing Statistics Collector
# ═══════════════════════════════════════════════════════════════════

class RoutingStatsCollector:
    """
    Attaches forward hooks to every MoE layer in a model to capture
    expert selection indices and gating weights without modifying
    the model code.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hooks = []
        self.stats: dict[str, dict] = {}  # layer_name -> accumulated stats
        self._attach_hooks()

    def _attach_hooks(self):
        """Find all MoE modules and register forward hooks."""
        for name, module in self.model.named_modules():
            if isinstance(module, MOE_CLASSES):
                hook = module.register_forward_hook(
                    self._make_hook(name, module.num_experts, module.num_experts_per_tok)
                )
                self.hooks.append(hook)
                self.stats[name] = {
                    "num_experts": module.num_experts,
                    "num_experts_per_tok": module.num_experts_per_tok,
                    "expert_counts": torch.zeros(module.num_experts, dtype=torch.int64),
                    "expert_weight_sums": torch.zeros(module.num_experts, dtype=torch.float64),
                    "total_tokens": 0,
                    "total_assignments": 0,  # = total_tokens * K
                }

    def _make_hook(self, layer_name: str, num_experts: int, num_experts_per_tok: int):
        """Create a hook closure that captures routing decisions."""

        def hook_fn(module, input, output):
            with torch.no_grad():
                x = input[0]
                x_flat = x.view(-1, x.size(-1))
                T = x_flat.size(0)

                # Re-run gating (lightweight — just a linear + topk)
                gate_logits = module.gate(x_flat)                        # [T, E]
                top_k_logits, top_k_indices = torch.topk(
                    gate_logits, num_experts_per_tok, dim=-1
                )                                                        # [T, K]
                top_k_weights = top_k_logits.softmax(dim=-1)             # [T, K]

                # Accumulate counts
                stats = self.stats[layer_name]
                flat_indices = top_k_indices.view(-1).cpu()
                flat_weights = top_k_weights.view(-1).cpu().to(torch.float64)

                # Expert counts
                counts = torch.zeros(num_experts, dtype=torch.int64)
                counts.scatter_add_(0, flat_indices.long(), torch.ones_like(flat_indices, dtype=torch.int64))
                stats["expert_counts"] += counts

                # Expert weight sums (for average gating weight per expert)
                weight_sums = torch.zeros(num_experts, dtype=torch.float64)
                weight_sums.scatter_add_(0, flat_indices.long(), flat_weights)
                stats["expert_weight_sums"] += weight_sums

                stats["total_tokens"] += T
                stats["total_assignments"] += T * num_experts_per_tok

        return hook_fn

    def reset(self):
        """Clear all accumulated statistics."""
        for name, stats in self.stats.items():
            E = stats["num_experts"]
            stats["expert_counts"] = torch.zeros(E, dtype=torch.int64)
            stats["expert_weight_sums"] = torch.zeros(E, dtype=torch.float64)
            stats["total_tokens"] = 0
            stats["total_assignments"] = 0

    def remove_hooks(self):
        """Detach all hooks from the model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# ═══════════════════════════════════════════════════════════════════
# Metrics Computation
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(stats: dict) -> dict:
    """
    Compute load balancing metrics from accumulated routing statistics.

    Returns a dict with:
      - expert_utilization: fraction of assignments each expert received
      - load_balance_ratio: max_utilization / ideal_utilization (1.0 = perfect)
      - routing_entropy: entropy of the utilization distribution
      - ideal_entropy: log2(E), entropy of uniform distribution
      - entropy_ratio: routing_entropy / ideal_entropy
      - gini_coefficient: 0 = perfect equality, 1 = total inequality
      - dead_experts: list of expert indices with < 1% of ideal load
      - num_dead_experts: count of dead experts
      - avg_gating_weight_per_expert: mean gating weight when expert is selected
    """
    E = stats["num_experts"]
    K = stats["num_experts_per_tok"]
    counts = stats["expert_counts"].float().numpy()
    weight_sums = stats["expert_weight_sums"].numpy()
    total_assignments = stats["total_assignments"]

    if total_assignments == 0:
        return {"error": "No data collected"}

    # ── Utilization (fraction of total assignments) ────────────
    utilization = counts / total_assignments          # sums to 1.0
    ideal_utilization = 1.0 / E                       # if perfectly balanced

    # ── Load Balance Ratio ────────────────────────────────────
    load_balance_ratio = utilization.max() / ideal_utilization

    # ── Routing Entropy ───────────────────────────────────────
    # Higher = more uniform; max = log2(E)
    eps = 1e-10
    p = utilization + eps
    p = p / p.sum()  # re-normalize after eps
    routing_entropy = -np.sum(p * np.log2(p))
    ideal_entropy = np.log2(E)
    entropy_ratio = routing_entropy / ideal_entropy if ideal_entropy > 0 else 0.0

    # ── Gini Coefficient ──────────────────────────────────────
    # Classic Gini: measures inequality. 0 = perfect equality.
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n

    # ── Dead Experts (< 1% of ideal load) ─────────────────────
    ideal_count = total_assignments / E
    dead_threshold = 0.01 * ideal_count
    dead_experts = [int(i) for i in range(E) if counts[i] < dead_threshold]

    # ── Average gating weight per expert ──────────────────────
    avg_weight = np.where(counts > 0, weight_sums / counts, 0.0)

    return {
        "num_experts": E,
        "num_experts_per_tok": K,
        "total_tokens": stats["total_tokens"],
        "total_assignments": total_assignments,
        "expert_counts": counts.tolist(),
        "expert_utilization": utilization.tolist(),
        "load_balance_ratio": float(load_balance_ratio),
        "routing_entropy": float(routing_entropy),
        "ideal_entropy": float(ideal_entropy),
        "entropy_ratio": float(entropy_ratio),
        "gini_coefficient": float(gini),
        "dead_experts": dead_experts,
        "num_dead_experts": len(dead_experts),
        "avg_gating_weight_per_expert": avg_weight.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# Pretty Printing
# ═══════════════════════════════════════════════════════════════════

def print_layer_report(layer_name: str, metrics: dict):
    """Print a formatted report for one MoE layer."""
    E = metrics["num_experts"]
    K = metrics["num_experts_per_tok"]
    print(f"\n{'═' * 70}")
    print(f"  Layer: {layer_name}")
    print(f"  Experts: {E}  |  Top-K: {K}  |  Tokens processed: {metrics['total_tokens']:,}")
    print(f"{'═' * 70}")

    # Summary metrics
    print(f"\n  {'Metric':<30} {'Value':>12}  {'Assessment':>15}")
    print(f"  {'─' * 60}")

    # Load Balance Ratio (1.0 = perfect, >2.0 = severely imbalanced)
    lbr = metrics["load_balance_ratio"]
    lbr_status = "✅ Balanced" if lbr < 1.5 else ("⚠️  Moderate" if lbr < 2.5 else "🔴 Severe")
    print(f"  {'Load Balance Ratio':<30} {lbr:>12.3f}  {lbr_status:>15}")

    # Entropy Ratio (1.0 = perfectly uniform)
    er = metrics["entropy_ratio"]
    er_status = "✅ Uniform" if er > 0.95 else ("⚠️  Skewed" if er > 0.85 else "🔴 Collapsed")
    print(f"  {'Entropy Ratio':<30} {er:>12.3f}  {er_status:>15}")

    # Gini Coefficient (0 = equal, >0.3 = high inequality)
    gini = metrics["gini_coefficient"]
    gini_status = "✅ Equal" if gini < 0.1 else ("⚠️  Unequal" if gini < 0.3 else "🔴 Extreme")
    print(f"  {'Gini Coefficient':<30} {gini:>12.3f}  {gini_status:>15}")

    # Dead experts
    nd = metrics["num_dead_experts"]
    nd_status = "✅ All Active" if nd == 0 else f"🔴 {nd}/{E} Dead"
    print(f"  {'Dead Experts (<1% ideal)':<30} {nd:>12d}  {nd_status:>15}")

    # Per-expert breakdown
    counts = metrics["expert_counts"]
    utils = metrics["expert_utilization"]
    weights = metrics["avg_gating_weight_per_expert"]
    ideal = 1.0 / E

    print(f"\n  {'Expert':<8} {'Count':>10} {'Util%':>8} {'vs Ideal':>10} {'Avg Weight':>12}  {'Bar'}")
    print(f"  {'─' * 70}")

    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(E):
        bar_len = int(40 * counts[i] / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        ratio = utils[i] / ideal if ideal > 0 else 0
        marker = " 💀" if i in metrics["dead_experts"] else ""
        print(f"  E{i:<5d} {counts[i]:>10,.0f} {utils[i]*100:>7.2f}% {ratio:>9.2f}x {weights[i]:>11.4f}  {bar}{marker}")

    print()


def print_summary(all_metrics: dict[str, dict]):
    """Print an overall summary across all layers."""
    print(f"\n{'═' * 70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'═' * 70}")

    ginis = [m["gini_coefficient"] for m in all_metrics.values()]
    entropies = [m["entropy_ratio"] for m in all_metrics.values()]
    lbrs = [m["load_balance_ratio"] for m in all_metrics.values()]
    dead = [m["num_dead_experts"] for m in all_metrics.values()]

    print(f"\n  {'Metric':<35} {'Min':>8} {'Mean':>8} {'Max':>8}")
    print(f"  {'─' * 60}")
    print(f"  {'Load Balance Ratio':<35} {min(lbrs):>8.3f} {np.mean(lbrs):>8.3f} {max(lbrs):>8.3f}")
    print(f"  {'Entropy Ratio':<35} {min(entropies):>8.3f} {np.mean(entropies):>8.3f} {max(entropies):>8.3f}")
    print(f"  {'Gini Coefficient':<35} {min(ginis):>8.3f} {np.mean(ginis):>8.3f} {max(ginis):>8.3f}")
    print(f"  {'Dead Experts':<35} {min(dead):>8d} {np.mean(dead):>8.1f} {max(dead):>8d}")

    total_dead = sum(dead)
    total_experts = sum(m["num_experts"] for m in all_metrics.values())
    print(f"\n  Total dead experts across all layers: {total_dead}/{total_experts}")

    if np.mean(ginis) > 0.2:
        print(f"\n  ⚠️  RECOMMENDATION: High Gini coefficient detected across layers.")
        print(f"     Consider adding a load-balancing auxiliary loss to encourage")
        print(f"     uniform expert utilization (e.g., Switch Transformer's aux loss).")
    elif np.mean(entropies) < 0.9:
        print(f"\n  ⚠️  RECOMMENDATION: Routing entropy is below ideal.")
        print(f"     Some experts are being favored. An auxiliary loss or")
        print(f"     expert capacity factor may help distribute load.")
    else:
        print(f"\n  ✅ Expert load distribution looks reasonable.")

    print(f"{'═' * 70}\n")


# ═══════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════

def build_model(args):
    """Build the model matching the checkpoint architecture."""
    from cs336_basics.model import TransformerLM_MoE, TransformerLM_TritonMoE

    model_kwargs = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
    )

    if args.model_type == "moe_einsum":
        model = TransformerLM_MoE(**model_kwargs)
    elif args.model_type == "moe_triton":
        model = TransformerLM_TritonMoE(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model_type for expert analysis: {args.model_type}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE expert load balancing")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--model_type", type=str, default="moe_triton", choices=["moe_einsum", "moe_triton"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches to analyze")
    parser.add_argument("--batch_size", type=int, default=8)

    # Model architecture (must match checkpoint)
    parser.add_argument("--vocab_size", type=int, default=None, help="Auto-detected from data if None")
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--num_experts_per_tok", type=int, default=4)

    parser.add_argument("--output_json", type=str, default=None, help="Save metrics to JSON file")

    args = parser.parse_args()

    # ── Load Data ─────────────────────────────────────────────
    from cs336_basics.data import ShardedDataset

    dataset = ShardedDataset(args.data_dir, args.context_length)
    if args.vocab_size is None:
        args.vocab_size = dataset.meta["vocab_size"]
        print(f"Auto-detected vocab_size={args.vocab_size} from data metadata.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ── Build & Load Model ────────────────────────────────────
    print(f"Building {args.model_type} model...")
    model = build_model(args)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.to(args.device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters\n")

    # ── Collect Routing Stats ─────────────────────────────────
    collector = RoutingStatsCollector(model)
    print(f"Hooked {len(collector.stats)} MoE layers.")
    print(f"Running {args.num_batches} batches (batch_size={args.batch_size})...\n")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=args.num_batches, desc="Collecting routing stats")):
            if i >= args.num_batches:
                break
            input_ids = batch["input_ids"].to(args.device)
            model(input_ids)

    # ── Compute & Print Metrics ───────────────────────────────
    all_metrics = {}
    for layer_name, stats in collector.stats.items():
        metrics = compute_metrics(stats)
        all_metrics[layer_name] = metrics
        print_layer_report(layer_name, metrics)

    print_summary(all_metrics)

    # ── Save to JSON ──────────────────────────────────────────
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Metrics saved to {args.output_json}")

    collector.remove_hooks()


if __name__ == "__main__":
    main()
