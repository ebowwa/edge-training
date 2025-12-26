#!/usr/bin/env python
"""
CLI for Storage and Compute Planning.
Run quick estimates without starting the full API server.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from service.planner import plan_training, DatasetSpec, ModelSpec, TrainingSpec, StoragePlanner, ComputePlanner
from integrations.catalog import GPUCatalog, ProviderType


def print_plan(result: dict) -> None:
    """Pretty print planning results."""
    print("\n" + "=" * 60)
    print("         TRAINING RESOURCE PLANNER")
    print("=" * 60)

    print("\n📊 SPECS")
    print("-" * 40)
    for key, val in result["specs"].items():
        print(f"  {key.capitalize()}: {val}")

    print("\n💾 STORAGE REQUIREMENTS")
    print("-" * 40)
    for key, val in result["storage"].items():
        if key != "total_mb":
            label = key.replace("_", " ").capitalize()
            print(f"  {label}: {val}")
    print(f"  {'Total':>20}: {result['storage']['total']}")

    print("\n🖥️  GPU REQUIREMENTS")
    print("-" * 40)
    print(f"  {'GPU':>20}: {result['compute']['recommended_gpu']}")
    print(f"  {'VRAM':>20}: {result['compute']['vram']}")
    print(f"  {'Training time':>20}: {result['compute']['training_hours']}")
    if result["compute"].get("estimated_cost"):
        print(f"  {'Est. cloud cost':>20}: {result['compute']['estimated_cost']}")

    print("\n⚙️  CPU REQUIREMENTS (if no GPU)")
    print("-" * 40)
    if result["compute"].get("recommended_cpu"):
        print(f"  {'CPU':>20}: {result['compute']['recommended_cpu']}")
        print(f"  {'RAM':>20}: {result['compute']['ram']}")
        print(f"  {'Training time':>20}: {result['compute']['cpu_hours']}")

    print("\n" + "=" * 60 + "\n")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plan storage and compute for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick estimate: 10k images, YOLOv8m, 60 epochs
  python cli.py plan --images 10000 --model m --epochs 60

  # High-res training with augmentation
  python cli.py plan --images 5000 --resolution 1280 720 --model l --augment 3

  # Small model, large batch
  python cli.py plan --images 20000 --model n --batch 64

  # Compare GPUs across providers
  python cli.py gpu --min-vram 24 --hours 10

  # Find cheapest GPU from Hetzner
  python cli.py gpu --provider hetzner --min-vram 48
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Plan training resources")

    plan_parser.add_argument(
        "--images", "-i",
        type=int,
        required=True,
        help="Number of training images"
    )
    plan_parser.add_argument(
        "--resolution", "-r",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="Image resolution (default: 640 640)"
    )
    plan_parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="m",
        help="YOLO model variant (default: m)"
    )
    plan_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=60,
        help="Training epochs (default: 60)"
    )
    plan_parser.add_argument(
        "--batch", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    plan_parser.add_argument(
        "--augment", "-a",
        type=int,
        default=1,
        help="Augmentation factor (default: 1)"
    )
    plan_parser.add_argument(
        "--gpu", "-g",
        type=str,
        help="Specific GPU for cost estimation"
    )
    plan_parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["runpod", "hetzner"],
        help="Cloud provider for cost estimation"
    )

    # GPU comparison command
    gpu_parser = subparsers.add_parser("gpu", help="Compare GPU options across providers")

    gpu_parser.add_argument(
        "--min-vram",
        type=float,
        default=0,
        help="Minimum VRAM in GB (default: 0)"
    )
    gpu_parser.add_argument(
        "--max-price",
        type=float,
        help="Maximum hourly price in USD"
    )
    gpu_parser.add_argument(
        "--provider",
        type=str,
        choices=["runpod", "hetzner"],
        help="Filter by provider"
    )
    gpu_parser.add_argument(
        "--hours",
        type=float,
        default=1,
        help="Training hours for cost comparison (default: 1)"
    )

    args = parser.parse_args()

    if args.command == "plan":
        provider = ProviderType(args.provider) if args.provider else None
        result = plan_training(
            num_images=args.images,
            resolution=tuple(args.resolution),
            model_variant=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            augment_factor=args.augment,
            gpu_name=args.gpu,
            provider=provider,
        )
        print_plan(result)

    elif args.command == "gpu":
        catalog = GPUCatalog()
        provider_type = ProviderType(args.provider) if args.provider else None

        print("\n" + "=" * 100)
        print("                 GPU COMPARISON")
        print("=" * 100)

        gpus = catalog.list_gpus(
            provider=provider_type,
            min_vram_gb=args.min_vram,
            max_price_per_hour=args.max_price,
        )

        if not gpus:
            print("No GPUs found matching criteria.")
            return

        # Print header
        print(f"\n  {'GPU':<35} {'VRAM':<8} {'Price/hr':<12} {'Price/{args.hours}h':<15} {'Provider':<10}")
        print("-" * 100)

        # Print sorted by price
        for gpu in sorted(gpus, key=lambda g: g.price_per_hour)[:15]:
            cost = gpu.price_per_hour * args.hours
            monthly = f" (${gpu.price_per_month:.0f}/mo)" if gpu.price_per_month else ""
            print(
                f"  {gpu.gpu_name:<33} {gpu.vram_gb:<8.0f} "
                f"${gpu.price_per_hour:<6.2f}/h{monthly:<6} "
                f"${cost:<10.2f} {gpu.provider.value:<10}"
            )

        print("\n" + "=" * 100 + "\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
