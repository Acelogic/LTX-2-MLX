#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.20.0",
#     "rich>=13.0.0",
# ]
# ///
"""
Interactive script to download LTX-2 and Gemma weights from Hugging Face.

Usage:
    # With uv (recommended)
    uv run scripts/download_weights.py

    # With python (requires huggingface_hub and rich)
    python scripts/download_weights.py

    # Non-interactive (download specific weights)
    uv run scripts/download_weights.py --weights distilled spatial gemma

    # With HuggingFace token
    uv run scripts/download_weights.py --token YOUR_HF_TOKEN
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import get_token, hf_hub_download, login, snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

# Weight definitions
WEIGHTS = {
    "distilled": {
        "name": "LTX-2 19B Distilled",
        "description": "Fast generation (8 steps), recommended for most users",
        "repo": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-distilled.safetensors",
        "local_path": "weights/ltx-2/ltx-2-19b-distilled.safetensors",
        "size": "~43GB",
        "required": True,
    },
    "distilled-fp8": {
        "name": "LTX-2 19B Distilled (FP8)",
        "description": "Quantized version, smaller download (~27GB)",
        "repo": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-distilled-fp8.safetensors",
        "local_path": "weights/ltx-2/ltx-2-19b-distilled-fp8.safetensors",
        "size": "~27GB",
        "required": False,
    },
    "dev": {
        "name": "LTX-2 19B Dev",
        "description": "Higher quality (25-50 steps), slower generation",
        "repo": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-dev.safetensors",
        "local_path": "weights/ltx-2/ltx-2-19b-dev.safetensors",
        "size": "~43GB",
        "required": False,
    },
    "dev-fp8": {
        "name": "LTX-2 19B Dev (FP8)",
        "description": "Quantized dev version, smaller download",
        "repo": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-dev-fp8.safetensors",
        "local_path": "weights/ltx-2/ltx-2-19b-dev-fp8.safetensors",
        "size": "~27GB",
        "required": False,
    },
    "spatial": {
        "name": "Spatial Upscaler 2x",
        "description": "2x resolution upscaling (256->512, 512->1024)",
        "repo": "Lightricks/LTX-2",
        "filename": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "local_path": "weights/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "size": "~995MB",
        "required": False,
    },
    "temporal": {
        "name": "Temporal Upscaler 2x",
        "description": "2x framerate upscaling (17->33 frames, etc.)",
        "repo": "Lightricks/LTX-2",
        "filename": "ltx-2-temporal-upscaler-x2-1.0.safetensors",
        "local_path": "weights/ltx-2/ltx-2-temporal-upscaler-x2-1.0.safetensors",
        "size": "~262MB",
        "required": False,
    },
    "distilled-lora": {
        "name": "Distilled LoRA",
        "description": "LoRA for Stage 2 refinement in two-stage pipeline",
        "repo": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-distilled-lora-384.safetensors",
        "local_path": "weights/ltx-2/ltx-2-19b-distilled-lora-384.safetensors",
        "size": "~1.5GB",
        "required": False,
    },
    "gemma": {
        "name": "Gemma 3 12B Text Encoder",
        "description": "Required for text-to-video generation",
        "repo": "google/gemma-3-12b-it",
        "filename": None,  # Full repo download
        "local_path": "weights/gemma-3-12b",
        "size": "~25GB",
        "required": True,
        "needs_license": True,
    },
}


def print_header():
    """Print welcome header."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]LTX-2 MLX Weight Downloader[/bold blue]\n"
        "Download model weights from Hugging Face",
        border_style="blue"
    ))
    console.print()


def print_weights_table(selected: Optional[set[str]] = None):
    """Print table of available weights."""
    table = Table(title="Available Weights", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="yellow")
    table.add_column("Name", style="white")
    table.add_column("Size", style="green")
    table.add_column("Description", style="dim")
    table.add_column("Status", style="magenta")

    for key, info in WEIGHTS.items():
        local_path = Path(info["local_path"])

        if local_path.exists() or (local_path.is_dir() and any(local_path.iterdir())):
            status = "[green]Downloaded[/green]"
        elif selected and key in selected:
            status = "[yellow]Selected[/yellow]"
        elif info.get("required"):
            status = "[red]Required[/red]"
        else:
            status = "[dim]Optional[/dim]"

        table.add_row(
            key,
            info["name"],
            info["size"],
            info["description"],
            status
        )

    console.print(table)
    console.print()


def get_interactive_selection() -> set[str]:
    """Interactively select weights to download."""
    selected = set()

    # Check what's already downloaded
    already_downloaded = set()
    for key, info in WEIGHTS.items():
        local_path = Path(info["local_path"])
        if local_path.exists() or (local_path.is_dir() and any(local_path.iterdir())):
            already_downloaded.add(key)

    if already_downloaded:
        console.print(f"[green]Already downloaded:[/green] {', '.join(already_downloaded)}")
        console.print()

    # Quick selection options
    console.print("[bold]Quick selection:[/bold]")
    console.print("  [yellow]1[/yellow] - Essential (distilled + gemma) - Recommended for getting started")
    console.print("  [yellow]2[/yellow] - Full (all weights) - Everything including upscalers")
    console.print("  [yellow]3[/yellow] - Custom - Choose individual weights")
    console.print()

    choice = Prompt.ask(
        "Select option",
        choices=["1", "2", "3"],
        default="1"
    )

    if choice == "1":
        selected = {"distilled", "gemma"}
    elif choice == "2":
        selected = set(WEIGHTS.keys())
    else:
        # Custom selection
        console.print("\n[bold]Select weights to download:[/bold]")
        for key, info in WEIGHTS.items():
            if key in already_downloaded:
                continue

            default = info.get("required", False)
            if Confirm.ask(
                f"  Download [yellow]{key}[/yellow] ({info['name']}, {info['size']})?",
                default=default
            ):
                selected.add(key)

    # Remove already downloaded
    selected -= already_downloaded

    return selected


def download_weight(key: str, info: dict, token: Optional[str] = None) -> bool:
    """Download a single weight file."""
    local_path = Path(info["local_path"])

    # Create parent directory
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if info.get("filename") is None:
            # Full repo download (for Gemma)
            console.print(f"  Downloading full repository...")
            snapshot_download(
                repo_id=info["repo"],
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "original/*"],
                token=token,
            )
        else:
            # Single file download
            console.print(f"  Downloading {info['filename']}...")
            hf_hub_download(
                repo_id=info["repo"],
                filename=info["filename"],
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False,
                token=token,
            )

            # Rename to expected local name if different
            downloaded_path = local_path.parent / info["filename"]
            if downloaded_path.exists() and downloaded_path != local_path:
                downloaded_path.rename(local_path)

        return True

    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return False


def download_weights(selected: set[str], token: Optional[str] = None):
    """Download selected weights."""
    if not selected:
        console.print("[yellow]No weights selected for download.[/yellow]")
        return

    console.print()
    console.print(Panel(f"Downloading {len(selected)} weight(s)...", style="blue"))
    console.print()

    # Check for Gemma license requirement
    if "gemma" in selected:
        console.print("[yellow]Note:[/yellow] Gemma 3 requires accepting the license at:")
        console.print("  https://huggingface.co/google/gemma-3-12b-it")
        console.print()
        if not token:
            # Check for stored token first
            stored_token = get_token()
            if stored_token:
                console.print("[green]Using stored HuggingFace credentials.[/green]")
                token = stored_token
            elif os.isatty(0):  # Only prompt if running interactively
                console.print("[yellow]You may need a HuggingFace token for Gemma.[/yellow]")
                token = Prompt.ask("Enter HuggingFace token (or press Enter to skip)", default="")
                if token:
                    login(token=token)
            console.print()

    success = []
    failed = []

    for key in selected:
        info = WEIGHTS[key]
        console.print(f"[bold cyan]{info['name']}[/bold cyan] ({info['size']})")

        if download_weight(key, info, token):
            success.append(key)
            console.print(f"  [green]Success![/green] Saved to {info['local_path']}")
        else:
            failed.append(key)

        console.print()

    # Summary
    console.print(Panel.fit(
        f"[green]Downloaded: {len(success)}[/green]  "
        f"[red]Failed: {len(failed)}[/red]",
        title="Summary",
        border_style="green" if not failed else "yellow"
    ))

    if success:
        console.print()
        console.print("[bold]You can now run:[/bold]")
        console.print("  uv run python scripts/generate.py \"Your prompt here\"")


def main():
    parser = argparse.ArgumentParser(
        description="Download LTX-2 and Gemma weights from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (recommended)
    uv run scripts/download_weights.py

    # Download specific weights
    uv run scripts/download_weights.py --weights distilled gemma

    # Download all weights
    uv run scripts/download_weights.py --weights all

    # With HuggingFace token
    uv run scripts/download_weights.py --token YOUR_TOKEN
        """
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        choices=list(WEIGHTS.keys()) + ["all", "essential"],
        help="Weights to download (default: interactive)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available weights and exit"
    )
    args = parser.parse_args()

    # Handle token
    token = args.token or os.environ.get("HF_TOKEN")

    print_header()

    if args.list:
        print_weights_table()
        return

    # Determine selection
    if args.weights:
        if "all" in args.weights:
            selected = set(WEIGHTS.keys())
        elif "essential" in args.weights:
            selected = {"distilled", "gemma"}
        else:
            selected = set(args.weights)

        # Remove already downloaded
        already_downloaded = set()
        for key in selected:
            info = WEIGHTS[key]
            local_path = Path(info["local_path"])
            if local_path.exists() or (local_path.is_dir() and any(local_path.iterdir())):
                already_downloaded.add(key)

        if already_downloaded:
            console.print(f"[dim]Skipping already downloaded: {', '.join(already_downloaded)}[/dim]")

        selected -= already_downloaded
    else:
        # Interactive mode
        print_weights_table()
        selected = get_interactive_selection()

    if selected:
        download_weights(selected, token)
    else:
        console.print("[green]All selected weights are already downloaded![/green]")


if __name__ == "__main__":
    main()
