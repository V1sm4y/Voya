"""Shared CLI arguments. Every pipeline script must use `base_parser()` so it accepts --sample dev|full."""
import argparse


def base_parser(description: str, split_default: str = "train") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--sample", choices=["dev", "full"], required=True,
                   help="dev = ~5%% of S1 ids + all S2/S3 of the same countries; full = everything")
    p.add_argument("--split", choices=["train", "test"], default=split_default)
    return p
