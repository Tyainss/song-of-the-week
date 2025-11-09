
from pathlib import Path
import argparse

from core.datasets.build_training_set import build_unified_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    build_unified_dataset(repo_root=args.repo_root)


if __name__ == "__main__":
    main()
