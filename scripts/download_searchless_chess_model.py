#!/usr/bin/env python3
"""Download a Searchless Chess checkpoint from Hugging Face."""

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="dbest-isi/searchless-chess-9M-selfplay",
        help="Hugging Face repository containing the checkpoint.",
    )
    parser.add_argument(
        "--local-dir",
        default="searchless_chess_model",
        help="Directory in which to store the downloaded files.",
    )
    args = parser.parse_args()

    model_path = snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir)
    print(f"Model downloaded to: {model_path}")


if __name__ == "__main__":
    main()
