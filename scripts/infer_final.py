from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer_final.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--safe_mode", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "scripts.infer_rag",
        "--config",
        args.config,
        "--model_type",
        "dpo_rag",
        "--query",
        args.query,
    ]

    if args.top_k is not None:
        cmd.extend(["--top_k", str(args.top_k)])

    if args.safe_mode:
        cmd.append("--safe_mode")

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)

    print(result.stdout)


if __name__ == "__main__":
    main()