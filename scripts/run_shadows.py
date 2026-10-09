"""Train (or finish training) the shadow bank for a config.

Resumable: shadows already on disk are skipped. Shardable across GPUs:

    python scripts/run_shadows.py --config configs/e5_lira.yaml
    python scripts/run_shadows.py --config configs/e5_lira.yaml --shard 0/2 --set fl.device=cuda:0
    python scripts/run_shadows.py --config configs/e5_lira.yaml --shard 1/2 --set fl.device=cuda:1

The bank lands in results/shadows/<setup_hash>-S<num_shadow>/ and is shared by
every config with the same data, model, FL settings and seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.experiment import prepare, start_run  # noqa: E402
from pof.shadows import bank_dir, train_bank  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--shard", default="0/1", help="k/n: train shadows with index %% n == k")
    args = ap.parse_args(argv)
    k, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= k < n:
        ap.error("--shard must be k/n with 0 <= k < n")

    cfg, _, log, _ = start_run(args.config, args.set, source="scripts/run_shadows.py")
    env = prepare(cfg, log.info)
    result = train_bank(cfg, env["train"], env["part"], env["k"], shard=(k, n), log=log.info)
    log.info(f"shard {k}/{n}: trained {result['trained']}, already present {result['skipped']}")
    print(bank_dir(cfg))
    return result


if __name__ == "__main__":
    main()
