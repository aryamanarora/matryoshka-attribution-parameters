"""``python -m mask_learning_finetuning <config.yaml>`` -- run one experiment.

Configs are files, not flags, so what ran is reproducible from one artifact. The resolved
config (after any ``extends:`` chain) is written to ``<output>/config.yaml``.

    uv run python -m mask_learning_finetuning configs/french_lr1e-4.yaml
    uv run python -m mask_learning_finetuning configs/french_lr1e-4.yaml --print-config
"""

import argparse
import logging
import sys

from .config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="mask_learning_finetuning", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="path to a YAML experiment config")
    p.add_argument("--print-config", action="store_true",
                   help="resolve the extends chain, validate, print, and exit without training")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.print_config:
        import yaml

        from .config import to_dict
        print(yaml.safe_dump(to_dict(cfg), sort_keys=False))
        return 0

    from .train import train
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
