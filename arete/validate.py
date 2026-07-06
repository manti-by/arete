from __future__ import annotations

import argparse


def cmd_validate(args: argparse.Namespace) -> None:
    from arete.services.validation import validate_dataset

    validate_dataset(data_dir=args.data_dir)
