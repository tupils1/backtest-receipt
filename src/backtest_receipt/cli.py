"""Command line interface.

``check`` exits 0 when a receipt is internally consistent and says pass.
It exits 1 when the receipt is a consistent fail or blocked result.
It exits 2 when the file is unreadable or disagrees with itself.
``check --study`` also recomputes the receipt from the study directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backtest_receipt.errors import BacktestReceiptError
from backtest_receipt.prereg import load_preregistration
from backtest_receipt.receipt import validate_receipt
from backtest_receipt.study import Study
from backtest_receipt.version import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backtest-receipt")
    parser.add_argument("--version", action="version", version=f"backtest-receipt {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="exit nonzero unless the receipt passed")
    check.add_argument("receipt")
    check.add_argument(
        "--study", help="recompute the receipt from this study directory and compare fingerprints"
    )
    check.set_defaults(func=_check)

    digest = sub.add_parser("hash", help="print the content hash of a pre-registration YAML file")
    digest.add_argument("prereg")
    digest.set_defaults(func=_hash)

    issue = sub.add_parser("issue", help="write receipt.json and receipt.md for a study")
    issue.add_argument("study")
    issue.add_argument("--json", dest="json_path")
    issue.add_argument("--markdown", dest="markdown_path")
    issue.set_defaults(func=_issue)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (BacktestReceiptError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _check(args: argparse.Namespace) -> int:
    document = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    validate_receipt(document)
    if args.study:
        fresh = Study.open(args.study).issue_receipt(write=False)
        if fresh["fingerprint"] != document["fingerprint"]:
            print(
                f"FAIL {document.get('study_id', '')} receipt does not match the study",
                file=sys.stderr,
            )
            return 1
    if document["passed"]:
        print(f"PASS {document['study_id']}")
        return 0
    names = [gate["name"] for gate in document["gates"] if gate["state"] != "pass"]
    print(f"{str(document['status']).upper()} {document['study_id']} {', '.join(names)}")
    return 1


def _hash(args: argparse.Namespace) -> int:
    spec = load_preregistration(args.prereg)
    print(spec.content_hash())
    return 0


def _issue(args: argparse.Namespace) -> int:
    study = Study.open(args.study)
    receipt = study.issue_receipt(json_path=args.json_path, markdown_path=args.markdown_path)
    print(receipt["status"])
    return 0
