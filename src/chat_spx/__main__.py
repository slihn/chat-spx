"""Command-line wrapper for price and range memory inference."""

from __future__ import annotations

import argparse
import json

from .core import get_px, get_px_by_date_range


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render close prices.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--price", metavar="DATE", help="Print one close price for DATE.")
    group.add_argument(
        "-r",
        "--range",
        nargs=2,
        metavar=("START_DATE", "END_DATE"),
        dest="date_range",
        help="Print close prices for trading dates in the inclusive range.",
    )
    parser.add_argument("date", nargs="?", help="Trading date shorthand for --price.")
    args = parser.parse_args(argv)

    if args.date_range is not None:
        if args.date is not None:
            parser.error("positional DATE cannot be used with --range")
        print(json.dumps(get_px_by_date_range(*args.date_range)))
        return

    date = args.price or args.date
    if date is None:
        parser.error("provide DATE, --price DATE, or --range START_DATE END_DATE")
    print(get_px(date))


if __name__ == "__main__":
    main()
