"""Write a random dummy submission for the TEST set and run the organisers' validator on it.

This proves the writer's output format end to end. It is not a real prediction. With
--sample dev, only the test-dev S1 ids get random links, but the file still has a row for every
test S1, because the scorer requires one.

Usage (from src/):  python -m ber.make_dummy --sample dev [--check-ids]
"""
import numpy as np

from . import paths
from .cli import base_parser
from .data import load
from .submission import run_validator, write_submission


def main():
    p = base_parser(__doc__, split_default="test")
    p.add_argument("--check-ids", action="store_true", help="also verify ids exist (uses a few GB)")
    p.add_argument("--out", default=str(paths.WORK_DIR / "dummy"))
    args = p.parse_args()
    if args.split != "test":
        raise SystemExit("dummy submissions are for the test split")

    required = list(load("test", "full").s1.entity_id)   # submission must cover every test S1
    b = load("test", args.sample)
    rng = np.random.default_rng(0)
    tg = b.targets
    matches, cands = {}, {}
    for s1, country in zip(b.s1.entity_id, b.s1.country):
        r = rng.random()
        if r < 0.5:
            continue                                       # empty row
        pool = tg.entity_id.values[rng.integers(0, len(tg), 4)]
        cands[s1] = list(dict.fromkeys(pool))
        if r > 0.6:
            matches[s1] = cands[s1][: rng.integers(1, 3)]
    stats = write_submission(args.out, matches, cands, required)
    print(stats)
    code, log = run_validator(args.out, paths.DATA_DIR / "test", paths.VALIDATOR, check_ids=args.check_ids)
    print(log)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
