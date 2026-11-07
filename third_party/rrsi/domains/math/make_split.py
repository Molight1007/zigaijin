"""Freeze the evolve / held-out split for the math domain.

Rules (deliberate, recorded in split_meta.json so the split can be audited):

1. Stratify by `domain`. A missing domain becomes its own stratum
   `(unlabeled)`; a compound label such as "离散数学/高等代数" is keyed by its
   first part.
2. Within a stratum of n >= 4, take `max(1, round(0.25 * n))` tasks into
   held-out. Smaller strata go entirely into evolve — a 1-task held-out stratum
   cannot support any statistic.
3. Fixed seed. Re-running the script reproduces the same split; changing the
   seed invalidates every earlier comparison, so do not.

Output: data/evolve_ids.txt, data/heldout_ids.txt, data/split_meta.json.
"""

from __future__ import annotations

import collections
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261027
HELDOUT_FRAC = 0.25
MIN_STRATUM_FOR_HELDOUT = 4


def bank_path() -> Path:
    env = os.environ.get("MATHPILOT_BANK")
    if env:
        return Path(env)
    root = os.environ.get("MATHPILOT_ROOT")
    base = Path(root) if root else (HERE.parent.parent / "third_party" / "mathpilot")
    return base / "题库" / "official112_本地测试题库" / "official112_full.jsonl"


def stratum_of(rec: dict) -> str:
    label = (rec.get("domain") or "").strip()
    if not label:
        return "(unlabeled)"
    return label.split("/")[0].strip() or "(unlabeled)"


def main() -> int:
    p = bank_path()
    if not p.exists():
        print(f"[FAIL] bank not found: {p}")
        return 1
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    by: dict[str, list[str]] = collections.defaultdict(list)
    for r in recs:
        by[stratum_of(r)].append(str(r.get("id")))

    rng = random.Random(SEED)
    evolve, heldout, meta = [], [], {}
    for name in sorted(by):
        ids = sorted(by[name])
        rng.shuffle(ids)
        if len(ids) >= MIN_STRATUM_FOR_HELDOUT:
            n_hold = max(1, round(HELDOUT_FRAC * len(ids)))
        else:
            n_hold = 0
        heldout += ids[:n_hold]
        evolve += ids[n_hold:]
        meta[name] = {"n": len(ids), "heldout": n_hold, "evolve": len(ids) - n_hold}

    evolve, heldout = sorted(evolve), sorted(heldout)
    if set(evolve) & set(heldout):
        print("[FAIL] overlap between evolve and heldout")
        return 1
    if len(evolve) + len(heldout) != len(recs):
        print(f"[FAIL] union {len(evolve) + len(heldout)} != bank {len(recs)}")
        return 1

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "evolve_ids.txt").write_text("\n".join(evolve) + "\n", encoding="utf-8")
    (DATA / "heldout_ids.txt").write_text("\n".join(heldout) + "\n", encoding="utf-8")
    (DATA / "smoke_ids.txt").write_text("official112-019\nofficial112-092\n", encoding="utf-8")
    (DATA / "split_meta.json").write_text(json.dumps({
        "bank": str(p), "seed": SEED, "heldout_frac": HELDOUT_FRAC,
        "min_stratum_for_heldout": MIN_STRATUM_FOR_HELDOUT,
        "n_total": len(recs), "n_evolve": len(evolve), "n_heldout": len(heldout),
        "strata": meta,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"bank      {p.name}: {len(recs)} 题")
    print(f"evolve    {len(evolve)} 题")
    print(f"heldout   {len(heldout)} 题")
    print("分层:")
    for k, v in sorted(meta.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {k:16} n={v['n']:3}  evolve={v['evolve']:3}  heldout={v['heldout']:2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
