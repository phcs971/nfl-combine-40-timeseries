"""Pick the runs to label by hand: 20 athletes per class, split train/test.

Selection spreads over the class's 40-time distribution rather than taking the
fastest, and keeps one run per athlete so no athlete sits in both splits.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from build_frame import _norm

CLASSES = ("skill", "strong", "line")


def usable(runs: pd.DataFrame) -> pd.DataFrame:
    r = runs[runs.complete & ~runs.duplicate & runs.official_forty.notna()].copy()
    r["key"] = _norm(r.player_name)
    return r.sort_values("official_forty").drop_duplicates("key", keep="first")


def pick(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """One athlete from each of n quantile bins of official_forty."""
    df = df.sort_values("official_forty").reset_index(drop=True)
    edges = np.linspace(0, len(df), n + 1).astype(int)
    out = []
    seen_video: dict[str, int] = {}
    for a, b in zip(edges[:-1], edges[1:]):
        band = df.iloc[a:max(b, a + 1)]
        # inside a band, prefer the least-used video and then a drafted/undrafted
        # alternation, so neither the corpus nor the draft split is lopsided
        want = not (len(out) % 2)
        band = band.assign(
            _v=band.video_id.map(lambda v: seen_video.get(v, 0)),
            _d=(band.drafted != want).astype(int),
        ).sort_values(["_v", "_d"])
        row = band.iloc[0]
        seen_video[row.video_id] = seen_video.get(row.video_id, 0) + 1
        out.append(row.drop(labels=["_v", "_d"]))
    return pd.DataFrame(out)


def balance(sel: pd.DataFrame) -> pd.DataFrame:
    """Even out drafted counts between the splits by swapping neighbouring pairs,
    which keeps each split's 40-time spread intact."""
    for _ in range(len(sel)):
        n = sel.groupby("split").drafted.sum()
        if abs(n.get("train", 0) - n.get("test", 0)) <= 1:
            break
        heavy = "train" if n.get("train", 0) > n.get("test", 0) else "test"
        inner = sel.index[1:-1]
        a = sel.loc[inner][(sel.loc[inner].split == heavy) & sel.loc[inner].drafted]
        b = sel.loc[inner][(sel.loc[inner].split != heavy) & ~sel.loc[inner].drafted]
        if a.empty or b.empty:
            break
        i, j = a.index[0], b.index[0]
        sel.loc[i, "split"], sel.loc[j, "split"] = sel.loc[j, "split"], heavy
    return sel


def build(runs: pd.DataFrame, frame: pd.DataFrame, fits: pd.DataFrame,
          per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = usable(runs)
    f = frame.assign(key=_norm(frame.player_name))[["key", "pos", "drafted", "forty"]]
    r = r.merge(f, on="key", how="left")
    r["drafted"] = r.drafted.fillna(False).astype(bool)

    q = fits[fits.quality == "ok"][["video_id", "bib", "v_max_yd_s", "tau_s",
                                    "mat_offset"]]
    r = r.merge(q, on=["video_id", "bib"], how="left")

    out = []
    for cls in CLASSES:
        sel = pick(r[r.cls == cls], per_class, rng).sort_values("official_forty")
        sel["split"] = ["train" if i % 2 == 0 else "test" for i in range(len(sel))]
        sel = balance(sel)
        out.append(sel)
    cols = ["video_id", "cls", "pos", "split", "bib", "player_name", "official_forty",
            "drafted", "t_zero", "final_clock", "v_max_yd_s", "tau_s", "mat_offset"]
    return pd.concat(out)[cols].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/label_plan.csv")
    a = ap.parse_args()

    plan = build(pd.read_csv("data/runs.csv"), pd.read_csv("data/frame.csv"),
                 pd.read_csv("data/fits.csv"), a.per_class, a.seed)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(a.out, index=False)

    print(f"{len(plan)} runs -> {a.out}")
    print(plan.groupby(["cls", "split"]).agg(
        n=("bib", "size"), drafted=("drafted", "sum"),
        forty=("official_forty", lambda s: f"{s.min():.2f}-{s.max():.2f}"),
        seeded=("v_max_yd_s", lambda s: int(s.notna().sum()))).to_string())
    print("\nby video:")
    print(plan.groupby(["cls", "video_id"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
