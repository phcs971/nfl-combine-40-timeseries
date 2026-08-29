"""Build the athlete sampling frame from nflverse combine + draft releases."""

import argparse
import pathlib
import sys

import pandas as pd

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
COMBINE_URL = f"{NFLVERSE}/combine/combine.parquet"
DRAFT_URL = f"{NFLVERSE}/draft_picks/draft_picks.parquet"

CLASS_MAP = {
    "WR": "skill", "CB": "skill", "SAF": "skill",
    "RB": "strong", "TE": "strong", "LB": "strong", "EDGE": "strong",
    "OT": "line", "G": "line", "C": "line", "DT": "line",
}


def _norm(s: pd.Series) -> pd.Series:
    return (
        s.str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.strip()
    )


def build(seasons: list[int]) -> pd.DataFrame:
    combine = pd.read_parquet(COMBINE_URL)
    draft = pd.read_parquet(DRAFT_URL)

    df = combine[combine.season.isin(seasons) & combine.forty.notna()].copy()
    df["cls"] = df.pos.map(CLASS_MAP)
    df = df[df.cls.notna()].copy()

    picks = draft[draft.season.isin(seasons)][
        ["season", "pfr_player_id", "pfr_player_name", "round", "pick", "team"]
    ].copy()

    # The combine release ships draft_round unpopulated for the most recent
    # class, so draft status is rebuilt from draft_picks rather than trusted.
    df = df.merge(
        picks.drop(columns="pfr_player_name"),
        left_on=["season", "pfr_id"],
        right_on=["season", "pfr_player_id"],
        how="left",
    )

    # ~8% of recent-year combine rows carry no pfr_id; recover those by name.
    missing = df["round"].isna()
    if missing.any():
        by_name = picks.assign(key=_norm(picks.pfr_player_name))
        by_name = by_name.drop_duplicates(["season", "key"])
        fill = (
            df.loc[missing, ["season"]]
            .assign(key=_norm(df.loc[missing, "player_name"]))
            .merge(by_name, on=["season", "key"], how="left")
        )
        for col in ("round", "pick", "team"):
            df.loc[missing, col] = fill[col].to_numpy()

    df["drafted"] = df["round"].notna()
    df["forty_tercile"] = df.groupby("cls").forty.transform(
        lambda s: pd.qcut(s, 3, labels=["fast", "mid", "slow"])
    )

    cols = [
        "season", "player_name", "pos", "cls", "school", "ht", "wt",
        "forty", "forty_tercile", "drafted", "round", "pick", "team",
        "pfr_id", "cfb_id",
    ]
    return df[cols].sort_values(["cls", "forty"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--out", default="data/frame.csv")
    args = ap.parse_args()

    df = build(args.seasons)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"{len(df)} athletes -> {out}")
    summary = df.groupby("cls").agg(
        n=("forty", "size"),
        drafted=("drafted", "sum"),
        forty_mean=("forty", "mean"),
        forty_sd=("forty", "std"),
    )
    print(summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
