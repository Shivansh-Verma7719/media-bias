"""
Interactive Gold Set Annotation Tool.

Shows article titles one at a time with the full annotation rubric.
Records labels to a CSV. Supports two-annotator mode for computing
Cohen's kappa (inter-annotator agreement).

Usage:
  # First annotator
  python 02_annotate_gold.py -i unlabeled.csv -o gold_annotator1.csv --annotator your_name

  # Second annotator (same input, different output)
  python 02_annotate_gold.py -i unlabeled.csv -o gold_annotator2.csv --annotator second_name

  # After both annotators finish, compute agreement:
  python 02_annotate_gold.py --iaa gold_annotator1.csv gold_annotator2.csv
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

RUBRIC = """
╔══════════════════════════════════════════════════════════════════╗
║                    ANNOTATION RUBRIC                           ║
╠══════════════════════════════════════════════════════════════════╣
║  RELEVANT (y) — ALL THREE must be true:                        ║
║  1. The named company is the PRIMARY subject of the article    ║
║  2. The article covers a MATERIAL corporate event:             ║
║       • Earnings, revenue, guidance, dividends, buybacks       ║
║       • M&A, major contracts, partnerships, divestitures       ║
║       • Regulatory/legal actions against the company itself    ║
║       • Executive appointments/departures (C-suite, board)     ║
║       • Major layoffs, restructuring, plant closures           ║
║       • Major product launches or discontinuations             ║
║       • Labor strikes with operational impact                  ║
║       • Credit ratings, debt issuance, bankruptcy              ║
║  3. It is NOT any of the following:                            ║
║       • Individual employee/driver/customer incident or crime  ║
║       • Entertainment, lifestyle, or consumer content          ║
║       • Unconfirmed rumour or product leak                     ║
║       • Consumer shopping guide or price comparison            ║
║       • Macro/political article where company is one example   ║
║       • Analyst prediction (vs. company as subject)            ║
║       • Social media trend with no documented business impact  ║
║       • Minor app feature or UI update                         ║
╠══════════════════════════════════════════════════════════════════╣
║  When in doubt → IRRELEVANT (n)    Skip if truly unclear → (s) ║
╚══════════════════════════════════════════════════════════════════╝
"""


def compute_kappa(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Compute Cohen's kappa between two annotators on shared IDs."""
    merged = df1[["id", "label"]].merge(df2[["id", "label"]], on="id", suffixes=("_1", "_2"))
    if merged.empty:
        print("No shared IDs between the two files.")
        return float("nan")

    a1 = merged["label_1"].map({"relevant": 1, "irrelevant": 0})
    a2 = merged["label_2"].map({"relevant": 1, "irrelevant": 0})

    n = len(merged)
    p_o = (a1 == a2).sum() / n  # observed agreement

    p1_rel = a1.mean()
    p2_rel = a2.mean()
    p_e = p1_rel * p2_rel + (1 - p1_rel) * (1 - p2_rel)  # expected agreement

    if p_e == 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1 - p_e)

    disagreements = merged[a1 != a2]
    print(f"\nIAA Report")
    print(f"  Shared articles:    {n}")
    print(f"  Agreed:             {(a1 == a2).sum()} ({100*p_o:.1f}%)")
    print(f"  Disagreed:          {len(disagreements)}")
    print(f"  Cohen's kappa (κ):  {kappa:.3f}")
    if kappa >= 0.8:
        print("  Quality:            EXCELLENT (κ ≥ 0.8) ✓")
    elif kappa >= 0.7:
        print("  Quality:            GOOD (κ ≥ 0.7) ✓ — publishable")
    elif kappa >= 0.6:
        print("  Quality:            MODERATE (κ ≥ 0.6) — resolve disagreements")
    else:
        print("  Quality:            POOR (κ < 0.6) — rubric needs clarification")

    if len(disagreements) > 0:
        print(f"\n  Sample disagreements (annotator1 / annotator2):")
        for _, row in disagreements.head(10).iterrows():
            title = df1[df1["id"] == row["id"]]["title"].values
            title_str = title[0] if len(title) > 0 else "?"
            print(f"    [{row['label_1']} / {row['label_2']}] {str(title_str)[:80]}")

    return kappa


def annotate(args):
    df_in = pd.read_csv(args.input, encoding="utf-8")

    # Load existing progress
    if os.path.exists(args.output):
        done_df = pd.read_csv(args.output, encoding="utf-8")
        done_ids = set(done_df["id"].astype(str))
        print(f"Resuming: {len(done_ids)} already labeled, {len(df_in) - len(done_ids)} remaining.")
    else:
        done_df = pd.DataFrame()
        done_ids = set()

    pending = df_in[~df_in["id"].astype(str).isin(done_ids)].reset_index(drop=True)

    if pending.empty:
        print("All articles already labeled.")
        return

    print(RUBRIC)
    print(f"Annotator: {args.annotator}")
    print(f"Controls:  y=relevant  n=irrelevant  s=skip  q=quit\n")

    new_labels = []
    try:
        for idx, row in pending.iterrows():
            company = str(row.get("company_name", "?"))
            title   = str(row.get("title", "?"))
            art_id  = str(row["id"])

            remaining = len(pending) - idx
            print(f"\n[{idx+1}/{len(pending)}] Company: {company}")
            print(f"Title:   {title}")

            while True:
                choice = input("Label (y/n/s/q): ").strip().lower()
                if choice in ("y", "n", "s", "q"):
                    break
                print("  Enter y, n, s, or q")

            if choice == "q":
                print("Quitting. Progress saved.")
                break
            if choice == "s":
                continue

            label = "relevant" if choice == "y" else "irrelevant"
            new_labels.append({
                "id":           art_id,
                "title":        title,
                "company_name": company,
                "label":        label,
                "annotator":    args.annotator,
                "labeled_at":   datetime.now().isoformat(),
                "source":       "gold_manual",
            })

            # Save after every label so progress is never lost
            new_df = pd.DataFrame(new_labels)
            combined = pd.concat([done_df, new_df], ignore_index=True)
            combined.to_csv(args.output, index=False)

    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")

    final = pd.read_csv(args.output)
    print(f"\nDone. Total labeled: {len(final)}")
    print(f"  Relevant:   {(final['label']=='relevant').sum()}")
    print(f"  Irrelevant: {(final['label']=='irrelevant').sum()}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    # Annotate mode
    ann = sub.add_parser("annotate", help="Label articles interactively")
    ann.add_argument("--input",      "-i", required=True, help="CSV of articles to label")
    ann.add_argument("--output",     "-o", required=True, help="Output gold CSV")
    ann.add_argument("--annotator",  "-a", required=True, help="Your name/ID")

    # IAA mode
    iaa = sub.add_parser("iaa", help="Compute inter-annotator agreement")
    iaa.add_argument("files", nargs=2, help="Two annotator CSV files")

    # Default to annotate if old-style args passed
    args, remaining = parser.parse_known_args()
    if args.mode is None:
        # Backwards compat: parse flat args
        parser2 = argparse.ArgumentParser()
        parser2.add_argument("--input",      "-i", required=False)
        parser2.add_argument("--output",     "-o", required=False)
        parser2.add_argument("--annotator",  "-a", default="annotator1")
        parser2.add_argument("--iaa",             nargs=2, metavar="FILE")
        args2 = parser2.parse_args()
        if args2.iaa:
            df1 = pd.read_csv(args2.iaa[0])
            df2 = pd.read_csv(args2.iaa[1])
            compute_kappa(df1, df2)
        else:
            annotate(args2)
        return

    if args.mode == "iaa":
        df1 = pd.read_csv(args.files[0])
        df2 = pd.read_csv(args.files[1])
        compute_kappa(df1, df2)
    else:
        annotate(args)


if __name__ == "__main__":
    main()
