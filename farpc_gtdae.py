import argparse
import os
import pandas as pd

from src.merrec_preprocess import load_merrec_events, compute_dns_and_sessions
from src.proxy_analysis import proxy_counterfactual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing MerRec parquet files")
    parser.add_argument("--out-dir", default="outputs_merrec")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--sample-frac", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--user-col", default=None)
    parser.add_argument("--item-col", default=None)
    parser.add_argument("--category-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--event-col", default=None)
    parser.add_argument(
        "--purchase-values",
        default="purchase,buy,order,transaction",
        help="Comma-separated event values treated as purchase",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    purchase_values = [x.strip() for x in args.purchase_values.split(",") if x.strip()]

    df = load_merrec_events(
        data_dir=args.data_dir,
        max_files=args.max_files,
        sample_frac=args.sample_frac,
        user_col=args.user_col,
        item_col=args.item_col,
        category_col=args.category_col,
        time_col=args.time_col,
        event_col=args.event_col,
        purchase_values=purchase_values,
    )

    print(f"Loaded events: {len(df):,}")
    users_dns, sessions, summary = compute_dns_and_sessions(
        df,
        top_k=args.top_k,
        out_dir=args.out_dir,
    )

    print("\nProxy summary:")
    print(summary)

    out, meta = proxy_counterfactual(
        users_dns,
        sessions,
        summary,
        out_dir=args.out_dir,
    )

    print("\nProxy counterfactual:")
    print(out)
    print("\nThreshold meta:")
    print(meta)
    print(f"\nSaved MerRec proxy outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
