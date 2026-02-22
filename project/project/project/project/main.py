"""CLI entrypoint for FX log-return anomaly detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_source_yahoo import download_fx_yahoo
from detector import run_detector
from io_utils import read_csv_prices, save_df


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Detect anomalies in Forex log-returns using MAD + EWMA."
    )
    parser.add_argument("--source", choices=["yahoo", "csv"], default="yahoo")

    parser.add_argument("--symbol", default="EURUSD=X")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--period", default="30d")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--input", default=None)

    parser.add_argument("--price_mode", choices=["close", "mid"], default="close")
    parser.add_argument("--output_dir", default="./output")

    parser.add_argument("--mad_window", type=int, default=500)
    parser.add_argument("--mad_threshold", type=float, default=5.0)
    parser.add_argument("--ewma_lambda", type=float, default=0.94)
    parser.add_argument("--ewma_threshold", type=float, default=4.0)
    parser.add_argument("--combine_mode", choices=["AND", "OR"], default="AND")
    parser.add_argument("--spread_window", type=int, default=200)
    parser.add_argument("--spread_multiplier", type=float, default=3.0)
    parser.add_argument("--cooldown_bars", type=int, default=5)
    parser.add_argument("--reversion_window", type=int, default=5)
    parser.add_argument("--reversion_ratio", type=float, default=0.6)
    parser.add_argument("--follow_through_bars", type=int, default=3)
    parser.add_argument("--regime_bars", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser


def main() -> None:
    """Run pipeline from CLI arguments and export outputs."""
    parser = build_parser()
    args = parser.parse_args()

    if args.source == "yahoo":
        period = None if str(args.period).lower() == "none" else args.period
        if period is None and (not args.start or not args.end):
            parser.error("For Yahoo source with period=None, both --start and --end are required.")
        df_raw = download_fx_yahoo(
            symbol=args.symbol,
            interval=args.interval,
            period=period,
            start=args.start,
            end=args.end,
            auto_adjust=False,
        )
    else:
        if not args.input:
            parser.error("--input is required when --source csv")
        df_raw = read_csv_prices(args.input)

    params = {
        "mad_window": args.mad_window,
        "mad_threshold": args.mad_threshold,
        "ewma_lambda": args.ewma_lambda,
        "ewma_threshold": args.ewma_threshold,
        "combine_mode": args.combine_mode,
        "spread_window": args.spread_window,
        "spread_multiplier": args.spread_multiplier,
        "cooldown_bars": args.cooldown_bars,
        "reversion_window": args.reversion_window,
        "reversion_ratio": args.reversion_ratio,
        "follow_through_bars": args.follow_through_bars,
        "regime_bars": args.regime_bars,
        "eps": args.eps,
    }

    df_bars, df_events = run_detector(df_raw=df_raw, price_mode=args.price_mode, params=params)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "yahoo":
        save_df(df_raw, str(output_dir / "raw_prices.csv"))
    save_df(df_bars, str(output_dir / "anomaly_bars.csv"))
    save_df(df_events, str(output_dir / "anomaly_events.csv"))

    print(f"Eventos detectados: {len(df_events)}")
    if not df_events.empty:
        top5 = df_events.sort_values("max_abs_return", ascending=False).head(5)
        print("Top 5 eventos por max_abs_return:")
        for _, row in top5.iterrows():
            print(
                f"- event_id={row['event_id']} start={row['start_time']} "
                f"end={row['end_time']} type={row['anomaly_type']} "
                f"max_abs_return={row['max_abs_return']:.6f}"
            )


if __name__ == "__main__":
    main()
