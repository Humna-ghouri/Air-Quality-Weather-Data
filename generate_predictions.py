"""
Generate prediction CSV for next N days: loads trained model and daily data, writes predictions.
Run after train.py (e.g. python train.py --save-dir . && python generate_predictions.py --save-dir . --n-days 3).
"""
import argparse
from pathlib import Path
from predict import run_forecast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default=".", help="Directory with model.joblib, scaler.joblib, daily_aggregated.csv")
    parser.add_argument("--n-days", type=int, default=7, help="Forecast horizon (days)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV filename")
    args = parser.parse_args()
    run_forecast(save_dir=args.save_dir, n_days=args.n_days, output_csv=args.output)
