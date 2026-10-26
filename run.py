"""
Entry point for the SSU Data Dash submission.

Required sub-commands:
- python run.py predict
  Predict using the final, trained model(s) and generate two submission files.
- python run.py check
  Check the two submission files for format and consistency.

Custom sub-commands:
- python run.py backtest [model_name] [track] [cutoff]
  Backtest the specified model on the given track and cutoff date.
- python run.py clean
  Clean up generated backtest files.
- python run.py benchmark [track]
  Read all predictions for the specified track and generate a benchmark report.
"""
import argparse
import check_submissions
import harness
import os
import model_hub
from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm


# Set up argument parser and subparsers
parser = argparse.ArgumentParser(
    description='SSU Data Dash submission and development tool'
)
subparsers = parser.add_subparsers(required=True, dest='subcommand')
predict_parser = subparsers.add_parser(
    'predict',
    help='Predict using the final, trained model(s) and generate two submission files.'
)
check_parser = subparsers.add_parser(
    'check',
    help='Check the two submission files for format and consistency.'
)
backtest_parser = subparsers.add_parser(
    'backtest',
    help='Backtest the specified model on the given track and cutoff date.'
)
backtest_parser.add_argument(
    'model_name',
    help='Name of the model to backtest',
)
backtest_parser.add_argument(
    '-t', '--track',
    help='Track to evaluate the model on',
    default='all'
)
backtest_parser.add_argument(
    '-c', '--cutoff',
    help='Cutoff date for the backtest in YYYY-MM format',
    required=True
)
clean_parser = subparsers.add_parser(
    'clean',
    help='Clean up generated backtest files.'
)
benchmark_parser = subparsers.add_parser(
    'benchmark',
    help='Read all predictions for the specified track and generate a benchmark report.'
)
benchmark_parser.add_argument('track', help='Track to generate benchmark for')

# !!! Final submission models !!!
# TODO: UPDATE THE FINAL SUBMISSION MODELS BEFORE FINAL SUBMISSION
SUBMISSION_MODELS = {
    'off-market': 'NearestSalesBaseline',
    'list-aware': 'ListPriceBaseline'
}


# Entry point for the script
if __name__ == '__main__':
    args = parser.parse_args()
    subcommand = args.subcommand
    if subcommand == 'predict':
        os.makedirs('submissions', exist_ok=True)
        # Generate predictions for the off-market track and write to submission file
        harness.predict_and_write(
            model_hub.load_model(SUBMISSION_MODELS['off-market']),
            'off-market',
            Path('submissions', 'submission_off_market.csv')
        )
        harness.predict_and_write(
            model_hub.load_model(SUBMISSION_MODELS['list-aware']),
            'list-aware',
            Path('submissions', 'submission_list_aware.csv')
        )

    elif subcommand == 'check':
        check_submissions.check_all()

    elif subcommand == 'clean':
        import shutil
        if os.path.exists('backtest'):
            shutil.rmtree('backtest')

    elif subcommand == 'backtest':
        model = model_hub.load_model(args.model_name)
        track = args.track
        if track == 'all':
            tracks = ['off-market', 'list-aware']
        else:
            tracks = [track]

        date_cutoff = args.cutoff
        if date_cutoff == 'all':
            date_cutoff = pd.period_range('2025-12', '2026-05', freq='M')
            single_pass = False
        else:
            date_cutoff = [pd.Period(date_cutoff, freq='M')]
            single_pass = True

        # Separate handling for single pass vs multiple date cutoffs
        if single_pass:
            for t in tracks:
                harness.backtest(model, date_cutoff[0], t, verbose=True, save_results=True)
        else:
            for t in tracks:
                experiment_dataframes = []
                for dc in tqdm(date_cutoff, desc=f'Backtesting'):
                    r = harness.backtest(model, dc, t, verbose=False, save_results=False)
                    experiment_dataframes.append(r)
                # Create pooled statistics
                pooled_df = pd.concat(experiment_dataframes, ignore_index=True)
                metrics = harness.report_metrics(pooled_df['pred'], pooled_df['target'])
                # Report pooled metrics
                logger.success(f'Backtest completed for model {model.get_id()}, track: {t}')
                logger.info(f'Aggregated metrics')
                logger.info(f'  MdAPE={metrics["MdAPE"]:.4f} (CI: [{metrics["MdAPE_95CI"][0]:.3f}, {metrics["MdAPE_95CI"][1]:.3f}])')
                logger.info(f'  MAPE={metrics["MAPE"]:.4f} (CI: [{metrics["MAPE_95CI"][0]:.3f}, {metrics["MAPE_95CI"][1]:.3f}])')
                logger.info(f'  Signed error: median={metrics["median_signed_err"]:.4f}, mean={metrics["mean_signed_err"]:.4f}')



    elif subcommand == 'benchmark':
        ...