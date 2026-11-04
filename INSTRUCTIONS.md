# Instructions

## 1. Make sure you have Python installed

Run the command in terminal to check your Python version:
```bash
python --version
```

Must be using 3.13+. Recommended version 3.13.13.

## 2. Install all dependencies

Go to the root directory of this project and run
```bash
pip install -r requirements.txt
```

Make sure all libraries are installed. If you are running on Mac you may encounter OpenMP problems when installing XGBoost, you will have to install the library mannually:

```bash
brew install libomp
pip install xgboost
```

`brew` is HomeBrew installer ([website](https://brew.sh/)). If you don't have that installed please visit the website and install HomeBrew by following the instructions.

## 3. Download the data

Data is holded at [this HuggingFace repo](https://huggingface.co/datasets/Baliyang/SSU_data_dash). Download the files and put it in the root like this:
```
SSUDataDash
├──data
│   ├── README.md
│   ├── listings.csv
│   ├── listings.parquet
│   ├── test_list_aware.csv
│   ├── test_list_aware.parquet
│   ├── test_off_market.csv
│   └── test_off_market.parquet
└── run.py (and other files)
```

## 4. Generate the submissions

If you have everything installed, you should be able to train the models and make predictions with

```bash
python run.py predict
```

A run usually takes last a couple minutes depending on the processing power of your machine (there will be a progress bar so you know it's running). The program will consume 1-2 GB of memory. When it finishes, no errors should be raised and a new folder called `submissions/` will appear with the submission files.

Your generated off market submission (`submission_off_market.csv`) should have similar numbers like this:
```csv
ml_num,predicted_price
SYN-0011818,468498.0
SYN-0117584,1.092368e+06
SYN-0065643,625810.0
...
```

And list aware submission (`submission_list_aware.csv`) should look like this:
```csv
ml_num,predicted_price
SYN-0120320,578799.0
SYN-0041629,1339665.0
SYN-0070828,2839574.0
...
```

> Numbers may vary only slightly

### 5. Running individual backtests

You can test the individual backtests with

```bash
python run.py backtest <ModelName> [--ref <ReferenceModelName>]
```

This tests models with a six-month walk-forward backtest setup (aligns with the starter code).

Example results:
```bash
$ python run.py backtest OffMarketBuildingV1 --ref OffMarketBuildingXGBoostV1
Backtesting: 100%|██████████████████████████████████| 6/6 [02:47<00:00, 27.85s/it]
2026-11-03 21:04:51.497 | SUCCESS  | __main__:<module>:157 - Backtest completed for model OffMarketBuildingV1, track: off_market
2026-11-03 21:04:51.497 | INFO     | __main__:<module>:158 - Aggregated metrics
2026-11-03 21:04:51.497 | INFO     | __main__:<module>:159 -   MdAPE=7.1763 (CI: [7.016, 7.380])
2026-11-03 21:04:51.497 | INFO     | __main__:<module>:162 -   MAPE=9.6661 (CI: [9.480, 9.859])
2026-11-03 21:04:51.497 | INFO     | __main__:<module>:165 -   Signed error: median=0.2502, mean=0.8219
2026-11-03 21:04:51.497 | INFO     | __main__:<module>:170 - Running extra comparison with reference model OffMarketBuildingXGBoostV1
Backtesting reference model: 100%|██████████████████████████████████| 6/6 [03:10<00:00, 31.68s/it]
2026-11-03 21:08:02.338 | SUCCESS  | __main__:<module>:180 - Model comparison completed
2026-11-03 21:08:02.338 | INFO     | __main__:<module>:181 - Reference model MdAPE=6.9711 (CI: [6.808, 7.150])
2026-11-03 21:08:02.338 | INFO     | __main__:<module>:183 - Significance test result: delta=0.1748, p=1.09e-09
2026-11-03 21:08:02.338 | INFO     | __main__:<module>:186 - The model is significantly WORSE than the reference model.
```

As you can see, `OffMarketBuildingXGBoostV1` is significantly better, and it reports a median average percentage error (MdAPE) of $6.9711$, with 95% confidence interval of $[6.808, 7.150]$.

Alternatively, use

```bash
python run.py fulltest <ModelName>
```

to initiate a 4-month single round backtest. The results are more representative because it mimics final testing setup.

Specific usage can be found with
```bash
python run.py --help
```
