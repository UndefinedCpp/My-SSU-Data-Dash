# Selected model results

Defaults: `OffMarketRefinedV6` and `ListAwareRefinedV6`.

Compared with the original models at commit `e8769e4`, using the same
142,227 training sales before February 2026 and 6,047 February–May evaluation
sales listed on or after the training cutoff:

| Track | Original MdAPE | Selected MdAPE | Relative error reduction |
|---|---:|---:|---:|
| Off-market | 7.4902% | 7.1969% | 3.92% |
| List-aware | 4.1470% | 4.0152% | 3.18% |

MdAPE is median absolute percentage error; lower is better. These periods were
reused for development and selection, so these are not fresh holdout or hidden-test
scores. Later experimental and fine-tuned models did not justify replacing these
selections.

Off-market uses a geometric blend of XGBoost, categorical/text CatBoost, and
frozen MiniLM semantic features. List-aware blends the original building model
with historical TF-IDF and comparable-price features. Settings are in
`models/refined_config.json`; semantic weights use a pinned MiniLM revision.

Run with Python 3.12 and the listing/test data downloaded from
[Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash):

```sh
pip install -r requirements.txt
DATA_DIR=/path/to/dataset python run.py predict
DATA_DIR=/path/to/dataset python run.py check
```

The first semantic-model run downloads the pinned encoder from Hugging Face.
`predict` retrains the selected price models. Datasets, fitted checkpoints,
submissions, and experimental variants are not included in this commit.
