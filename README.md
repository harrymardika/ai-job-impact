# AI Job Replacement Risk Prediction (2030) with LightGBM

A regression project that tries to predict how likely a job is to be replaced by AI by 2030 (`AI_Replacement_Risk`, 0 to 1) from job, worker, and workplace attributes, using a LightGBM regressor.

It was my final project for the Universitas Gunadarma Data Science course (2 to 13 June 2026), under the **Associate Data Scientist** competency scheme. The notebook (in Indonesian) follows the scheme's units, from data collection to model evaluation.

## Results

| Metric (test set, 600 rows) | Target | Result | Met? |
|---|---|---|---|
| R² | ≥ 0.85 | -0.0062 | No |
| RMSE | ≤ 0.30 | 0.2605 | Yes |
| MAE | ≤ 0.30 | 0.2261 | Yes |

Best hyperparameters from `RandomizedSearchCV` (15 iterations, 3-fold CV): `num_leaves=50`, `n_estimators=100`, `max_depth=3`, `learning_rate=0.01`. The untuned baseline had MAE 0.2336, RMSE 0.2719, and R² -0.0959.

### What the results mean

The MAE and RMSE targets were met, but an R² near zero means the model does no better than always predicting the average risk (about 0.50). On test samples the predictions all fall between 0.50 and 0.54, while the true values range from 0.20 to 0.87.

The data analysis explains why: every numeric feature has a very weak correlation with the target (|r| < 0.04), and Pearson tests found no significant link between risk and years of experience (r = -0.0337, p = 0.065) or salary (r = -0.0067, p = 0.713). Average risk is also almost identical across automation levels (0.49 to 0.51). In this dataset the features carry almost no signal, so the main finding is that **replacement risk cannot be predicted from these attributes**, rather than that the model is ready for use.

## Dataset

- **Source:** [AI Impact in Future on Jobs Market in 2030](https://www.kaggle.com/datasets/muhammadwaqas023/ai-impact-in-future-on-jobs-market-in-2030) (Kaggle), file `AI_Impact_on_Jobs_2030.csv`
- **Size:** 3,000 rows and 20 columns, no missing values or duplicates
- **Target:** `AI_Replacement_Risk` (mean 0.50, std 0.26, range 0.05 to 0.95)
- **Features:** job title, industry, country, education level, years of experience, future demand score, remote work possibility, salary, required skills, automation level, job growth 2030, weekly hours, company size, AI tool usage, upskilling needs, hiring trend, performance score, job satisfaction

## Approach

1. **Business understanding:** goals and success metrics (R² ≥ 0.85, RMSE ≤ 0.30, MAE ≤ 0.30).
2. **Data collection:** download with `kagglehub`.
3. **Data review:** types, statistics, correlation heatmap, target distribution, hypothesis tests (α = 0.05).
4. **Validation:** completeness, IQR outliers, category coverage, 80/20 split (2,400 train / 600 test).
5. **Cleaning:** drop the `Employee_ID` identifier and duplicates.
6. **Feature construction:** Z-score scaling, ordinal mapping, label encoding, and a `Skills_Count` feature from `Required_Skills`.
7. **Modeling:** baseline `LGBMRegressor`, then `RandomizedSearchCV`.
8. **Evaluation:** metrics against targets, predictions on unseen samples, actual vs. predicted plot.

## Tech Stack

Python, Jupyter Notebook, pandas, NumPy, SciPy, LightGBM 4.6.0, scikit-learn 1.8.0, Matplotlib, seaborn, kagglehub.

## Project Structure

```
ai-job-impact/
├── notebook.ipynb   # Analysis and modeling notebook
└── .gitignore       # Ignores datasets/ and documents/
```

The dataset is not stored in the repository; the notebook reads `datasets/AI_Impact_on_Jobs_2030.csv`.

## Getting Started

```bash
git clone https://github.com/harrymardika/ai-job-impact.git
cd ai-job-impact
pip install pandas numpy scipy matplotlib seaborn lightgbm scikit-learn kagglehub jupyter
```

Download the dataset into `datasets/` (or uncomment the `kagglehub` cell in the notebook):

```python
import kagglehub
kagglehub.dataset_download(
    "muhammadwaqas023/ai-impact-in-future-on-jobs-market-in-2030",
    output_dir="./datasets",
)
```

Then run `jupyter notebook notebook.ipynb`.

## Possible Improvements

- Check whether the dataset is synthetic, and look for data with real labels.
- Reframe as classification (low, medium, high risk) and compare against a majority-class baseline.
- Use permutation importance or SHAP to confirm that no feature is informative.

## Author

**Harry Mardika** · [GitHub](https://github.com/harrymardika)
