# HIT140 Assessment 3 Group Project

This project completes Objective 2 using the 104 FIFA World Cup 2026 matches. It contains two related linear regression tasks:

- Model 2.1 predicts the goal difference between the two teams in each match using 104 match rows.
- Model 2.2 predicts the number of goals scored by one team using 208 team-match rows.

Both datasets contain exactly eight explanatory variables. Every predictor is calculated from information available before the match. The final 21 matches are retained as a chronological holdout, while the earlier matches are used for expanding-window model selection.

## Team members and responsibilities

| Team member | Student ID | Main responsibilities | Independent review responsibilities |
|---|---:|---|---|
| Senuja Linal | S404282 | Project coordination; repository and environment setup; end-to-end workflow integration; automated requirement validation; final report integration; reproducibility checks; final output and submission quality assurance | Reviewed dataset invariants, chronological evaluation, reported metrics, figure consistency and final submission readiness |
| Sasindu Ranwadana | S403521 | Exploratory analysis for both tasks; Model 2.2 table validation; target distributions and correlation summaries; interpretation of the team-goals task | Reviewed the processed datasets, Model 2.1 interpretation and the clarity of data visualisations |
| Hansana Edirisinghe Arachchige | S403123 | Chronological modelling framework; implementation and comparison of OLS, Ridge, Lasso, Elastic Net and Huber regression; hyperparameter comparison; Model 2.1 evaluation and model diagnostics | Reviewed Model 2.2 performance, residual interpretation and the consistency of modelling choices across both tasks |
| Lahiru Samarasinghage | S397804 | Source dataset verification; leakage-safe feature construction; 104-row and 208-row dataset generation; source checksum and data audit; Model 2.1 dataset and distribution checks | Reviewed exploratory summaries, row and predictor counts, same-day processing and pre-match leakage controls |



## Shared team work

All four members contributed to the final review. Shared tasks included checking the eight predictors for each objective, confirming that current-match information was excluded, reviewing chronological validation, verifying numerical claims against generated results, and agreeing on the final limitations and recommendations.

The review structure was:

| Work area | First owner | Reviewer |
|---|---|---|
| Source data and leakage-safe dataset construction | Lahiru | Senuja |
| Exploratory analysis and distributions | Sasindu | Lahiru |
| Regression implementation and model comparison | Hansana | Senuja |
| Model 2.1 interpretation | Hansana | Sasindu |
| Model 2.2 interpretation | Sasindu | Hansana |
| Integration, requirement validation and final report | Senuja | All members |

## Method summary

The workflow processes matches in date order and freezes all features before updating results from that date. New teams receive a neutral prior equivalent to three matches. Matches 1 to 83 form the development sample and matches 84 to 104 form the untouched final holdout.

Four expanding chronological folds compare a historical-mean baseline with ordinary least squares, Ridge, Lasso, Elastic Net and Huber regression. Selection is based on mean cross-validation RMSE. Final evaluation reports MAE, RMSE, R squared, prediction bias, bootstrap uncertainty, standardised coefficients, permutation importance and residual diagnostics.

## Project structure

```text
data/raw/             Source World Cup JSON file
data/processed/       Validated Model 2.1 and Model 2.2 tables
src/                  Dataset construction, EDA, modelling and validation
outputs/figures/      Generated distributions and model diagnostic figures
outputs/results/      Generated metrics, diagnostics and validation records
report/               Final four-page group report
run_all.py            Rebuilds and validates the complete workflow
requirements.txt      Required Python packages
```

## Reproducing the analysis

From the project directory, install the listed requirements and run:

```bash
python run_all.py
```

This command rebuilds both modelling tables, repeats the exploratory analysis, compares the regression candidates, evaluates the chronological holdout, regenerates the figures and checks the assignment's required row and predictor counts.

## Main conclusion

Ridge regression was the strongest linear candidate in chronological cross-validation for both tasks. It modestly reduced holdout MAE, particularly for team goals, but did not materially improve RMSE or holdout R squared. The models should therefore be treated as reproducible and interpretable linear benchmarks rather than confident forecasts.
