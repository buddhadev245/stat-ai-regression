# StatAI — Statistical & AI Regression Analyzer

A public-facing Streamlit application for comparing statistical regression, machine-learning regression, and a cross-fitted residual hybrid model.

## Features

- CSV and Excel upload
- Numeric and categorical predictors
- Automatic numeric-like conversion
- Missing predictor-value imputation inside modelling pipelines
- Linear, polynomial, Ridge, Lasso, Elastic Net
- Random Forest and Gradient Boosting
- Linear + AI residual hybrid using out-of-fold residuals
- Test-set RMSE, MAE and R²
- OLS coefficients, confidence intervals, ANOVA, VIF and Breusch-Pagan diagnostics when available
- Prediction table and plots
- CSV and Excel downloads

## Local run

```bash
python -m pip install -r requirements.txt
python -m streamlit run stat_ai_app.py
```

## Public deployment

This project is structured for Streamlit Community Cloud. Put `stat_ai_app.py` and `requirements.txt` in the root of a GitHub repository, then deploy the app from Streamlit Community Cloud.

Do not commit confidential datasets, credentials, API keys, or secrets.

## Research note

The hybrid model is a methodological implementation for prediction comparison. The application does not claim that hybridisation always improves performance. Results should be interpreted using the unseen test-set metrics, validation design, diagnostics, and domain knowledge.
