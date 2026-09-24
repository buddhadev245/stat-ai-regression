import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="StatAI — Statistical & AI Regression Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;}
.subtitle {font-size: 1.05rem; color: #666; margin-bottom: 1rem;}
.step-card {padding: 1rem; border: 1px solid #ddd; border-radius: 12px; margin-bottom: .75rem;}
.small-note {font-size: .9rem; color: #666;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 StatAI — Statistical & AI Regression Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload your dataset, select an outcome and predictors, '
    'then compare statistical, AI and hybrid regression models.</div>',
    unsafe_allow_html=True
)

def read_uploaded_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError("Please upload a CSV or Excel file.")

def coerce_numeric_like(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            converted = pd.to_numeric(
                out[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            )
            non_missing = out[col].notna().sum()
            if non_missing and converted.notna().sum() / non_missing >= 0.85:
                out[col] = converted
    return out

def clean_target(df, target):
    y = pd.to_numeric(
        df[target].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )
    return y

def infer_columns(df):
    numeric, categorical = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            converted = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            )
            non_missing = df[c].notna().sum()
            if non_missing and converted.notna().sum() / non_missing >= 0.85:
                numeric.append(c)
            else:
                categorical.append(c)
    return numeric, categorical

def metric_values(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R²": float(r2_score(y_true, y_pred)),
    }

def make_preprocessor(X):
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_cols
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def make_pipeline_model(model, X):
    return Pipeline([
        ("preprocess", make_preprocessor(X)),
        ("model", model),
    ])

def fit_hybrid_oof(base_estimator, residual_estimator, X_train, y_train,
                   X_test, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    Xtr = X_train.reset_index(drop=True)
    ytr = y_train.reset_index(drop=True)
    oof_residuals = np.zeros(len(Xtr))

    for train_idx, valid_idx in kf.split(Xtr):
        base_fold = clone(base_estimator)
        base_fold.fit(Xtr.iloc[train_idx], ytr.iloc[train_idx])
        pred_valid = base_fold.predict(Xtr.iloc[valid_idx])
        oof_residuals[valid_idx] = ytr.iloc[valid_idx].to_numpy() - pred_valid

    residual_model = clone(residual_estimator)
    residual_model.fit(Xtr, oof_residuals)

    base_final = clone(base_estimator)
    base_final.fit(X_train, y_train)

    base_test_pred = base_final.predict(X_test)
    residual_test_pred = residual_model.predict(X_test)
    hybrid_pred = base_test_pred + residual_test_pred

    return base_final, residual_model, hybrid_pred, residual_test_pred, oof_residuals

def make_excel(comparison, predictions, coefficients, vif, diagnostics):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="Model Comparison", index=False)
        predictions.to_excel(writer, sheet_name="Test Predictions", index=False)
        coefficients.to_excel(writer, sheet_name="OLS Coefficients", index=False)
        diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)
        if not vif.empty:
            vif.to_excel(writer, sheet_name="VIF", index=False)
    output.seek(0)
    return output

def safe_numeric_series(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )

with st.sidebar:
    st.header("⚙️ Analysis settings")
    test_size = st.slider("Test-set proportion", 0.15, 0.40, 0.25, 0.05)
    random_state = st.number_input("Random seed", 1, 9999, 42, 1)
    n_splits = st.slider("Hybrid cross-validation folds", 3, 10, 5, 1)
    rf_trees = st.slider("Random Forest trees", 100, 800, 300, 50)
    gb_trees = st.slider("Gradient Boosting trees", 50, 500, 200, 25)
    poly_degree = st.selectbox("Polynomial degree", [2, 3], index=0)

    st.divider()
    st.markdown("**Privacy note**")
    st.caption(
        "Uploaded data are processed for the current analysis session. "
        "Do not upload confidential or personally identifiable data unless you are "
        "comfortable with the hosting environment."
    )

st.subheader("1️⃣ Upload your data")
uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
    help="Your file should contain one outcome column and one or more predictor columns."
)

if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file to begin.")
    st.markdown("""
### How it works
1. **Upload** your dataset.
2. **Select Y** — the outcome you want to predict.
3. **Select X variables** — the predictors.
4. **Run the analysis.**
5. **Compare** statistical, AI and hybrid models.
6. **Download** the results as CSV or Excel.

The application can handle both **numeric and categorical predictors** such as crop, district, region, season, education level, or industry.
""")
    st.stop()

try:
    df_raw = read_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

if df_raw.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

df = coerce_numeric_like(df_raw)
df = df.replace([np.inf, -np.inf], np.nan)

st.success(f"Loaded **{len(df):,} rows × {len(df.columns):,} columns**")

tab1, tab2, tab3 = st.tabs(["📋 Data preview", "🔎 Data quality", "📚 Guide"])

with tab1:
    st.dataframe(df.head(15), use_container_width=True)

with tab2:
    quality = pd.DataFrame({
        "Variable": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("""
**Outcome (Y):** the variable you want to predict.

**Predictors (X):** variables used to explain or predict Y.

**RMSE / MAE:** lower values indicate smaller prediction errors.

**R²:** higher values indicate more variance explained on the test data.

**Hybrid model:** the statistical model makes a base prediction, then an AI model learns predictable residual structure using out-of-fold residuals.

**Important:** model performance is evaluated on data not used to fit the final models. A model with the best result on one split is not automatically the best model for every future dataset.
""")

numeric_cols, categorical_cols = infer_columns(df)
candidate_y = numeric_cols.copy()

if not candidate_y:
    st.error("No numeric outcome variable was detected. The outcome must be numeric for regression.")
    st.stop()

st.subheader("2️⃣ Select outcome and predictors")
y_col = st.selectbox(
    "Outcome / dependent variable (Y)",
    candidate_y,
    index=0
)

available_x = [c for c in df.columns if c != y_col]
default_x = [c for c in available_x if c in numeric_cols][:5]
if not default_x:
    default_x = available_x[:min(5, len(available_x))]

x_cols = st.multiselect(
    "Predictors / independent variables (X)",
    available_x,
    default=default_x,
    help="You may select numeric and categorical variables."
)

if not x_cols:
    st.warning("Select at least one predictor.")
    st.stop()

y = clean_target(df, y_col)
X = df[x_cols].copy()

# Remove rows where the target is missing; predictors are imputed inside pipelines.
valid_y = y.notna()
X = X.loc[valid_y].reset_index(drop=True)
y = y.loc[valid_y].reset_index(drop=True)

if len(y) < max(30, len(x_cols) + 10):
    st.error(
        f"Only {len(y)} usable outcome observations were found. "
        "Please use a larger dataset (at least about 30 observations is recommended)."
    )
    st.stop()

constant_x = [c for c in x_cols if X[c].nunique(dropna=True) < 2]
if constant_x:
    st.error("These predictors have fewer than two distinct non-missing values: " + ", ".join(constant_x))
    st.stop()

st.info(
    f"**Y:** `{y_col}`  |  **X:** {', '.join(x_cols)}  |  "
    f"**Usable observations:** {len(y):,}"
)

with st.expander("Model choices included"):
    st.write(
        "Linear Regression, Polynomial Regression, Ridge, Lasso, Elastic Net, "
        "Random Forest, Gradient Boosting, and Linear + AI Residual Hybrid."
    )

run_models = st.button(
    "🚀 Run Analysis & Compare Models",
    type="primary",
    use_container_width=True
)

if not run_models:
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state)
)

models = {
    "Linear Regression": make_pipeline_model(LinearRegression(), X),
    "Polynomial Regression": Pipeline([
        ("preprocess", make_preprocessor(X)),
        ("poly", PolynomialFeatures(degree=int(poly_degree), include_bias=False)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Ridge Regression": make_pipeline_model(
        Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]), X
    ),
    "Lasso Regression": make_pipeline_model(
        Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=20000))]), X
    ),
    "Elastic Net": make_pipeline_model(
        Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000))]), X
    ),
    "Random Forest": make_pipeline_model(
        RandomForestRegressor(
            n_estimators=int(rf_trees),
            random_state=int(random_state),
            n_jobs=-1
        ), X
    ),
    "Gradient Boosting": make_pipeline_model(
        GradientBoostingRegressor(
            n_estimators=int(gb_trees),
            learning_rate=0.05,
            max_depth=3,
            random_state=int(random_state)
        ), X
    ),
}

predictions = {}
rows = []

progress = st.progress(0, text="Running models...")
total = len(models) + 1

for i, (name, estimator) in enumerate(models.items(), start=1):
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    pred = fitted.predict(X_test)
    predictions[name] = pred
    m = metric_values(y_test, pred)
    rows.append({
        "Model": name,
        "Type": "AI" if name in ["Random Forest", "Gradient Boosting"] else "Statistical",
        **m
    })
    progress.progress(i / total, text=f"Completed {name}")

hybrid_base = make_pipeline_model(LinearRegression(), X)
hybrid_residual = make_pipeline_model(
    GradientBoostingRegressor(
        n_estimators=int(gb_trees),
        learning_rate=0.05,
        max_depth=3,
        random_state=int(random_state)
    ), X
)

_, _, hybrid_pred, residual_test_pred, oof_residuals = fit_hybrid_oof(
    hybrid_base, hybrid_residual, X_train, y_train, X_test,
    n_splits=int(n_splits), random_state=int(random_state)
)

predictions["Linear + AI Residual Hybrid"] = hybrid_pred
m = metric_values(y_test, hybrid_pred)
rows.append({
    "Model": "Linear + AI Residual Hybrid",
    "Type": "Hybrid",
    **m
})

progress.progress(1.0, text="Analysis complete")
progress.empty()

comparison = pd.DataFrame(rows)
comparison["RMSE Rank"] = comparison["RMSE"].rank(method="min", ascending=True).astype(int)
comparison["MAE Rank"] = comparison["MAE"].rank(method="min", ascending=True).astype(int)
comparison["R² Rank"] = comparison["R²"].rank(method="min", ascending=False).astype(int)
comparison["Average Rank"] = comparison[["RMSE Rank", "MAE Rank", "R² Rank"]].mean(axis=1)
comparison = comparison.sort_values("RMSE").reset_index(drop=True)

st.subheader("3️⃣ Model comparison — unseen test data")
st.dataframe(
    comparison.style.format({
        "RMSE": "{:.6f}", "MAE": "{:.6f}", "R²": "{:.6f}",
        "Average Rank": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)
st.caption(
    "The table is ordered by test RMSE for readability. "
    "The displayed order is not a universal ranking of models."
)

st.subheader("4️⃣ Hybrid incremental value")
linear_row = comparison.loc[comparison["Model"] == "Linear Regression"].iloc[0]
hybrid_row = comparison.loc[comparison["Model"] == "Linear + AI Residual Hybrid"].iloc[0]
rmse_change = linear_row["RMSE"] - hybrid_row["RMSE"]
mae_change = linear_row["MAE"] - hybrid_row["MAE"]
rmse_pct = 100 * rmse_change / linear_row["RMSE"] if linear_row["RMSE"] else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("Linear RMSE", f"{linear_row['RMSE']:.6f}")
c2.metric("Hybrid RMSE", f"{hybrid_row['RMSE']:.6f}")
c3.metric("Hybrid RMSE change", f"{rmse_pct:+.2f}%" if np.isfinite(rmse_pct) else "N/A")

st.write(
    f"**RMSE change:** {rmse_change:.6f}  |  "
    f"**MAE change:** {mae_change:.6f}"
)
st.caption(
    "Positive RMSE/MAE change means the hybrid had lower error on this test split. "
    "It does not guarantee improvement on future data."
)

st.subheader("5️⃣ Actual vs predicted")
fig, ax = plt.subplots(figsize=(10, 6))
for name, pred in predictions.items():
    ax.scatter(y_test, pred, alpha=0.55, s=30, label=name)
mn = min(float(y_test.min()), min(float(np.min(p)) for p in predictions.values()))
mx = max(float(y_test.max()), max(float(np.max(p)) for p in predictions.values()))
ax.plot([mn, mx], [mn, mx], "--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted — Test Set")
ax.grid(alpha=0.25)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
st.pyplot(fig)
plt.close(fig)

st.subheader("6️⃣ Test RMSE comparison")
fig, ax = plt.subplots(figsize=(10, 6))
plot_df = comparison.sort_values("RMSE")
ax.bar(plot_df["Model"], plot_df["RMSE"])
ax.set_ylabel("Test RMSE")
ax.set_title("Test RMSE by Model")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.subheader("7️⃣ Test-set prediction table")
prediction_table = X_test.reset_index(drop=True).copy()
prediction_table.insert(0, "Actual", y_test.reset_index(drop=True))
for name, pred in predictions.items():
    prediction_table[f"Predicted — {name}"] = pred
prediction_table["Hybrid Residual Correction"] = residual_test_pred
st.dataframe(prediction_table, use_container_width=True)

# OLS diagnostics are provided only when all selected predictors can be represented numerically.
st.subheader("8️⃣ Statistical diagnostics for the linear model")

diagnostics_rows = []
coef_df = pd.DataFrame()
vif_df = pd.DataFrame()

X_diag = pd.DataFrame(index=X_train.index)
for col in x_cols:
    if pd.api.types.is_numeric_dtype(X_train[col]):
        X_diag[col] = pd.to_numeric(X_train[col], errors="coerce")
    else:
        # For diagnostics, use one-hot encoding and median/mode imputation.
        dummies = pd.get_dummies(
            X_train[col].astype("string").fillna("Missing"),
            prefix=col,
            drop_first=True,
            dtype=float
        )
        X_diag = pd.concat([X_diag, dummies], axis=1)

X_diag = X_diag.replace([np.inf, -np.inf], np.nan)
X_diag = X_diag.fillna(X_diag.median(numeric_only=True))
X_diag = X_diag.loc[:, X_diag.nunique(dropna=True) > 1]

try:
    X_sm = sm.add_constant(X_diag.astype(float), has_constant="add")
    ols_model = sm.OLS(y_train.reset_index(drop=True), X_sm.reset_index(drop=True)).fit()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Training observations", f"{int(ols_model.nobs):,}")
    m2.metric("Training R²", f"{ols_model.rsquared:.4f}")
    m3.metric("Adjusted R²", f"{ols_model.rsquared_adj:.4f}")
    m4.metric("F-statistic", f"{ols_model.fvalue:.4f}")
    m5.metric("Model p-value", f"{ols_model.f_pvalue:.6g}")

    conf = ols_model.conf_int(alpha=0.05)
    coef_df = pd.DataFrame({
        "Variable": ols_model.params.index,
        "Coefficient": ols_model.params.values,
        "Std. Error": ols_model.bse.values,
        "t-statistic": ols_model.tvalues.values,
        "p-value": ols_model.pvalues.values,
        "95% CI Lower": conf[0].values,
        "95% CI Upper": conf[1].values,
    })

    with st.expander("OLS coefficients"):
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    ss_total = float(np.sum((y_train - y_train.mean()) ** 2))
    ss_resid = float(np.sum(ols_model.resid ** 2))
    ss_reg = ss_total - ss_resid
    df_reg = int(ols_model.df_model)
    df_resid = int(ols_model.df_resid)
    ms_reg = ss_reg / df_reg if df_reg else np.nan
    ms_resid = ss_resid / df_resid if df_resid else np.nan
    f_stat = ms_reg / ms_resid if ms_resid else np.nan
    f_p = stats.f.sf(f_stat, df_reg, df_resid) if np.isfinite(f_stat) else np.nan

    anova_df = pd.DataFrame({
        "Source": ["Regression", "Residual", "Total"],
        "Sum of Squares": [ss_reg, ss_resid, ss_total],
        "df": [df_reg, df_resid, df_reg + df_resid],
        "Mean Square": [ms_reg, ms_resid, np.nan],
        "F": [f_stat, np.nan, np.nan],
        "p-value": [f_p, np.nan, np.nan],
    })
    with st.expander("ANOVA"):
        st.dataframe(anova_df, use_container_width=True, hide_index=True)

    if X_diag.shape[1] >= 2:
        vif_rows = []
        for i, col in enumerate(X_diag.columns):
            try:
                value = variance_inflation_factor(X_diag.astype(float).values, i)
            except Exception:
                value = np.inf
            vif_rows.append({"Variable": col, "VIF": value})
        vif_df = pd.DataFrame(vif_rows)
        with st.expander("VIF"):
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

    try:
        bp = het_breuschpagan(ols_model.resid, X_sm)
        bp_df = pd.DataFrame({
            "Test": ["Breusch-Pagan LM", "Breusch-Pagan F"],
            "Statistic": [bp[0], bp[2]],
            "p-value": [bp[1], bp[3]]
        })
        with st.expander("Breusch-Pagan test"):
            st.dataframe(bp_df, use_container_width=True, hide_index=True)
    except Exception as e:
        bp_df = pd.DataFrame()
        st.info(f"Breusch-Pagan test was not available: {e}")

    with st.expander("Full OLS summary"):
        st.text(ols_model.summary().as_text())

    diagnostics_rows.extend([
        {"Metric": "Training observations", "Value": ols_model.nobs},
        {"Metric": "Training R²", "Value": ols_model.rsquared},
        {"Metric": "Adjusted R²", "Value": ols_model.rsquared_adj},
        {"Metric": "F-statistic", "Value": ols_model.fvalue},
        {"Metric": "Model p-value", "Value": ols_model.f_pvalue},
    ])
except Exception as e:
    st.warning(
        "The predictive models completed, but the detailed OLS diagnostics could not be "
        f"calculated for this dataset: {e}"
    )
    diagnostics_rows.append({"Metric": "OLS diagnostics", "Value": "Unavailable"})

diagnostics_df = pd.DataFrame(diagnostics_rows)

st.subheader("9️⃣ Download your results")
st.download_button(
    "⬇️ Download model comparison CSV",
    comparison.to_csv(index=False).encode("utf-8"),
    "model_comparison.csv",
    "text/csv"
)
st.download_button(
    "⬇️ Download all predictions CSV",
    prediction_table.to_csv(index=False).encode("utf-8"),
    "all_model_predictions.csv",
    "text/csv"
)
st.download_button(
    "⬇️ Download complete Excel report",
    make_excel(comparison, prediction_table, coef_df, vif_df, diagnostics_df),
    "statistical_ai_hybrid_results.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.divider()
st.caption(
    "StatAI — Statistical, AI and Hybrid Regression Analyzer | "
    "For research and educational use. Results depend on data quality, model assumptions, "
    "validation design and the selected settings."
)
