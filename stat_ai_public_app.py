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
:root { --stat-accent: #176b87; --stat-accent-dark: #0d4f66; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
.main-title { font-size: clamp(1.8rem, 4vw, 3rem); font-weight: 800; letter-spacing: -0.04em; line-height: 1.1; margin-bottom: .35rem; color: #123746; }
.subtitle { font-size: 1.08rem; color: #52636b; margin-bottom: 1.25rem; max-width: 900px; }
.hero { padding: 1.35rem 1.5rem; border-radius: 18px; background: linear-gradient(135deg, #e9f5f8 0%, #f8fbfc 100%); border: 1px solid #c9e3ea; margin-bottom: 1.25rem; }
.hero-kicker { text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; font-weight: 800; color: #176b87; margin-bottom: .45rem; }
.section-label { font-size: .78rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: #176b87; margin-top: .8rem; }
.step-card { padding: 1rem 1.15rem; border: 1px solid #dce7eb; border-radius: 14px; margin-bottom: .75rem; background: #ffffff; }
.small-note { font-size: .9rem; color: #66777e; }
div[data-testid="stMetric"] { background: #f6fafb; border: 1px solid #dcecef; padding: .8rem 1rem; border-radius: 13px; }
div[data-testid="stMetricValue"] { color: #123746; }
.stButton > button { border-radius: 10px; font-weight: 700; }
[data-testid="stSidebar"] { background: #f4f8fa; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><div class="hero-kicker">Research & predictive analytics</div><div class="main-title">📊 Statistical–AI Hybrid Modelling Platform</div><div class="subtitle">A practical environment for prediction, statistical inference, diagnostics, and comparison of statistical, artificial-intelligence, and hybrid regression models.</div></div>""", unsafe_allow_html=True)

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
    st.header("⚙️ Analysis controls")
    st.markdown("**Validation**")
    test_size = st.slider("Test-set proportion", 0.15, 0.40, 0.25, 0.05)
    random_state = st.number_input("Random seed", 1, 9999, 42, 1)
    n_splits = st.slider("Cross-validation folds", 3, 10, 5, 1)

    st.markdown("**Model settings**")
    rf_trees = st.slider("Random Forest trees", 100, 800, 300, 50)
    gb_trees = st.slider("Gradient Boosting trees", 50, 500, 200, 25)
    poly_degree = st.selectbox("Polynomial degree", [2, 3], index=0)

    st.markdown("**Model selection**")
    selection_criterion = st.selectbox(
        "Choose model by",
        ["Cross-validation RMSE", "Cross-validation MAE", "Cross-validation R²", "Test RMSE", "Test MAE", "Test R²"],
        index=0,
        help="For rigorous research, use cross-validation for model selection and reserve the final test set for confirmation."
    )
    model_choice = st.radio(
        "Final model display",
        ["Automatically select", "Choose manually"],
        index=0
    )

    st.divider()
    st.markdown("**Privacy note**")
    st.caption(
        "Uploaded data are processed for the current analysis session. "
        "Do not upload confidential or personally identifiable data unless you are "
        "comfortable with the hosting environment."
    )

st.markdown('<div class="section-label">Step 1 · Data input</div>', unsafe_allow_html=True)
st.subheader("Upload your dataset")
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

st.success(f"Dataset loaded successfully: **{len(df):,} rows × {len(df.columns):,} columns**")

tab1, tab2, tab3 = st.tabs(["📋 Data preview", "🔎 Data quality", "📚 Method guide"])

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

model_options = [
    "Linear Regression", "Polynomial Regression", "Ridge Regression",
    "Lasso Regression", "Elastic Net", "Random Forest",
    "Gradient Boosting", "Linear + AI Residual Hybrid"
]
selected_models = st.multiselect(
    "Models to run",
    model_options,
    default=model_options,
    help="Choose which statistical, AI and hybrid models to include in the analysis."
)

if model_choice == "Choose manually":
    manual_model = st.selectbox("Model to report as final selected model", selected_models or model_options)
else:
    manual_model = None

with st.expander("What the models mean"):
    st.markdown("""
- **Linear Regression:** interpretable parametric baseline.
- **Polynomial Regression:** adds nonlinear polynomial terms.
- **Ridge / Lasso / Elastic Net:** regularized linear models that can reduce overfitting and handle correlated predictors.
- **Random Forest:** tree ensemble that can capture nonlinearities and interactions.
- **Gradient Boosting:** sequential tree ensemble that learns residual structure.
- **Linear + AI Residual Hybrid:** linear statistical structure plus an AI learner trained on out-of-fold residuals.
""")

if not selected_models:
    st.warning("Select at least one model.")
    st.stop()

run_models = st.button(
    "🚀 Run Analysis & Generate Statistical Report",
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
train_predictions = {}
rows = []
cv_rows = []

progress = st.progress(0, text="Running models...")
selected_standard = [m for m in selected_models if m != "Linear + AI Residual Hybrid"]
total_steps = max(1, len(selected_standard) * 2 + (2 if "Linear + AI Residual Hybrid" in selected_models else 0))
step = 0

for name in selected_standard:
    estimator = models[name]
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    pred_test = fitted.predict(X_test)
    pred_train = fitted.predict(X_train)
    predictions[name] = pred_test
    train_predictions[name] = pred_train
    m_test = metric_values(y_test, pred_test)
    m_train = metric_values(y_train, pred_train)
    rows.append({
        "Model": name,
        "Type": "AI" if name in ["Random Forest", "Gradient Boosting"] else "Statistical",
        "Train RMSE": m_train["RMSE"], "Train MAE": m_train["MAE"], "Train R²": m_train["R²"],
        "RMSE": m_test["RMSE"], "MAE": m_test["MAE"], "R²": m_test["R²"]
    })
    step += 1
    progress.progress(step / total_steps, text=f"Testing {name}")

    # Cross-validation on the training portion; final test set remains untouched.
    cv_rmse, cv_mae, cv_r2 = [], [], []
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    for tr_idx, va_idx in kf.split(X_train):
        fold_model = clone(estimator)
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        fold_pred = fold_model.predict(X_train.iloc[va_idx])
        fold_metrics = metric_values(y_train.iloc[va_idx], fold_pred)
        cv_rmse.append(fold_metrics["RMSE"])
        cv_mae.append(fold_metrics["MAE"])
        cv_r2.append(fold_metrics["R²"])
    cv_rows.append({
        "Model": name,
        "CV RMSE": float(np.mean(cv_rmse)),
        "CV RMSE SD": float(np.std(cv_rmse, ddof=1)) if len(cv_rmse) > 1 else 0.0,
        "CV MAE": float(np.mean(cv_mae)),
        "CV MAE SD": float(np.std(cv_mae, ddof=1)) if len(cv_mae) > 1 else 0.0,
        "CV R²": float(np.mean(cv_r2)),
        "CV R² SD": float(np.std(cv_r2, ddof=1)) if len(cv_r2) > 1 else 0.0,
    })
    step += 1
    progress.progress(step / total_steps, text=f"Cross-validating {name}")

if "Linear + AI Residual Hybrid" in selected_models:
    hybrid_base = make_pipeline_model(LinearRegression(), X)
    hybrid_residual = make_pipeline_model(
        GradientBoostingRegressor(
            n_estimators=int(gb_trees), learning_rate=0.05, max_depth=3,
            random_state=int(random_state)
        ), X
    )
    hybrid_base_final, hybrid_residual_final, hybrid_pred, residual_test_pred, oof_residuals = fit_hybrid_oof(
        hybrid_base, hybrid_residual, X_train, y_train, X_test,
        n_splits=int(n_splits), random_state=int(random_state)
    )
    # In-sample training prediction from the fitted final components.
    base_train_pred = hybrid_base_final.predict(X_train)
    residual_train_pred = hybrid_residual_final.predict(X_train)
    hybrid_train_pred = base_train_pred + residual_train_pred
    predictions["Linear + AI Residual Hybrid"] = hybrid_pred
    train_predictions["Linear + AI Residual Hybrid"] = hybrid_train_pred
    m_test = metric_values(y_test, hybrid_pred)
    m_train = metric_values(y_train, hybrid_train_pred)
    rows.append({
        "Model": "Linear + AI Residual Hybrid", "Type": "Hybrid",
        "Train RMSE": m_train["RMSE"], "Train MAE": m_train["MAE"], "Train R²": m_train["R²"],
        "RMSE": m_test["RMSE"], "MAE": m_test["MAE"], "R²": m_test["R²"]
    })
    # The hybrid is evaluated through repeated OOF base residual construction on the training data.
    # This is reported separately and is not used as a substitute for the untouched final test set.
    cv_rows.append({
        "Model": "Linear + AI Residual Hybrid",
        "CV RMSE": float(np.sqrt(np.mean(oof_residuals ** 2))),
        "CV RMSE SD": np.nan,
        "CV MAE": float(np.mean(np.abs(oof_residuals))),
        "CV MAE SD": np.nan,
        "CV R²": float(1 - np.sum(oof_residuals ** 2) / np.sum((y_train.to_numpy() - y_train.mean()) ** 2)),
        "CV R² SD": np.nan,
    })
    step += 2
    progress.progress(min(1.0, step / total_steps), text="Completed hybrid residual analysis")
else:
    residual_test_pred = np.zeros(len(X_test))

progress.progress(1.0, text="Analysis complete")
progress.empty()

comparison = pd.DataFrame(rows)
cv_comparison = pd.DataFrame(cv_rows)
comparison = comparison.merge(cv_comparison, on="Model", how="left")
comparison["Test RMSE Rank"] = comparison["RMSE"].rank(method="min", ascending=True).astype(int)
comparison["Test MAE Rank"] = comparison["MAE"].rank(method="min", ascending=True).astype(int)
comparison["Test R² Rank"] = comparison["R²"].rank(method="min", ascending=False).astype(int)
comparison["CV RMSE Rank"] = comparison["CV RMSE"].rank(method="min", ascending=True).astype(int)
comparison["CV MAE Rank"] = comparison["CV MAE"].rank(method="min", ascending=True).astype(int)
comparison["CV R² Rank"] = comparison["CV R²"].rank(method="min", ascending=False).astype(int)
comparison["Average Test Rank"] = comparison[["Test RMSE Rank", "Test MAE Rank", "Test R² Rank"]].mean(axis=1)
comparison["Generalization Gap R²"] = comparison["Train R²"] - comparison["R²"]
comparison = comparison.sort_values("RMSE").reset_index(drop=True)

# Determine the selected model using the user's criterion.
if model_choice == "Choose manually":
    selected_final_model = manual_model
else:
    if selection_criterion == "Cross-validation RMSE":
        selected_final_model = comparison.loc[comparison["CV RMSE"].idxmin(), "Model"]
    elif selection_criterion == "Cross-validation MAE":
        selected_final_model = comparison.loc[comparison["CV MAE"].idxmin(), "Model"]
    elif selection_criterion == "Cross-validation R²":
        selected_final_model = comparison.loc[comparison["CV R²"].idxmax(), "Model"]
    elif selection_criterion == "Test RMSE":
        selected_final_model = comparison.loc[comparison["RMSE"].idxmin(), "Model"]
    elif selection_criterion == "Test MAE":
        selected_final_model = comparison.loc[comparison["MAE"].idxmin(), "Model"]
    else:
        selected_final_model = comparison.loc[comparison["R²"].idxmax(), "Model"]

selected_row = comparison.loc[comparison["Model"] == selected_final_model].iloc[0]

st.subheader("3️⃣ Model selection and complete performance report")
sel_col1, sel_col2, sel_col3 = st.columns(3)
sel_col1.metric("Selected model", selected_final_model)
sel_col2.metric("Selection criterion", selection_criterion)
sel_col3.metric("Test R²", f"{selected_row['R²']:.4f}")

st.info(
    "**How to use this selection:** for a research analysis, the preferred default is a cross-validation criterion. "
    "The final hold-out test set should then be used as an independent confirmation of performance. "
    "If you choose a test-set criterion, interpret it as a descriptive selection on this particular split, not as an unbiased future-performance estimate."
)

report_cols = [
    "Model", "Type", "Train RMSE", "Train MAE", "Train R²",
    "CV RMSE", "CV RMSE SD", "CV MAE", "CV MAE SD", "CV R²", "CV R² SD",
    "RMSE", "MAE", "R²", "Generalization Gap R²",
    "Test RMSE Rank", "Test MAE Rank", "Test R² Rank", "Average Test Rank"
]
st.dataframe(
    comparison[report_cols].style.format({
        c: "{:.6f}" for c in report_cols if c not in ["Model", "Type"] and "Rank" not in c
    }).format({
        c: "{:.2f}" for c in report_cols if "Rank" in c
    }),
    use_container_width=True, hide_index=True
)
st.caption(
    "Train metrics describe fit to the training data. CV metrics summarize training-set cross-validation. "
    "Test metrics come from the untouched hold-out test set. Lower RMSE/MAE is better; higher R² is better. "
    "The report is descriptive and should be interpreted with the validation design and model assumptions."
)

with st.expander("📘 Statistical interpretation of the performance report", expanded=True):
    st.markdown(f"""
**Selected model:** `{selected_final_model}` using **{selection_criterion}**.

- **RMSE:** measures the typical prediction error on the outcome scale, with larger errors receiving greater weight.
- **MAE:** measures the average absolute prediction error and is less sensitive to very large errors than RMSE.
- **R²:** measures the proportion of outcome variation accounted for relative to the test-set mean baseline; it can be negative for a poor model.
- **Cross-validation:** estimates expected performance across multiple validation folds within the training data.
- **Generalization gap:** `Train R² − Test R²`; a large positive value can indicate overfitting, although it is not by itself a formal overfitting test.
- **Model selection:** the app lets you choose the criterion rather than silently declaring one algorithm universally superior.
""")

st.subheader("4️⃣ Hybrid incremental value")
if "Linear Regression" in comparison["Model"].values and "Linear + AI Residual Hybrid" in comparison["Model"].values:
    linear_row = comparison.loc[comparison["Model"] == "Linear Regression"].iloc[0]
    hybrid_row = comparison.loc[comparison["Model"] == "Linear + AI Residual Hybrid"].iloc[0]
    rmse_change = linear_row["RMSE"] - hybrid_row["RMSE"]
    mae_change = linear_row["MAE"] - hybrid_row["MAE"]
    rmse_pct = 100 * rmse_change / linear_row["RMSE"] if linear_row["RMSE"] else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Linear RMSE", f"{linear_row['RMSE']:.6f}")
    c2.metric("Hybrid RMSE", f"{hybrid_row['RMSE']:.6f}")
    c3.metric("Hybrid RMSE improvement", f"{rmse_pct:+.2f}%" if np.isfinite(rmse_pct) else "N/A")

    st.write(
        f"**RMSE improvement:** {rmse_change:.6f}  |  "
        f"**MAE improvement:** {mae_change:.6f}"
    )
    st.caption(
        "A positive improvement means the hybrid had lower error than linear regression on this test split. "
        "This does not establish that the hybrid will outperform other models or future datasets."
    )
else:
    st.info("Hybrid incremental-value analysis requires both Linear Regression and Linear + AI Residual Hybrid to be selected.")

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

st.subheader("7️⃣ Cross-validation performance")
cv_plot = comparison.sort_values("CV RMSE")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(cv_plot["Model"], cv_plot["CV RMSE"], yerr=cv_plot["CV RMSE SD"].fillna(0), capsize=4)
ax.set_ylabel("Mean CV RMSE")
ax.set_title("Cross-validation RMSE on the Training Set")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.subheader("8️⃣ Test-set prediction table")
prediction_table = X_test.reset_index(drop=True).copy()
prediction_table.insert(0, "Actual", y_test.reset_index(drop=True))
for name, pred in predictions.items():
    prediction_table[f"Predicted — {name}"] = pred
prediction_table["Hybrid Residual Correction"] = residual_test_pred
st.dataframe(prediction_table, use_container_width=True)

# OLS diagnostics are provided only when all selected predictors can be represented numerically.
st.subheader("9️⃣ Statistical diagnostics for the linear model")

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

st.subheader("🔟 Download your complete statistical report")
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

report_text = f"""Statistical–AI Regression Analysis Report

Outcome: {y_col}
Predictors: {', '.join(x_cols)}
Usable observations: {len(y)}
Training observations: {len(y_train)}
Test observations: {len(y_test)}
Test-set proportion: {test_size:.2f}
Random seed: {int(random_state)}
Cross-validation folds: {int(n_splits)}

Selected model: {selected_final_model}
Selection criterion: {selection_criterion}

Model performance:
{comparison[report_cols].to_string(index=False)}

Interpretation:
Lower RMSE and MAE indicate smaller prediction errors. Higher R² indicates greater explained variation relative to the test-set mean baseline. Cross-validation summarizes training-set validation performance, while the hold-out test set provides the final independent performance check for this split.

Important limitation:
Model performance is dataset- and validation-design dependent. The selected model is not universally optimal. Statistical inference additionally depends on model specification and assumptions.
"""
st.download_button(
    "⬇️ Download statistical report (TXT)",
    report_text.encode("utf-8"),
    "statistical_ai_full_report.txt",
    "text/plain"
)

st.divider()
st.caption(
    "StatAI — Statistical, AI and Hybrid Regression Analyzer | "
    "For research and educational use. Results depend on data quality, model assumptions, "
    "validation design and the selected settings."
)
