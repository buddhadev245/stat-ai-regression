import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="StatAI | Statistical–AI Hybrid Modelling", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.4rem;padding-bottom:3rem}
.hero{padding:1.6rem 1.8rem;border:1px solid #cfe2e8;border-radius:20px;background:linear-gradient(135deg,#edf8fb,#fff);margin-bottom:1rem}
.kicker{font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#176b87}
.title{font-size:clamp(2rem,4vw,3.1rem);font-weight:850;letter-spacing:-.04em;color:#123746;line-height:1.05}
.subtitle{font-size:1.02rem;color:#5b6d74;max-width:1050px;margin-top:.5rem}
.card{padding:1.1rem 1.25rem;border:1px solid #dce8ec;border-radius:16px;background:#fff;box-shadow:0 4px 18px rgba(18,55,70,.05);margin:.6rem 0 1rem}
.pill{display:inline-block;padding:.35rem .7rem;border-radius:999px;background:#f1f8fa;border:1px solid #d7e8ed;color:#28576a;font-weight:700;font-size:.8rem;margin:.15rem}
.muted,.small{color:#64757c}
.formula{padding:1rem;background:#f7fbfc;border-left:4px solid #176b87;border-radius:8px;font-size:1.02rem}
div[data-testid="stMetric"]{background:#f6fafb;border:1px solid #dcecef;padding:.75rem 1rem;border-radius:13px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#f4f8fa,#edf5f7)}
.wf-viewport{width:100%;overflow-x:auto;overflow-y:hidden;padding:14px 8px 18px;border:1px solid #d9e8ee;border-radius:14px;background:#fbfdfe}
.wf-track{display:flex;align-items:stretch;gap:12px;width:max-content;min-width:max-content}
.wf-card{width:230px;min-height:145px;flex:0 0 230px;border:1px solid #b9d9e5;border-radius:15px;background:#fff;padding:15px;box-shadow:0 3px 12px rgba(18,55,70,.06);box-sizing:border-box}
.wf-num{font-size:.76rem;font-weight:800;color:#176b87;text-transform:uppercase;letter-spacing:.08em}.wf-title{font-weight:800;color:#123746;font-size:1.02rem;margin:.35rem 0}.wf-desc{font-size:.86rem;color:#5f7077;line-height:1.4}.wf-arrow{flex:0 0 30px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;color:#176b87;font-weight:800}
.note{margin-top:10px;padding:11px 14px;border-radius:10px;background:#eaf3ff;color:#15508a;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="kicker">Research • prediction • inference • reproducibility</div><div class="title">📊 Statistical–AI Hybrid Modelling Platform</div><div class="subtitle">A browser-based workspace for exploratory analysis, statistical regression, machine-learning comparison, residual hybrid modelling, diagnostics, interpretation and reproducible reporting.</div></div>', unsafe_allow_html=True)

ID_NAMES={"id","identifier","employee_id","student_id","customer_id","record_id","serial_no","serial_number","roll_no","roll_number","row_id","index"}
MODEL_NAMES=["Linear Regression","Polynomial Regression","Ridge Regression","Lasso Regression","Elastic Net","Random Forest","Gradient Boosting","Linear + AI Residual Hybrid"]


def read_data(upload):
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)


def coerce_numeric_like(df):
    out=df.copy()
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            converted=pd.to_numeric(out[c].astype(str).str.replace(",","",regex=False).str.strip(),errors="coerce")
            if converted.notna().mean() >= .85:
                out[c]=converted
    return out


def metrics(y_true,y_pred):
    return {"RMSE":float(np.sqrt(mean_squared_error(y_true,y_pred))),"MAE":float(mean_absolute_error(y_true,y_pred)),"R²":float(r2_score(y_true,y_pred))}


def make_preprocessor(X):
    num=[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat=[c for c in X.columns if c not in num]
    transformers=[]
    if num:
        transformers.append(("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num))
    if cat:
        transformers.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat))
    return ColumnTransformer(transformers=transformers,remainder="drop"),num,cat


def make_pipe(estimator,X):
    prep,_,_=make_preprocessor(X)
    return Pipeline([("prep",prep),("model",estimator)])


def model_specs(X,rf_trees,gb_trees,degree,seed):
    return {
        "Linear Regression":make_pipe(LinearRegression(),X),
        "Polynomial Regression":Pipeline([("prep",make_preprocessor(X)[0]),("poly",PolynomialFeatures(degree=degree,include_bias=False)),("scale",StandardScaler()),("model",LinearRegression())]),
        "Ridge Regression":make_pipe(Ridge(alpha=1.0),X),
        "Lasso Regression":make_pipe(Lasso(alpha=.1,max_iter=30000),X),
        "Elastic Net":make_pipe(ElasticNet(alpha=.1,l1_ratio=.5,max_iter=30000),X),
        "Random Forest":make_pipe(RandomForestRegressor(n_estimators=rf_trees,random_state=seed,n_jobs=-1),X),
        "Gradient Boosting":make_pipe(GradientBoostingRegressor(n_estimators=gb_trees,learning_rate=.05,max_depth=3,random_state=seed),X),
    }


def hybrid_fit(base,residual,X_train,y_train,X_eval,folds,seed):
    X_train=X_train.reset_index(drop=True); y_train=y_train.reset_index(drop=True); X_eval=X_eval.reset_index(drop=True)
    kf=KFold(n_splits=min(folds,len(X_train)),shuffle=True,random_state=seed)
    oof=np.zeros(len(y_train))
    for ti,vi in kf.split(X_train):
        b=make_clone(base); b.fit(X_train.iloc[ti],y_train.iloc[ti]); oof[vi]=y_train.iloc[vi].to_numpy()-b.predict(X_train.iloc[vi])
    r=make_clone(residual); r.fit(X_train,oof)
    bfinal=make_clone(base); bfinal.fit(X_train,y_train)
    return bfinal,r,bfinal.predict(X_eval)+r.predict(X_eval),oof


def make_clone(obj):
    from sklearn.base import clone
    return clone(obj)


def rank(s,ascending):
    return s.rank(method="min",ascending=ascending).astype(int)


def build_ols_data(X):
    pieces=[]
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            z=pd.to_numeric(X[c],errors="coerce").rename(c)
        else:
            z=pd.get_dummies(X[c].astype("string").fillna("Missing"),prefix=c,drop_first=True,dtype=float)
        pieces.append(z if isinstance(z,pd.DataFrame) else z.to_frame())
    if not pieces:
        return pd.DataFrame(index=X.index)
    z=pd.concat(pieces,axis=1).replace([np.inf,-np.inf],np.nan)
    for c in z.columns:
        if z[c].isna().any(): z[c]=z[c].fillna(z[c].median())
    return z.loc[:,z.nunique(dropna=False)>1]

# Sidebar
with st.sidebar:
    st.header("⚙️ Analysis controls")
    test_size=st.slider("Test-set proportion",0.15,0.40,0.20,0.05)
    seed=int(st.number_input("Random seed",1,9999,42,1))
    folds=st.slider("Cross-validation folds",3,10,5,1)
    st.divider(); st.subheader("Model settings")
    rf_trees=st.slider("Random Forest trees",50,500,200,50)
    gb_trees=st.slider("Gradient Boosting trees",50,400,150,25)
    degree=st.selectbox("Polynomial degree",[2,3],index=0)
    st.divider(); st.subheader("Model selection")
    criterion=st.selectbox("Selection criterion",["Cross-validation RMSE","Cross-validation MAE","Cross-validation R²","Test RMSE","Test MAE","Test R²"])
    selection_mode=st.radio("Final model display",["Automatically select","Choose manually"])
    st.divider(); st.caption("Variable selection is available prominently in the main workspace. Full-data fit, statistical inference and out-of-sample prediction are reported separately.")

upload=st.file_uploader("📥 Upload CSV or Excel dataset",type=["csv","xlsx","xls"])
if upload is None:
    st.markdown('<div class="card"><h3>Start here</h3><p>Upload a CSV or Excel dataset. The app will identify usable numeric variables, exclude common identifier columns from the default predictor selection, and then let you run the analysis.</p></div>',unsafe_allow_html=True)
    st.stop()

try:
    df=coerce_numeric_like(read_data(upload)).replace([np.inf,-np.inf],np.nan)
except Exception as e:
    st.error(f"Could not read the dataset: {e}"); st.stop()
if df.empty:
    st.error("The uploaded dataset is empty."); st.stop()

numeric=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not numeric:
    st.error("No usable numeric variables were detected."); st.stop()

st.success(f"Loaded **{len(df):,} rows × {len(df.columns):,} columns**")

# Main variable-selection panel
# Y must be selected before variables depending on y_col are calculated.
st.markdown("### 🎯 Choose your regression variables")
st.caption("Select the dependent variable (Y) and one or more independent variables (X). The X selector accepts both numeric and categorical predictors. Common identifier columns are excluded from the default selection but remain available if you intentionally want to use one.")
sel1, sel2 = st.columns([1, 2], gap="large")
with sel1:
    y_col=st.selectbox("Dependent variable / outcome (Y)", numeric, key="main_y")
with sel2:
    available=[c for c in df.columns if c!=y_col]
    non_id=[c for c in available if str(c).strip().lower().replace(" ","_") not in ID_NAMES]
    default=[c for c in non_id if c in numeric][:5]
    if not default:
        default=[c for c in available if c in numeric][:5]
    x_cols=st.multiselect("Independent variables / predictors (X)", available, default=default, key="main_x", help="Choose the variables used to explain or predict Y. You can select multiple variables.")

if not x_cols:
    st.warning("⚠️ Please select at least one independent variable (X) before running the analysis.")
    st.stop()

excluded_ids=[c for c in available if c not in x_cols and str(c).strip().lower().replace(" ","_") in ID_NAMES]
if excluded_ids:
    st.info("ℹ️ Identifier columns excluded from the default predictors: " + ", ".join(excluded_ids))

# Show tabs before analysis so the app never crashes just because a model fails.
tabs=st.tabs(["🏠 Overview","📁 Data & EDA","📊 Full Data Results","📚 Theory, Workflow & Models","📈 Predictive Results","🔬 Hybrid Analysis","🩺 Diagnostics & Inference","💡 Results Interpretation","🎨 Chart Studio","📄 Report & Export"])

with tabs[0]:
    st.markdown('<div class="card"><h2>Analysis overview</h2><p>Upload data, inspect quality, choose the outcome and predictors, run the models, compare full-data fit with out-of-sample prediction, examine the residual hybrid, check diagnostics, and interpret the evidence.</p></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Rows",f"{len(df):,}"); b.metric("Columns",f"{len(df.columns):,}"); c.metric("Selected predictors",f"{len(x_cols):,}")
    st.markdown("### Analysis status")
    st.info("Click **Run analysis** below after selecting your variables. The model calculations are intentionally separated from the page layout so a model error does not crash the entire application.")

with tabs[1]:
    st.subheader("📁 Data & EDA")
    st.dataframe(df.head(20),use_container_width=True,hide_index=True)
    st.markdown("### Data quality")
    q=pd.DataFrame({"Variable":df.columns,"Type":[str(df[c].dtype) for c in df.columns],"Missing":df.isna().sum().values,"Unique":df.nunique(dropna=True).values})
    st.dataframe(q,use_container_width=True,hide_index=True)
    num_cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols)>=2:
        st.plotly_chart(px.imshow(df[num_cols].corr(),text_auto=".2f",aspect="auto",title="Numeric correlation matrix",color_continuous_scale="RdBu_r"),use_container_width=True)

with tabs[3]:
    st.subheader("📚 Regression Theory & How It Works")
    st.caption("This research-oriented guide explains what each model estimates, the mathematics behind it, how this application runs it, what to look for in the output, and the main limitations.")

    theory_tabs=st.tabs(["🧭 Complete process","📐 Regression models","🤖 AI models","🔗 Hybrid regression","📊 Validation & inference"])

    with theory_tabs[0]:
        st.markdown("### 1. How the complete regression process works")
        workflow=[
            ("1","Define the research question","Choose what you want to explain or predict. Select Y as the dependent/outcome variable and X variables as predictors."),
            ("2","Prepare the data","Check variable types, missing values, invalid values, identifiers, duplicates and the usable sample."),
            ("3","Explore the data","Inspect distributions, relationships, outliers, correlation and possible nonlinear patterns before fitting models."),
            ("4","Create the modelling sample","The app keeps the selected Y and X variables, handles numeric/categorical predictors through preprocessing, and removes unusable outcome rows."),
            ("5","Split train and test data","The training set is used to fit models. The untouched test set is kept for final out-of-sample evaluation."),
            ("6","Fit statistical models","Linear, polynomial, Ridge, Lasso and Elastic Net provide increasingly flexible or regularised statistical representations."),
            ("7","Fit AI models","Random Forest and Gradient Boosting learn nonlinearities and interactions from the predictors."),
            ("8","Build the hybrid","The statistical base model is fitted with out-of-fold predictions; its out-of-fold residuals are learned by an AI residual model; the correction is added to the statistical prediction."),
            ("9","Cross-validate","The training data are repeatedly divided into folds to estimate how models behave on unseen training-fold observations."),
            ("10","Evaluate the final test set","RMSE, MAE and R² are calculated on observations not used to fit the final models."),
            ("11","Check inference and diagnostics","OLS coefficients, standard errors, p-values, confidence intervals, ANOVA, VIF and Breusch–Pagan diagnostics are examined separately from predictive accuracy."),
            ("12","Interpret and report","Explain what the data support, distinguish inference from prediction, report uncertainty/limitations, and record the modelling choices for reproducibility."),
        ]
        cards=[]
        for i,(n,t,d) in enumerate(workflow):
            cards.append(f'<div class="wf-card"><div class="wf-num">Step {n}</div><div class="wf-title">{t}</div><div class="wf-desc">{d}</div></div>')
            if i<len(workflow)-1: cards.append('<div class="wf-arrow">→</div>')
        st.markdown('<div class="wf-viewport"><div class="wf-track">'+''.join(cards)+'</div></div><div class="note"><b>Read left → right.</b> Scroll horizontally through the complete analysis pipeline.</div>',unsafe_allow_html=True)

        st.markdown("### 2. What the app actually does when you press Run")
        run_steps=[
            ("A","Select Y and X","The chosen variables define the regression problem."),
            ("B","Preprocess X","Numeric predictors are imputed/scaled where required; categorical predictors are imputed and one-hot encoded."),
            ("C","Fit models on training data","Each statistical/AI estimator is fit independently using the same training observations."),
            ("D","Generate predictions","Each fitted model predicts the untouched test observations."),
            ("E","Calculate metrics","The app calculates RMSE, MAE and R² and also performs cross-validation within the training data."),
            ("F","Fit the full-data models","A second set of models is fit to all usable observations for descriptive in-sample comparison; these results are clearly labelled as full-data/in-sample."),
            ("G","Run OLS inference","The selected predictors are represented in an OLS design matrix so coefficient inference and diagnostics can be reported separately."),
            ("H","Store outputs","The tabs display the same completed analysis rather than starting a separate run."),
        ]
        for n,t,d in run_steps:
            st.markdown(f'<div class="card"><b>{n}. {t}</b><br><span class="muted">{d}</span></div>',unsafe_allow_html=True)

    with theory_tabs[1]:
        model_pick=st.selectbox("Select a regression method to study",["Linear Regression","Polynomial Regression","Ridge Regression","Lasso Regression","Elastic Net"],key="theory_stat_model")
        model_info={
            "Linear Regression": {
                "idea":"Estimates an additive linear relationship between Y and the predictors.",
                "equation":r"Y_i=\beta_0+\beta_1X_{i1}+\cdots+\beta_pX_{ip}+\varepsilon_i",
                "objective":r"\hat\beta=\arg\min_\beta\sum_{i=1}^{n}(Y_i-X_i^T\beta)^2",
                "workflow":"Data → specify Y/X → estimate coefficients by least squares → fitted values → residuals → diagnostics → prediction",
                "app":"The app fits LinearRegression inside the common preprocessing pipeline, predicts the held-out test data, calculates RMSE/MAE/R², runs cross-validation, and separately fits an OLS model for inferential statistics.",
                "interpret":"A coefficient describes the expected change in Y associated with a one-unit change in that predictor while the other included predictors are held constant, conditional on the model specification and assumptions.",
                "caution":"Linearity, influential observations, multicollinearity, non-constant variance and omitted structure can affect interpretation. Statistical significance is not the same as predictive usefulness or causality."
            },
            "Polynomial Regression": {
                "idea":"Adds polynomial terms such as X² or X³ to represent curvature while remaining linear in the coefficients.",
                "equation":r"Y=\beta_0+\beta_1X+\beta_2X^2+\cdots+\beta_dX^d+\varepsilon",
                "objective":"Estimate coefficients by least squares after generating polynomial features.",
                "workflow":"Data → generate polynomial features → scale → estimate coefficients → predict → validate",
                "app":"The app creates PolynomialFeatures at the selected degree (2 or 3), standardises the expanded features, fits linear regression, and compares the result with the other models.",
                "interpret":"Individual higher-order coefficients are often harder to interpret alone. The fitted curve and out-of-sample metrics are more informative for prediction.",
                "caution":"Higher degree increases flexibility and can create unstable extrapolation and overfitting, especially with small samples."
            },
            "Ridge Regression": {
                "idea":"Linear regression with an L2 penalty that shrinks coefficients toward zero and can stabilise estimation when predictors are correlated.",
                "equation":r"\hat\beta=\arg\min_\beta\left[\sum_i(Y_i-X_i^T\beta)^2+\lambda\sum_j\beta_j^2\right]",
                "objective":"Balance fit to the data against coefficient magnitude.",
                "workflow":"Data → preprocess/standardise → choose penalty α → shrink coefficients → predict → validate",
                "app":"The app uses Ridge(alpha=1.0) after preprocessing and evaluates it with the same train/test and cross-validation structure as the other models.",
                "interpret":"Ridge keeps all predictors in the model but reduces the size of their coefficients. It is often useful when predictors contain overlapping information.",
                "caution":"Coefficient shrinkage means Ridge coefficients are not equivalent to ordinary least-squares coefficients; the penalty parameter should ideally be tuned."
            },
            "Lasso Regression": {
                "idea":"Linear regression with an L1 penalty that encourages some coefficients to become exactly zero.",
                "equation":r"\hat\beta=\arg\min_\beta\left[\sum_i(Y_i-X_i^T\beta)^2+\lambda\sum_j|\beta_j|\right]",
                "objective":"Trade predictive fit against sparse coefficient structure.",
                "workflow":"Data → preprocess/standardise → apply L1 penalty → shrink/select coefficients → predict → validate",
                "app":"The app fits Lasso(alpha=0.1) with a high iteration limit, then compares its test and cross-validation performance with Ridge and the unregularised linear model.",
                "interpret":"A zero coefficient means the fitted penalised model has removed that feature from the linear predictor at the selected penalty strength.",
                "caution":"With correlated predictors, Lasso may select one variable and suppress another even when both contain information; coefficient selection can be unstable."
            },
            "Elastic Net": {
                "idea":"Combines L1 sparsity with L2 stabilisation.",
                "equation":r"\hat\beta=\arg\min_\beta\left[\sum_i(Y_i-X_i^T\beta)^2+\lambda\left(\alpha\sum_j|\beta_j|+(1-\alpha)\sum_j\beta_j^2\right)\right]",
                "objective":"Balance variable selection and coefficient shrinkage.",
                "workflow":"Data → preprocess/standardise → combine L1/L2 penalties → estimate → predict → validate",
                "app":"The app uses ElasticNet(alpha=0.1, l1_ratio=0.5) as a balanced regularised reference model.",
                "interpret":"A useful option when predictors are numerous or correlated and a mixture of shrinkage and sparsity is desirable.",
                "caution":"The penalty settings materially affect the fitted model and are better tuned with validation for research production use."
            },
        }[model_pick]
        st.markdown(f"### {model_pick}")
        st.markdown(f'<div class="card"><b>Core idea</b><br>{model_info["idea"]}</div>',unsafe_allow_html=True)
        st.markdown("**Mathematical form**")
        st.latex(model_info["equation"])
        st.markdown("**Estimation / optimisation**")
        st.latex(model_info["objective"])
        st.markdown("**Process flow**")
        st.markdown(f'<div class="formula">{model_info["workflow"]}</div>',unsafe_allow_html=True)
        st.markdown("**How this application runs it**")
        st.write(model_info["app"])
        st.markdown("**How to read the result**")
        st.write(model_info["interpret"])
        st.markdown("**Important limitations**")
        st.warning(model_info["caution"])

    with theory_tabs[2]:
        ai_pick=st.selectbox("Select an AI regression method to study",["Random Forest","Gradient Boosting"],key="theory_ai_model")
        if ai_pick=="Random Forest":
            st.markdown("### Random Forest Regression")
            st.write("Random Forest is an ensemble of many decision trees. Each tree repeatedly partitions the predictor space into regions and assigns a prediction within each terminal region. Randomness is introduced through resampled observations and random subsets of candidate predictors; the final regression prediction aggregates the trees.")
            st.markdown("**Conceptual flow**")
            st.markdown('<div class="formula">Data → bootstrap/resampled samples → many decision trees → nonlinear splits + interactions → average tree predictions → final prediction</div>',unsafe_allow_html=True)
            st.markdown("**What one tree is doing**")
            st.write("At each split the tree searches for a rule such as X₁ < c that reduces within-node prediction error. The tree continues until its stopping rules are reached. The forest averages many such trees, reducing the instability of a single tree.")
            st.markdown("**How this application runs it**")
            st.write("The app preprocesses the selected X variables, trains the requested number of trees on the training data, predicts the test set, performs cross-validation, and reports test and full-data metrics. Feature importance can be inspected in Chart Studio.")
            st.markdown("**Strengths and cautions**")
            st.write("It can capture nonlinear effects and interactions without the analyst specifying each interaction term. It can still overfit under inappropriate settings, and tree-based feature importance should not be interpreted as a causal effect or a regression coefficient.")
        else:
            st.markdown("### Gradient Boosting Regression")
            st.write("Gradient Boosting builds an additive ensemble sequentially. The first model gives an initial prediction; each new tree is fitted to the remaining error signal, and the ensemble is updated by a learning rate.")
            st.markdown("**Conceptual flow**")
            st.markdown('<div class="formula">Initial prediction → calculate errors → fit correction tree → shrink correction → update prediction → repeat → final prediction</div>',unsafe_allow_html=True)
            st.markdown("**Why it is powerful**")
            st.write("By repeatedly correcting previous mistakes, the ensemble can represent complex nonlinear relationships and interactions. The number of trees, tree depth and learning rate jointly control flexibility.")
            st.markdown("**How this application runs it**")
            st.write("The app fits GradientBoostingRegressor with the configured tree count, learning rate and depth settings, evaluates it on the test set, runs cross-validation, and exposes feature importance in Chart Studio.")
            st.markdown("**Cautions**")
            st.write("Boosting can fit noise when too flexible. Cross-validation and the untouched test set are therefore important. Feature importance describes predictive contribution within the fitted ensemble; it is not a causal effect size.")

    with theory_tabs[3]:
        st.markdown("### Residual-learning hybrid regression")
        st.latex(r"Y=m(X)+\varepsilon")
        st.latex(r"\hat m_H(X)=\hat g(X)+\hat h(X)")
        st.write("Here the statistical model provides the structured base prediction g(X), while an AI residual learner h(X) attempts to explain predictable structure that the statistical model has not captured.")
        st.markdown("**Step-by-step hybrid process**")
        hybrid_steps=[
            "Fit the statistical base model inside training folds.",
            "Generate out-of-fold predictions for each training observation.",
            "Calculate out-of-fold residuals: observed Y minus out-of-fold base prediction.",
            "Fit the AI residual learner to those residuals.",
            "Refit the statistical base model on the complete training data.",
            "For new observations, calculate base prediction + AI residual correction.",
            "Evaluate the hybrid only on the untouched test set.",
        ]
        for i,step in enumerate(hybrid_steps,1): st.markdown(f"**{i}.** {step}")
        st.markdown("**Why out-of-fold residuals matter**")
        st.write("Training an AI model on residuals calculated from predictions generated on the same observations used to fit the base model can leak overly optimistic structure into the residual learner. Out-of-fold residuals are designed to reduce that problem by making each residual correspond to a base prediction produced without using that observation for fitting.")
        st.markdown("**When hybridisation can help**")
        st.write("It can help when the statistical component captures a meaningful interpretable structure but leaves residual patterns that are systematic and predictable. If the base model already captures the systematic signal or the remaining residual is mostly irreducible noise, an AI correction may add little or may overfit.")
        st.markdown("**How the app evaluates it**")
        st.write("The hybrid is compared with the base statistical model using the same held-out test set. The important quantity is the out-of-sample change in error, not the improvement obtained by fitting the hybrid to the same observations used for evaluation.")

    with theory_tabs[4]:
        st.markdown("### Prediction metrics")
        st.latex(r"RMSE=\sqrt{\frac{1}{n}\sum_i(Y_i-\hat Y_i)^2}")
        st.latex(r"MAE=\frac{1}{n}\sum_i|Y_i-\hat Y_i|")
        st.latex(r"R^2=1-\frac{\sum_i(Y_i-\hat Y_i)^2}{\sum_i(Y_i-\bar Y)^2}")
        st.write("RMSE penalises large errors more strongly than MAE. MAE is in the same units as Y and is easier to interpret as average absolute error. R² is a relative fit measure and should always be interpreted on the same evaluation sample when comparing models.")
        st.markdown("### Cross-validation")
        st.write("K-fold cross-validation repeatedly fits a model on K−1 folds and evaluates it on the remaining fold. The app reports the mean cross-validation error and its fold-to-fold variation.")
        st.markdown("### Final test set")
        st.write("The final test set is intentionally excluded from model fitting and cross-validation. It is the most direct internal estimate in this app of how the selected analysis behaves on unseen observations from the same data-generating population.")
        st.markdown("### Inference versus prediction")
        st.write("OLS coefficients, standard errors, p-values and confidence intervals answer inferential questions under the model assumptions. Test-set RMSE/MAE/R² answer predictive questions. A model can have statistically important coefficients without producing the best out-of-sample predictions, and a predictive model can be useful without providing conventional coefficient inference.")
        st.markdown("### Overfitting")
        st.write("A large training performance with materially weaker test performance is a warning sign. Flexible models should therefore be compared using cross-validation and an untouched test set rather than training fit alone.")

# Central analysis control — shown only on the Data & EDA tab
with tabs[1]:
    st.markdown("### 🚀 Run analysis")
    st.caption("Choose the dependent variable (Y) and independent variables (X) above, then run the analysis. Other tabs only display the resulting analysis; they do not contain another Run button.")
    run_clicked=st.button("🚀 Run / Refresh complete analysis",type="primary",use_container_width=True,key="run_analysis_main")

if run_clicked:
    st.session_state["analysis_ready"]=True

if not st.session_state.get("analysis_ready",False):
    with tabs[0]:
        st.info("Choose Y and X in **Data & EDA**, then click **Run / Refresh complete analysis**. Results tabs will populate after the analysis is run.")
    with tabs[2]: st.info("Run the analysis first from **Data & EDA**.")
    with tabs[4]: st.info("Run the analysis first from **Data & EDA**.")
    with tabs[5]: st.info("Run the analysis first from **Data & EDA**.")
    with tabs[6]: st.info("Run the analysis first from **Data & EDA**.")
    with tabs[7]: st.info("Run the analysis first from **Data & EDA**.")
    with tabs[8]: st.info("Run the analysis first from **Data & EDA**. Dataset charts remain available in the Chart Studio after analysis is run.")
    with tabs[9]: st.info("Run the analysis first from **Data & EDA**.")
    st.stop()

try:
    model_df=df[[y_col]+x_cols].copy()
    model_df[y_col]=pd.to_numeric(model_df[y_col],errors="coerce")
    model_df=model_df.replace([np.inf,-np.inf],np.nan).dropna(subset=[y_col]).reset_index(drop=True)
    X=model_df[x_cols].copy(); y=model_df[y_col].copy()
    # Small datasets are allowed to run. They receive a clear warning rather than being blocked.
    small_data_threshold = max(30, 5 * (len(x_cols) + 1))
    if len(y) < small_data_threshold:
        st.warning(
            f"⚠️ Small sample warning: only {len(y):,} usable observations are available. "
            f"The analysis will still run, but estimates, cross-validation results, AI models, "
            f"and statistical inference may be unstable. Interpret the results cautiously and "
            f"consider collecting more observations."
        )
    if len(y) < 4:
        raise ValueError(
            "At least 4 usable observations are required to create a train/test split and run the analysis."
        )
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed)
    specs=model_specs(X,rf_trees,gb_trees,degree,seed)
    preds={}; train_preds={}; rows=[]; cvrows=[]; failures=[]
    for name,est in specs.items():
        try:
            fit=make_clone(est); fit.fit(Xtr,ytr); pt=fit.predict(Xte); ptr=fit.predict(Xtr); preds[name]=pt; train_preds[name]=ptr
            mt=metrics(yte,pt); mtr=metrics(ytr,ptr); vals=[]
            effective_folds = min(folds, len(ytr))
            if effective_folds < 2:
                raise ValueError("At least 2 training observations are required for cross-validation.")
            kf=KFold(n_splits=effective_folds,shuffle=True,random_state=seed)
            for ti,vi in kf.split(Xtr):
                fm=make_clone(est); fm.fit(Xtr.iloc[ti],ytr.iloc[ti]); vals.append(metrics(ytr.iloc[vi],fm.predict(Xtr.iloc[vi])))
            rows.append({"Model":name,"Type":"AI" if name in ["Random Forest","Gradient Boosting"] else "Statistical","Train RMSE":mtr["RMSE"],"Train MAE":mtr["MAE"],"Train R²":mtr["R²"],"RMSE":mt["RMSE"],"MAE":mt["MAE"],"R²":mt["R²"]})
            cvrows.append({"Model":name,"CV RMSE":np.mean([v["RMSE"] for v in vals]),"CV RMSE SD":np.std([v["RMSE"] for v in vals],ddof=1),"CV MAE":np.mean([v["MAE"] for v in vals]),"CV MAE SD":np.std([v["MAE"] for v in vals],ddof=1),"CV R²":np.mean([v["R²"] for v in vals]),"CV R² SD":np.std([v["R²"] for v in vals],ddof=1)})
        except Exception as e:
            failures.append(f"{name}: {e}")
    base=make_pipe(LinearRegression(),X); residual=make_pipe(GradientBoostingRegressor(n_estimators=gb_trees,learning_rate=.05,max_depth=3,random_state=seed),X)
    try:
        base_fit,res_fit,hyb_pred,oof=hybrid_fit(base,residual,Xtr,ytr,Xte,folds,seed)
        hyb_train=base_fit.predict(Xtr)+res_fit.predict(Xtr); preds["Linear + AI Residual Hybrid"]=hyb_pred; train_preds["Linear + AI Residual Hybrid"]=hyb_train
        mt=metrics(yte,hyb_pred); mtr=metrics(ytr,hyb_train)
        rows.append({"Model":"Linear + AI Residual Hybrid","Type":"Hybrid","Train RMSE":mtr["RMSE"],"Train MAE":mtr["MAE"],"Train R²":mtr["R²"],"RMSE":mt["RMSE"],"MAE":mt["MAE"],"R²":mt["R²"]})
        cvrows.append({"Model":"Linear + AI Residual Hybrid","CV RMSE":np.sqrt(np.mean(oof**2)),"CV RMSE SD":np.nan,"CV MAE":np.mean(np.abs(oof)),"CV MAE SD":np.nan,"CV R²":1-np.sum(oof**2)/np.sum((ytr-ytr.mean())**2),"CV R² SD":np.nan})
    except Exception as e:
        failures.append(f"Linear + AI Residual Hybrid: {e}")
        oof=np.array([]); hyb_pred=None
    if not rows: raise RuntimeError("No model could be fitted. Check the selected variables and data types.")
    comparison=pd.DataFrame(rows).merge(pd.DataFrame(cvrows),on="Model",how="left")
    comparison["Generalization Gap R²"]=comparison["Train R²"]-comparison["R²"]
    comparison["Test RMSE Rank"]=rank(comparison["RMSE"],True); comparison["Test MAE Rank"]=rank(comparison["MAE"],True); comparison["Test R² Rank"]=rank(comparison["R²"],False); comparison["Average Test Rank"]=comparison[["Test RMSE Rank","Test MAE Rank","Test R² Rank"]].mean(axis=1)
    comparison=comparison.sort_values("RMSE").reset_index(drop=True)
    if selection_mode=="Choose manually":
        manual=st.sidebar.selectbox("Manual final model",comparison["Model"].tolist()); selected_model=manual
    elif criterion=="Cross-validation RMSE": selected_model=comparison.loc[comparison["CV RMSE"].idxmin(),"Model"]
    elif criterion=="Cross-validation MAE": selected_model=comparison.loc[comparison["CV MAE"].idxmin(),"Model"]
    elif criterion=="Cross-validation R²": selected_model=comparison.loc[comparison["CV R²"].idxmax(),"Model"]
    elif criterion=="Test RMSE": selected_model=comparison.loc[comparison["RMSE"].idxmin(),"Model"]
    elif criterion=="Test MAE": selected_model=comparison.loc[comparison["MAE"].idxmin(),"Model"]
    else: selected_model=comparison.loc[comparison["R²"].idxmax(),"Model"]
    selrow=comparison[comparison["Model"]==selected_model].iloc[0]

    # Full-data models
    full_rows=[]; full_preds={}
    for name,est in specs.items():
        try:
            ff=make_clone(est); ff.fit(X,y); pp=ff.predict(X); full_preds[name]=pp; mm=metrics(y,pp); full_rows.append({"Model":name,"Type":"AI" if name in ["Random Forest","Gradient Boosting"] else "Statistical","RMSE":mm["RMSE"],"MAE":mm["MAE"],"R²":mm["R²"]})
        except Exception as e: failures.append(f"Full-data {name}: {e}")
    try:
        fb,fr,fp,full_oof=hybrid_fit(base,residual,X,y,X,folds,seed); full_preds["Linear + AI Residual Hybrid"]=fp; mm=metrics(y,fp); full_rows.append({"Model":"Linear + AI Residual Hybrid","Type":"Hybrid","RMSE":mm["RMSE"],"MAE":mm["MAE"],"R²":mm["R²"]})
    except Exception as e: failures.append(f"Full-data hybrid: {e}")
    full_comparison=pd.DataFrame(full_rows).sort_values("RMSE").reset_index(drop=True)

    # OLS diagnostics
    Xdiag=build_ols_data(X)
    osm=sm.OLS(y,sm.add_constant(Xdiag.astype(float),has_constant="add")).fit() if not Xdiag.empty else None
    if osm is not None:
        conf=osm.conf_int(); coef=pd.DataFrame({"Variable":osm.params.index,"Coefficient":osm.params.values,"Std. Error":osm.bse.values,"t":osm.tvalues.values,"p-value":osm.pvalues.values,"95% CI Lower":conf[0].values,"95% CI Upper":conf[1].values})
        sst=float(np.sum((y-y.mean())**2)); ssr=float(np.sum(osm.resid**2)); anova=pd.DataFrame({"Source":["Regression","Residual","Total"],"Sum of Squares":[sst-ssr,ssr,sst],"df":[osm.df_model,osm.df_resid,osm.df_model+osm.df_resid]})
        try: bp_p=float(het_breuschpagan(osm.resid,osm.model.exog)[1])
        except Exception: bp_p=np.nan
        vif_rows=[]
        if Xdiag.shape[1]>=2:
            for i,c in enumerate(Xdiag.columns):
                try: vv=float(variance_inflation_factor(Xdiag.astype(float).values,i))
                except Exception: vv=np.inf
                vif_rows.append({"Variable":c,"VIF":vv,"Interpretation":"Very high" if vv>=10 else ("High / potentially problematic" if vv>=5 else "No strong VIF indication")})
        vif=pd.DataFrame(vif_rows)
    else:
        coef=anova=vif=pd.DataFrame(); bp_p=np.nan
except Exception as e:
    st.error(f"Analysis could not be completed: {e}")
    st.exception(e)
    st.stop()

if failures:
    st.warning("Some model calculations were skipped. See details below.")
    with st.expander("Technical model messages"):
        for f in failures: st.write(f)

# Full data results
with tabs[2]:
    st.subheader("📊 Full Data Results")
    st.caption("These are in-sample results: each model is fitted using all usable observations and evaluated on those same observations. They are not independent test-set estimates.")
    if osm is not None:
        a,b,c,d=st.columns(4); a.metric("OLS R²",f"{osm.rsquared:.4f}"); b.metric("Adjusted R²",f"{osm.rsquared_adj:.4f}"); c.metric("F-statistic",f"{osm.fvalue:.3f}"); d.metric("Model p-value",f"{osm.f_pvalue:.3g}")
    st.dataframe(full_comparison,use_container_width=True,hide_index=True)
    fig=px.bar(full_comparison,x="Model",y="R²",color="Type",title="Full-data R² comparison",text_auto=".3f"); fig.update_layout(template="plotly_white",height=500,xaxis_tickangle=-35); st.plotly_chart(fig,use_container_width=True)
    if not coef.empty:
        with st.expander("Full-data OLS coefficients",expanded=True): st.dataframe(coef,use_container_width=True,hide_index=True)

# Predictive results
with tabs[4]:
    st.subheader("📈 Predictive Results")
    a,b,c,d=st.columns(4); a.metric("Selected model",selected_model); b.metric("Test RMSE",f"{float(selrow['RMSE']):.4f}"); c.metric("Test MAE",f"{float(selrow['MAE']):.4f}"); d.metric("Test R²",f"{float(selrow['R²']):.4f}")
    st.caption(f"Selection criterion: {criterion}. Training observations: {len(ytr):,}; untouched test observations: {len(yte):,}.")
    st.dataframe(comparison,use_container_width=True,hide_index=True)
    fig=px.bar(comparison.sort_values("RMSE"),x="Model",y="RMSE",color="Type",title="Test RMSE by model",text_auto=".3f"); fig.update_layout(template="plotly_white",height=500,xaxis_tickangle=-35); st.plotly_chart(fig,use_container_width=True)
    pred_long=pd.DataFrame({"Actual":np.tile(yte.to_numpy(),len(preds)),"Predicted":np.concatenate(list(preds.values())),"Model":np.repeat(list(preds.keys()),len(yte))})
    fig=px.scatter(pred_long,x="Actual",y="Predicted",facet_col="Model",facet_col_wrap=2,title="Hold-out test predictions"); mn=min(pred_long.Actual.min(),pred_long.Predicted.min()); mx=max(pred_long.Actual.max(),pred_long.Predicted.max()); fig.add_shape(type="line",x0=mn,x1=mx,y0=mn,y1=mx,line_dash="dash",row="all",col="all"); fig.update_layout(template="plotly_white",height=850); st.plotly_chart(fig,use_container_width=True)

with tabs[5]:
    st.subheader("🔬 Hybrid Analysis")
    if "Linear Regression" in comparison.Model.values and "Linear + AI Residual Hybrid" in comparison.Model.values:
        lin=comparison[comparison.Model=="Linear Regression"].iloc[0]; hyb=comparison[comparison.Model=="Linear + AI Residual Hybrid"].iloc[0]; imp=float(lin["RMSE"]-hyb["RMSE"]); pct=100*imp/float(lin["RMSE"])
        a,b,c=st.columns(3); a.metric("Linear test RMSE",f"{float(lin['RMSE']):.4f}"); b.metric("Hybrid test RMSE",f"{float(hyb['RMSE']):.4f}"); c.metric("Hybrid RMSE change",f"{pct:+.2f}%")
        st.write(f"RMSE difference (Linear − Hybrid): **{imp:.6f}**. Positive means lower hybrid error on this test split.")
    if len(oof):
        a,b,c=st.columns(3); a.metric("OOF residual SD",f"{np.std(oof):.4f}"); b.metric("OOF residual mean",f"{np.mean(oof):.4f}"); c.metric("OOF residual MAE",f"{np.mean(np.abs(oof)):.4f}")
        st.plotly_chart(px.histogram(x=oof,nbins=30,title="Out-of-fold statistical residuals"),use_container_width=True)
    st.markdown("### How to interpret the hybrid")
    st.write("The hybrid is useful only when the residuals from the statistical base model contain predictable, generalisable structure. A lower test error is evidence for incremental value under this validation design; it is not a universal claim about hybrid models.")

with tabs[6]:
    st.subheader("🩺 Diagnostics & Statistical Inference")
    if osm is None:
        st.warning("OLS diagnostics could not be calculated for this specification.")
    else:
        a,b,c,d,e=st.columns(5); a.metric("Full-data OLS R²",f"{osm.rsquared:.4f}"); b.metric("Adjusted R²",f"{osm.rsquared_adj:.4f}"); c.metric("F-statistic",f"{osm.fvalue:.3f}"); d.metric("Model p-value",f"{osm.f_pvalue:.3g}"); e.metric("OLS observations",f"{int(osm.nobs):,}")
        st.markdown("### OLS coefficients"); st.dataframe(coef,use_container_width=True,hide_index=True)
        with st.expander("ANOVA"): st.dataframe(anova,use_container_width=True,hide_index=True)
        with st.expander("Multicollinearity / VIF",expanded=True):
            if not vif.empty: st.dataframe(vif,use_container_width=True,hide_index=True)
            else: st.info("VIF requires at least two usable predictor columns.")
        st.markdown("### Heteroscedasticity"); st.metric("Breusch–Pagan p-value",f"{bp_p:.4g}" if np.isfinite(bp_p) else "N/A")
        st.caption("The Breusch–Pagan test is a diagnostic for non-constant error variance; it is not a complete validity test.")
        st.markdown("### OLS vs predictive evaluation"); st.write(f"Full-data OLS uses {len(y):,} usable observations for inference. Predictive evaluation fits models on {len(ytr):,} training observations and evaluates them on {len(yte):,} untouched test observations. Different R² values are therefore expected.")

with tabs[7]:
    st.subheader("💡 Results Interpretation")
    best_test=comparison.loc[comparison.RMSE.idxmin()]; best_cv=comparison.loc[comparison["CV RMSE"].idxmin()]
    st.markdown("### Predictive results"); st.write(f"The current test split reports the lowest RMSE for **{best_test.Model}** at **{best_test.RMSE:.4f}**. The lowest mean cross-validation RMSE is reported for **{best_cv.Model}** at **{best_cv['CV RMSE']:.4f}**. These are descriptive results for the current dataset and validation design, not universal algorithm rankings.")
    st.markdown("### Statistical inference");
    if osm is not None: st.write(f"The full-data OLS model has R² **{osm.rsquared:.4f}** and adjusted R² **{osm.rsquared_adj:.4f}**. Coefficient p-values describe evidence against zero conditional effects under the model assumptions; they do not establish causality or superior predictive performance.")
    st.markdown("### Diagnostics");
    if not vif.empty:
        high=vif[vif.VIF>=5]["Variable"].tolist(); st.write("Potential multicollinearity is flagged for: **"+", ".join(high)+"**." if high else "No predictor has VIF ≥ 5 in this specification.")
    st.markdown("### Research conclusion"); st.write(f"The analysis compares statistical, regularised, AI and hybrid models. The selected model under **{criterion}** is **{selected_model}**. Conclusions should report full-data inference, cross-validation, untouched-test performance, hybrid incremental value and diagnostics separately.")

with tabs[8]:
    st.subheader("🎨 Advanced Chart Studio")
    st.caption("Use this workspace for research-quality exploratory plots, regression diagnostics, model comparison visuals and publication-ready inspection. Colour controls are provided where useful.")
    chart_tabs=st.tabs(["🔎 Advanced Data Explorer","📉 Model Diagnostics","📊 Research Comparison"])

    with chart_tabs[0]:
        st.markdown("### Advanced exploratory data visualisation")
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cats=[c for c in df.columns if c not in nums]
        explorer_type=st.selectbox("Choose data plot",[
            "Distribution + box","Scatter with trend","Scatter matrix","Violin plot","Grouped box plot",
            "ECDF / cumulative distribution","Missing-value profile","Correlation heatmap"
        ],key="advanced_data_plot")
        fig=None

        if explorer_type=="Distribution + box":
            v=st.selectbox("Numeric variable",nums,key="dist_var")
            c1,c2,c3=st.columns(3)
            hist_colour=c1.color_picker("Histogram colour","#2f80ed",key="hist_colour")
            box_colour=c2.color_picker("Box colour","#f2994a",key="box_colour")
            bins=c3.slider("Number of bins",10,80,30,key="hist_bins")
            fig=go.Figure()
            fig.add_trace(go.Histogram(x=df[v].dropna(),nbinsx=bins,name="Distribution",marker_color=hist_colour,opacity=.80))
            fig.add_trace(go.Box(x=df[v].dropna(),name="Box summary",marker_color=box_colour,boxpoints="outliers",orientation="h",yaxis="y2"))
            fig.update_layout(title=f"Distribution and box summary — {v}",xaxis_title=v,yaxis=dict(title="Frequency"),yaxis2=dict(title="Box summary",overlaying="y",side="right",showticklabels=False),barmode="overlay")

        elif explorer_type=="Scatter with trend":
            c1,c2=st.columns(2)
            xx=c1.selectbox("X variable",nums,key="adv_scatter_x")
            yy=c2.selectbox("Y variable",nums,index=min(1,len(nums)-1),key="adv_scatter_y")
            colour_mode=st.selectbox("Point colouring",["Single colour","Categorical group","Continuous variable"],key="adv_scatter_colour_mode")
            pts_color=st.color_picker("Point colour","#176b87",key="adv_scatter_point") if colour_mode=="Single colour" else None
            group=None
            if colour_mode=="Categorical group" and cats:
                group=st.selectbox("Group",cats,key="adv_scatter_group")
            elif colour_mode=="Continuous variable":
                group=st.selectbox("Colour variable",nums,key="adv_scatter_cont")
            temp=df[[xx,yy]+([] if group is None else [group])].dropna()
            if colour_mode=="Single colour":
                fig=px.scatter(temp,x=xx,y=yy,trendline="ols" if len(temp)>=10 else None,title=f"{yy} vs {xx}",color_discrete_sequence=[pts_color])
            else:
                fig=px.scatter(temp,x=xx,y=yy,color=group,trendline="ols" if len(temp)>=10 else None,title=f"{yy} vs {xx}")

        elif explorer_type=="Scatter matrix":
            selected=st.multiselect("Select up to 6 numeric variables",nums,default=nums[:min(4,len(nums))],max_selections=6,key="scatter_matrix_vars")
            colour_var=st.selectbox("Optional colour/group variable",["None"]+cats,key="scatter_matrix_colour")
            if len(selected)>=2:
                smdf=df[selected+([] if colour_var=="None" else [colour_var])].dropna().head(1500)
                fig=px.scatter_matrix(smdf,dimensions=selected,color=None if colour_var=="None" else colour_var,title="Scatter matrix of selected variables",height=900)
            else:
                st.info("Select at least two numeric variables.")

        elif explorer_type=="Violin plot":
            c1,c2=st.columns(2)
            v=c1.selectbox("Numeric variable",nums,key="violin_var")
            g=c2.selectbox("Group",["None"]+cats,key="violin_group")
            fig=px.violin(df,y=v,x=None if g=="None" else g,box=True,points="outliers",title=f"Violin distribution — {v}")

        elif explorer_type=="Grouped box plot":
            c1,c2=st.columns(2)
            v=c1.selectbox("Numeric variable",nums,key="gbox_var")
            g=c2.selectbox("Grouping variable",cats or df.columns.tolist(),key="gbox_group")
            temp=df[[g,v]].dropna()
            fig=px.box(temp,x=g,y=v,points="outliers",title=f"{v} by {g}")

        elif explorer_type=="ECDF / cumulative distribution":
            v=st.selectbox("Numeric variable",nums,key="ecdf_var")
            fig=px.ecdf(df.dropna(subset=[v]),x=v,title=f"Empirical cumulative distribution — {v}")

        elif explorer_type=="Missing-value profile":
            miss=df.isna().sum().sort_values(ascending=False)
            miss=miss[miss>0]
            if miss.empty:
                st.success("No missing values were detected in the uploaded dataset.")
            else:
                pct=(100*miss/len(df)).round(2)
                mp=pd.DataFrame({"Variable":miss.index,"Missing count":miss.values,"Missing %":pct.values})
                fig=px.bar(mp,x="Variable",y="Missing %",text="Missing %",title="Missing-value profile",hover_data=["Missing count"])
                fig.update_traces(texttemplate="%{text:.2f}%")

        else:
            corr=df[nums].corr(numeric_only=True)
            if corr.shape[0]>=2:
                zmin=float(np.nanmin(corr.values)); zmax=float(np.nanmax(corr.values))
                fig=px.imshow(corr,text_auto=".2f",aspect="auto",title="Pearson correlation heatmap",color_continuous_scale="RdBu_r",zmin=-1,zmax=1)
            else:
                st.info("At least two numeric variables are required for a correlation heatmap.")

        if fig is not None:
            fig.update_layout(template="plotly_white",height=620,margin=dict(l=30,r=30,t=80,b=50),hovermode="closest")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Exploratory plots describe patterns in the observed data. They do not by themselves establish causation.")

    with chart_tabs[1]:
        st.markdown("### Regression and prediction diagnostics")
        diag_chart=st.selectbox("Choose diagnostic plot",[
            "Actual vs predicted — points and line","Actual vs predicted — identity plot","Residuals vs predicted",
            "Residual distribution","Prediction error by model","Prediction error by observation","Feature importance"
        ],key="advanced_diag_plot")
        fig=None
        model_options=list(preds.keys())

        if diag_chart=="Actual vs predicted — points and line":
            model_for_chart=st.selectbox("Model",model_options,key="adv_actual_pred_model")
            c1,c2,c3=st.columns(3)
            actual_colour=c1.color_picker("Actual point colour","#2ca02c",key="adv_actual_colour")
            predicted_colour=c2.color_picker("Predicted line colour","#1f77b4",key="adv_pred_colour")
            line_width=c3.slider("Predicted line width",1,8,3,key="adv_pred_width")
            actual_values=yte.to_numpy(dtype=float)
            predicted_values=np.asarray(preds[model_for_chart],dtype=float)
            obs=np.arange(1,len(actual_values)+1)
            temp=pd.DataFrame({"Observation":obs,"Actual":actual_values,"Predicted":predicted_values})
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=temp["Observation"],y=temp["Actual"],mode="markers",name="Actual — observed (points)",marker=dict(color=actual_colour,size=8),customdata=temp[["Actual","Predicted"]],hovertemplate="Obs %{x}<br>Actual: %{customdata[0]:.4f}<br>Predicted: %{customdata[1]:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=temp["Observation"],y=temp["Predicted"],mode="lines",name=f"Predicted — {model_for_chart} (line)",line=dict(color=predicted_colour,width=line_width),customdata=temp[["Actual","Predicted"]],hovertemplate="Obs %{x}<br>Actual: %{customdata[0]:.4f}<br>Predicted: %{customdata[1]:.4f}<extra></extra>"))
            fig.update_layout(title=f"Actual vs Predicted — {model_for_chart}",xaxis_title="Test observation order",yaxis_title=y_col,legend_title="Series",hovermode="x unified")
            st.info("Actual = observed points. Predicted = model-estimated line. The line follows test-set observation order; it is not a regression trendline or a fitted line through the points.")

        elif diag_chart=="Actual vs predicted — identity plot":
            model_for_chart=st.selectbox("Model",model_options,key="adv_identity_model")
            c1,c2=st.columns(2)
            actual_colour=c1.color_picker("Actual point colour","#176b87",key="identity_point_colour")
            identity_colour=c2.color_picker("1:1 reference line colour","#d62728",key="identity_line_colour")
            temp=pd.DataFrame({"Actual":yte.to_numpy(dtype=float),"Predicted":np.asarray(preds[model_for_chart],dtype=float)})
            fig=go.Figure(go.Scatter(x=temp["Actual"],y=temp["Predicted"],mode="markers",name="Test observations",marker=dict(color=actual_colour,size=8,opacity=.75)))
            mn=float(min(temp.min().min(),temp.min().min())); mx=float(max(temp.max().max(),temp.max().max()))
            fig.add_shape(type="line",x0=mn,x1=mx,y0=mn,y1=mx,line=dict(color=identity_colour,width=3,dash="dash"))
            fig.add_annotation(x=mx,y=mx,text="Perfect prediction: y = ŷ",showarrow=False,xanchor="right",yanchor="bottom")
            fig.update_layout(title=f"Actual vs predicted identity plot — {model_for_chart}",xaxis_title="Actual Y",yaxis_title="Predicted Ŷ")
            st.info("In this version, the diagonal 1:1 line is the perfect-prediction reference. Points close to the diagonal have smaller prediction error.")

        elif diag_chart=="Residuals vs predicted":
            model_for_chart=st.selectbox("Model",model_options,key="adv_resid_model")
            residual=yte.to_numpy()-preds[model_for_chart]
            temp=pd.DataFrame({"Predicted":preds[model_for_chart],"Residual":residual})
            point_colour=st.color_picker("Residual point colour","#9467bd",key="adv_resid_colour")
            fig=px.scatter(temp,x="Predicted",y="Residual",title=f"Residuals vs predicted — {model_for_chart}",color_discrete_sequence=[point_colour])
            fig.add_hline(y=0,line_dash="dash",line_width=2)
            st.info("Look for an approximately structureless cloud around zero. Curvature, funnels or clusters can indicate remaining structure or changing error variance.")

        elif diag_chart=="Residual distribution":
            model_for_chart=st.selectbox("Model",model_options,key="adv_resid_hist_model")
            residual=yte.to_numpy()-preds[model_for_chart]
            hist_colour=st.color_picker("Residual histogram colour","#2ca02c",key="adv_resid_hist_colour")
            fig=go.Figure(go.Histogram(x=residual,nbinsx=30,name="Residuals",marker_color=hist_colour))
            fig.update_layout(title=f"Residual distribution — {model_for_chart}",xaxis_title="Residual = Actual − Predicted",yaxis_title="Count")

        elif diag_chart=="Prediction error by model":
            err=pd.DataFrame({m:yte.to_numpy()-p for m,p in preds.items()})
            lm=err.melt(var_name="Model",value_name="Error")
            fig=px.box(lm,x="Model",y="Error",points="outliers",title="Test prediction error by model")
            fig.add_hline(y=0,line_dash="dash")
            fig.update_xaxes(tickangle=-35)

        elif diag_chart=="Prediction error by observation":
            model_for_chart=st.selectbox("Model",model_options,key="adv_error_obs_model")
            actual=yte.to_numpy(); pred=np.asarray(preds[model_for_chart]); err=actual-pred
            fig=go.Figure()
            fig.add_trace(go.Bar(x=np.arange(1,len(err)+1),y=err,name="Prediction error",marker_color="#17becf"))
            fig.add_hline(y=0,line_dash="dash")
            fig.update_layout(title=f"Prediction error by test observation — {model_for_chart}",xaxis_title="Test observation order",yaxis_title="Actual − Predicted")

        else:
            model_for_chart=st.selectbox("AI model",[m for m in ["Random Forest","Gradient Boosting"] if m in specs],key="adv_importance_model")
            ai_fit=make_clone(specs[model_for_chart]); ai_fit.fit(Xtr,ytr)
            prep=ai_fit.named_steps["prep"]; mdl=ai_fit.named_steps["model"]
            names=list(prep.get_feature_names_out()); vals=getattr(mdl,"feature_importances_",None)
            if vals is not None:
                fi=pd.DataFrame({"Feature":names,"Importance":vals}).sort_values("Importance",ascending=False).head(25)
                fig=px.bar(fi.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title=f"Top feature importances — {model_for_chart}",text_auto=".3f")
            else:
                st.info("Feature importance is not available for this model.")

        if fig is not None:
            fig.update_layout(template="plotly_white",height=640,margin=dict(l=30,r=30,t=80,b=50),hovermode="closest")
            st.plotly_chart(fig,use_container_width=True)

    with chart_tabs[2]:
        st.markdown("### Research comparison plots")
        research_plot=st.selectbox("Choose comparison visual",[
            "Model metric heatmap","Test RMSE with CV RMSE","Test R² by model","Train vs Test R²","CV RMSE with uncertainty","Full-data vs Test R²"
        ],key="research_comparison_plot")
        fig=None
        if research_plot=="Model metric heatmap":
            metric_cols=[c for c in ["RMSE","MAE","R²","CV RMSE"] if c in comparison.columns]
            hm=comparison.set_index("Model")[metric_cols]
            fig=px.imshow(hm,text_auto=".3f",aspect="auto",title="Model performance metric heatmap",color_continuous_scale="RdBu_r")

        elif research_plot=="Test RMSE with CV RMSE":
            p=comparison.copy()
            fig=go.Figure()
            fig.add_trace(go.Bar(x=p["Model"],y=p["RMSE"],name="Test RMSE"))
            fig.add_trace(go.Scatter(x=p["Model"],y=p["CV RMSE"],name="Mean CV RMSE",mode="lines+markers"))
            fig.update_layout(title="Test RMSE vs mean cross-validation RMSE",xaxis_title="Model",yaxis_title="RMSE")
            fig.update_xaxes(tickangle=-35)

        elif research_plot=="Test R² by model":
            p=comparison.sort_values("R²",ascending=False)
            fig=px.bar(p,x="Model",y="R²",color="Type",text_auto=".3f",title="Test-set R² by model")
            fig.update_xaxes(tickangle=-35)

        elif research_plot=="Train vs Test R²":
            p=comparison.melt(id_vars=["Model","Type"],value_vars=["Train R²","R²"],var_name="Split",value_name="R²")
            p["Split"]=p["Split"].replace({"Train R²":"Training","R²":"Test"})
            fig=px.bar(p,x="Model",y="R²",color="Split",barmode="group",text_auto=".3f",title="Training vs test R²")
            fig.update_xaxes(tickangle=-35)

        elif research_plot=="CV RMSE with uncertainty":
            p=comparison.sort_values("CV RMSE")
            fig=go.Figure(go.Bar(x=p["Model"],y=p["CV RMSE"],error_y=dict(type="data",array=p["CV RMSE SD"].fillna(0))))
            fig.update_layout(title="Mean cross-validation RMSE ± fold SD",xaxis_title="Model",yaxis_title="CV RMSE")
            fig.update_xaxes(tickangle=-35)

        else:
            merged=full_comparison[["Model","R²"]].merge(comparison[["Model","R²"]],on="Model",suffixes=(" — Full data"," — Test"))
            long=merged.melt("Model",var_name="Evaluation",value_name="R²")
            fig=px.bar(long,x="Model",y="R²",color="Evaluation",barmode="group",text_auto=".3f",title="Full-data vs test-set R²")
            fig.update_xaxes(tickangle=-35)

        if fig is not None:
            fig.update_layout(template="plotly_white",height=640,margin=dict(l=30,r=30,t=80,b=50),hovermode="closest")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Comparison plots are descriptive for the current dataset and validation design. Use the numerical tables alongside the visuals.")

with tabs[9]:
    st.subheader("📄 Report & Export")
    pred_table=Xte.reset_index(drop=True).copy(); pred_table.insert(0,"Actual",yte.reset_index(drop=True))
    for m,p in preds.items(): pred_table[f"Predicted — {m}"]=p
    report=f"""STATISTICAL–AI HYBRID MODELLING PLATFORM\n\nOutcome: {y_col}\nPredictors: {', '.join(x_cols)}\nObservations: {len(y)}\nTraining observations: {len(ytr)}\nTest observations: {len(yte)}\nTest proportion: {test_size}\nRandom seed: {seed}\nCV folds: {folds}\nSelection criterion: {criterion}\nSelected model: {selected_model}\n\nFULL-DATA RESULTS\n{full_comparison.to_string(index=False)}\n\nPREDICTIVE RESULTS\n{comparison.to_string(index=False)}\n\nOLS INFERENCE\n{coef.to_string(index=False) if not coef.empty else 'OLS unavailable'}\n\nINTERPRETATION\nFull-data fit, statistical inference and hold-out prediction are separate analyses. Hybrid incremental value should be assessed out-of-sample and should not be assumed a priori.\n"""
    st.text_area("Report preview",report,height=450)
    c1,c2,c3=st.columns(3); c1.download_button("⬇️ Comparison CSV",comparison.to_csv(index=False).encode(),"model_comparison.csv","text/csv"); c2.download_button("⬇️ Predictions CSV",pred_table.to_csv(index=False).encode(),"test_predictions.csv","text/csv"); c3.download_button("⬇️ Research TXT",report.encode(),"statistical_ai_report.txt","text/plain")
    excel=io.BytesIO()
    with pd.ExcelWriter(excel,engine="openpyxl") as w:
        df.to_excel(w,"Data",index=False); full_comparison.to_excel(w,"Full Data Results",index=False); comparison.to_excel(w,"Model Comparison",index=False); pred_table.to_excel(w,"Test Predictions",index=False); coef.to_excel(w,"OLS Coefficients",index=False); anova.to_excel(w,"ANOVA",index=False); vif.to_excel(w,"VIF",index=False)
    st.download_button("📘 Complete Excel report",excel.getvalue(),"statistical_ai_complete_report.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider(); st.caption("StatAI v14 • Statistical–AI Hybrid Modelling Platform • Research and educational use")
