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

THEORY_SOURCES={
    "Linear Regression": {
        "theory":"https://online.stat.psu.edu/stat501/Lesson01",
        "theory_label":"Penn State STAT 501 — least-squares coefficients",
        "code":"https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html",
        "code_label":"Statsmodels — OLS Python notebook"
    },
    "Polynomial Regression": {
        "theory":"https://docs.originlab.com/origin-help/pr-algorithm/",
        "theory_label":"OriginLab — polynomial least-squares algorithm and coefficient calculation",
        "code":"https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html",
        "code_label":"Scikit-learn — polynomial regression Python example"
    },
    "Ridge Regression": {
        "theory":"https://www2.stat.duke.edu/courses/Fall17/sta721/publication/ridge/",
        "theory_label":"Duke STA 721 — Ridge derivation and shrinkage theory",
        "code":"https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_coeffs.html",
        "code_label":"Scikit-learn — Ridge coefficients Python example"
    },
    "Lasso Regression": {
        "theory":"https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/coordinate_descent.pdf",
        "theory_label":"CMU — Lasso coordinate-descent and soft-threshold derivation",
        "code":"https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html",
        "code_label":"Scikit-learn — Lasso model-selection Python example"
    },
    "Elastic Net": {
        "theory":"https://faculty.cc.gatech.edu/~isbell/reading/papers/elasticnet.pdf",
        "theory_label":"Zou & Hastie — original Elastic Net theory",
        "code":"https://scikit-learn.org/stable/auto_examples/linear_model/plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.html",
        "code_label":"Scikit-learn — Elastic Net Python example"
    },
    "Random Forest": {
        "theory":"https://web.iitd.ac.in/~sumeet/Hastie.pdf",
        "theory_label":"Elements of Statistical Learning — Random Forest theory",
        "code":"https://www.datacamp.com/tutorial/random-forest-regression",
        "code_label":"DataCamp — Random Forest Regression Python example"
    },
    "Gradient Boosting": {
        "theory":"https://scikit-learn.org/stable/modules/ensemble.html",
        "theory_label":"Scikit-learn — Gradient Boosting objective and algorithm",
        "code":"https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html",
        "code_label":"Scikit-learn — Gradient Boosting Python example"
    },
    "Linear + AI Residual Hybrid": {
        "theory":"https://link.springer.com/article/10.1007/s11270-026-09831-4",
        "theory_label":"Springer — out-of-fold residual-correction framework",
        "code":"https://www.google.com/search?q=site%3Agithub.com+python+residual+correction+hybrid+regression+machine+learning",
        "code_label":"GitHub search — Python residual-correction examples"
    },
}


def chart_guidance(title):
    """Beginner-friendly explanation + research caution for a displayed Plotly chart."""
    t=str(title or "").lower()
    if "actual vs predicted" in t and "identity" not in t:
        return (
            "Actual values are the observed outcomes; predicted values are the model's estimates. "
            "Points/lines that stay close to one another indicate smaller prediction errors.",
            "Do not judge a model from the training plot alone. Use the untouched test set for the main predictive interpretation. "
            "A curve or systematic gap can indicate bias or missing structure, but random scatter is not proof of a perfect model.",
            "Compare the graph with RMSE, MAE and test R². Check whether high values are consistently under-predicted or low values over-predicted."
        )
    if "identity" in t:
        return (
            "Each point compares an observed value with its prediction. The diagonal 1:1 line is perfect prediction: predicted = actual.",
            "Distance from the diagonal represents prediction error. A close cloud is desirable, but a visually tight plot on training data can still reflect overfitting.",
            "Inspect the held-out test data, not only training data. Also look for systematic bending or widening away from the line."
        )
    if "residuals vs predicted" in t:
        return (
            "Residual = Actual − Predicted. A roughly random band around zero is compatible with a well-behaved error pattern.",
            "A curved pattern can suggest missing nonlinear structure; a funnel shape can suggest changing error variance; clusters may indicate groups or omitted variables.",
            "Use this plot with formal diagnostics such as Breusch–Pagan and with subject-matter knowledge. A pattern is a warning to investigate, not automatic proof of model failure."
        )
    if "residual distribution" in t or "prediction error distribution" in t:
        return (
            "The plot shows how prediction errors are distributed. A centre near zero means little average signed bias; a wide spread means larger prediction errors.",
            "Extreme tails or strong skewness can indicate outliers or asymmetric error behaviour. Do not assume normality solely because the histogram looks roughly bell-shaped.",
            "Interpret the error scale in the units of Y. Combine the graph with MAE/RMSE and, for OLS inference, Q–Q or formal residual diagnostics when needed."
        )
    if "missing" in t:
        return (
            "Each bar shows how much data are missing for a variable. Larger bars mean more observations are unavailable for that variable.",
            "High missingness can change the effective sample size and may introduce bias if missingness is systematic rather than random.",
            "Investigate why data are missing before simply deleting rows. The warning level depends on the variable, sample size and missing-data mechanism."
        )
    if "correlation" in t:
        return (
            "Correlation summarises the strength and direction of a linear association between numeric variables, from −1 to +1.",
            "Correlation does not establish causation and can miss nonlinear relationships. A high correlation can also indicate multicollinearity when both variables are used as predictors.",
            "Use the heatmap as an exploratory screen, then inspect scatterplots and VIF before making regression claims."
        )
    if "importance" in t:
        return (
            "Feature importance ranks variables by their contribution to the fitted tree ensemble under the chosen importance measure.",
            "Importance is not a regression coefficient and is not a causal effect. Tree importance can also be influenced by variable scale, cardinality and correlated predictors.",
            "Treat importance as a model-specific predictive explanation. For stronger interpretation, compare it with permutation importance or other model-agnostic methods."
        )
    if "rmse" in t and "cv" in t:
        return (
            "Lower RMSE means smaller typical prediction error, with greater penalty for large errors. Cross-validation RMSE summarises performance across training folds.",
            "Small differences can be unstable, especially with small samples. A single split should not be treated as universal evidence.",
            "Look at both the mean and fold-to-fold variation. Compare against the untouched test result before drawing conclusions."
        )
    if "r²" in t or "r2" in t:
        return (
            "R² describes how much variation in the target is explained relative to a mean-prediction reference on the evaluation sample.",
            "R² is not an accuracy percentage and can be negative on unseen data. A high R² does not establish causation or guarantee good performance outside the study population.",
            "Always state whether the value is from full-data, training, cross-validation or test observations."
        )
    if "train vs test" in t or "full-data vs test" in t:
        return (
            "The comparison shows how model performance changes between data used for fitting and data not used for fitting.",
            "A large training–test gap is a classic warning sign of overfitting. A small gap can occur because both performances are modest as well.",
            "Use the gap together with cross-validation, sample size and model complexity rather than using a fixed numerical cutoff."
        )
    if "metric heatmap" in t or "model performance" in t or "test rmse" in t or "model comparison" in t:
        return (
            "The chart compares models using common metrics on the same evaluation design, making relative performance easier to see.",
            "One metric can favour a model that behaves differently on another metric. Do not treat a visual ranking as proof of universal superiority.",
            "Read the numerical table beside the plot and consider uncertainty, validation design, model purpose and interpretability."
        )
    if "box" in t or "violin" in t:
        return (
            "The plot summarises the centre, spread and unusual values of a numeric variable, optionally by group.",
            "A group difference in the plot is descriptive; it does not by itself prove a statistically significant difference or a causal effect.",
            "Look at sample sizes per group and consider confidence intervals or formal tests where appropriate."
        )
    if "histogram" in t or "distribution" in t:
        return (
            "The distribution shows where observations are concentrated and whether the variable is symmetric, skewed or has unusual values.",
            "The appearance depends on bin width and sample size. A histogram alone is not a formal normality test.",
            "Use the distribution to guide preprocessing and model choice, and combine it with summary statistics and subject knowledge."
        )
    if "scatter" in t or "trend" in t:
        return (
            "Each point is an observation. The shape, direction and spread indicate whether the variables appear linearly or nonlinearly related.",
            "Association is not causation. Outliers, hidden groups and changing variance can strongly affect the apparent pattern.",
            "Check sample size, subgroup structure, unusual observations and whether the plotted scale hides important features."
        )
    return (
        "Read the title, axes, legend and units first. Then identify the main pattern, unusual observations and the amount of uncertainty or variation.",
        "A graph is evidence about the displayed sample and model; it is not automatically proof of causality or generalisation.",
        "Use the graph together with the numerical results, validation design and subject-matter context."
    )


def plot_with_guidance(fig):
    """Display a Plotly chart followed by plain-language interpretation, warning and next step."""
    st.plotly_chart(fig,use_container_width=True)
    title=""
    try:
        title=fig.layout.title.text or "this graph"
    except Exception:
        title="this graph"
    interpretation,warning,suggestion=chart_guidance(title)
    with st.expander("🧭 How to interpret this graph",expanded=False):
        st.markdown(f"**What you are seeing:** {interpretation}")
        st.markdown(f"**⚠️ Caution:** {warning}")
        st.markdown(f"**✅ What to check next:** {suggestion}")


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

# Central analysis control — placed immediately below Y/X selection.
st.markdown("### 🚀 Run Analysis")
st.caption("After selecting Y and X, click the button below to run the complete regression, AI and hybrid analysis. You do not need to run it again when moving between result tabs.")
run_clicked = st.button(
    "🚀 Run / Refresh Complete Analysis",
    type="primary",
    use_container_width=True,
    key="run_analysis_main_top"
)
if run_clicked:
    st.session_state["analysis_ready"] = True
    st.session_state["analysis_y"] = y_col
    st.session_state["analysis_x"] = list(x_cols)

# Show tabs after the run control.
tabs=st.tabs(["🏠 Overview","📁 Data & EDA","📊 Full Data Results","📚 Theory & How It Works","📈 Predictive Results","🩺 Diagnostics & Inference","💡 Results Interpretation & Limitations","🎨 Chart Studio","📄 Report & Export"])

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
        eda_fig=px.imshow(df[num_cols].corr(),text_auto=".2f",aspect="auto",title="Numeric correlation matrix",color_continuous_scale="RdBu_r")
        plot_with_guidance(eda_fig)

with tabs[3]:
    st.subheader("📚 Regression Theory & How It Works")
    st.caption("A single theory workspace replaces multiple sub-tabs. Select one topic below; each topic gives the research theory, mathematical estimation, step-by-step calculation, how StatAI runs it, interpretation, limitations and learning links.")

    theory_topic=st.selectbox(
        "📖 Select a topic to study",
        [
            "Complete regression workflow",
            "Linear Regression / OLS",
            "Polynomial Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Elastic Net Regression",
            "Random Forest Regression",
            "Gradient Boosting Regression",
            "Statistical–AI Residual Hybrid",
            "Prediction metrics, validation and inference",
        ],
        key="theory_topic_v17"
    )

    # Keep this defined for the standalone workflow topic so the later topic dispatch
    # never references an undefined variable. Model-specific details are populated below
    # only when a model topic is selected.
    model_details = {}

    def theory_header(title, purpose, formula=None):
        st.markdown(f"## {title}")
        st.info(f"**In simple words:** {purpose}")
        if formula:
            st.latex(formula)

    if theory_topic=="Complete regression workflow":
        theory_header("Complete regression workflow", "The app first understands the data, then estimates models, validates them, tests them on unseen observations, checks diagnostics, and finally explains the results.")
        st.markdown("### Step-by-step research workflow")
        workflow_details=[
            ("1. Define the question","Specify the outcome Y and predictors X. State whether the research goal is explanation/inference, prediction, or both."),
            ("2. Inspect the data","Check size, variable types, missing values, identifiers, duplicates, impossible values and the usable analysis sample."),
            ("3. Explore relationships","Use distributions, scatter plots, group comparisons and correlation to understand the variables before modelling."),
            ("4. Create the modelling data","Keep the chosen variables, remove unusable outcome rows, and preprocess numeric/categorical predictors consistently."),
            ("5. Split training and test data","Fit models on the training portion. Keep the final test set untouched for the main out-of-sample assessment."),
            ("6. Fit statistical models","OLS and polynomial regression provide interpretable functional forms; Ridge, Lasso and Elastic Net add regularisation."),
            ("7. Fit AI models","Random Forest and Gradient Boosting learn nonlinear relationships and interactions without requiring the analyst to specify every term."),
            ("8. Build the residual hybrid","Use out-of-fold base predictions, calculate residuals, train an AI residual learner, then add the learned correction to the statistical prediction."),
            ("9. Cross-validation","Repeat training/validation inside the training set so model performance is not judged from one fitting sample alone."),
            ("10. Final test","Calculate RMSE, MAE and R² on observations not used for model fitting or model selection."),
            ("11. Statistical inference","Use OLS coefficient estimates, standard errors, confidence intervals, p-values, ANOVA, VIF and heteroscedasticity diagnostics when appropriate."),
            ("12. Interpret and report","Explain what the results support, what they do not support, uncertainty, limitations and practical warnings."),
        ]
        # Horizontal research flowchart: fixed-width cards + arrows inside a scrollable viewport.
        flow_html=['<div class="wf-viewport"><div class="wf-track">']
        for i,(title,desc) in enumerate(workflow_details):
            num,_,short_title=title.partition('. ')
            flow_html.append(
                f'<div class="wf-card"><div class="wf-num">Step {num}</div>'
                f'<div class="wf-title">{short_title}</div><div class="wf-desc">{desc}</div></div>'
            )
            if i < len(workflow_details)-1:
                flow_html.append('<div class="wf-arrow">→</div>')
        flow_html.append('</div></div>')
        st.markdown(''.join(flow_html), unsafe_allow_html=True)
        st.caption("↔ Scroll horizontally to follow the complete research workflow from study definition to the final report.")

        st.markdown("### Detailed explanation of each step")
        for title,desc in workflow_details:
            with st.expander(title,expanded=False):
                st.write(desc)
                if title.startswith("1."):
                    st.markdown("**Research aim:** Decide whether the analysis is primarily inferential, predictive, or both. This determines how the results should be interpreted.")
                elif title.startswith("5."):
                    st.markdown("**Key rule:** The final test data should remain untouched during model fitting and model-selection decisions.")
                elif title.startswith("9."):
                    st.markdown("**Why it matters:** Cross-validation gives a more stable estimate of performance than relying on one training fit alone.")
                elif title.startswith("11."):
                    st.markdown("**Important:** Statistical inference and predictive performance answer different questions and should be reported separately.")
                elif title.startswith("12."):
                    st.markdown("**Research reporting:** State the data, preprocessing, model specification, validation design, results, diagnostics, limitations and practical implications so the analysis can be reproduced.")

        st.markdown("### How one observation moves through the app")
        st.markdown("**Data row → preprocessing → model fitting → prediction → error = Actual − Predicted → validation → interpretation**")
        st.markdown("### What the user should remember")
        st.warning("A high training R² is not enough. For prediction, the most important evidence comes from cross-validation and the untouched test set. For inference, coefficient uncertainty and model assumptions matter separately.")

    else:
        model_details={
        "Linear Regression / OLS": {
            "purpose":"Estimate an interpretable conditional mean relationship between a continuous outcome and one or more predictors.",
            "model":r"Y=X\beta+\varepsilon,\qquad E(\varepsilon\mid X)=0",
            "objective":r"\hat\beta=\arg\min_\beta\sum_{i=1}^{n}(y_i-x_i^T\beta)^2",
            "calculation":[
                "1. Build the design matrix X and include a column of ones for the intercept.",
                r"2. Write the fitted values as \hat Y=X\beta.",
                r"3. Calculate residuals e=Y-\hat Y.",
                r"4. Minimise SSE=e^Te=(Y-X\beta)^T(Y-X\beta).",
                r"5. Set the first derivative to zero, giving the normal equations X^TX\hat\beta=X^TY.",
                r"6. For full column rank, solve \hat\beta=(X^TX)^{-1}X^TY.",
                r"7. Obtain fitted values \hat Y=X\hat\beta and residuals e=Y-\hat Y.",
                r"8. Estimate residual variance and derive standard errors, t-statistics, p-values and confidence intervals under the OLS assumptions.",
            ],
            "interpret":"For a numeric predictor, βj is the estimated change in the conditional mean of Y for a one-unit increase in Xj, holding the other included predictors constant.",
            "app":"The app fits the linear model, evaluates it using train/test and cross-validation, and separately fits a full-data OLS specification for inferential statistics.",
            "limits":["Incorrect functional form can bias interpretation.","Multicollinearity can make coefficients unstable.","Influential observations can strongly affect the fitted equation.","Heteroscedasticity can invalidate conventional standard errors if unaddressed.","Association is not automatically causation.","Extrapolation beyond the observed X range can be unsafe."],
            "source":"Linear Regression / OLS"
        },
        "Polynomial Regression": {
            "purpose":"Represent curvature by transforming predictors into powers and interaction terms while remaining linear in the coefficients.",
            "model":r"Y=\beta_0+\beta_1X+\beta_2X^2+\cdots+\beta_dX^d+\varepsilon",
            "objective":r"\hat\beta=\arg\min_\beta\|Y-X^*\beta\|_2^2",
            "calculation":[
                "1. Start with the original predictor(s).",
                "2. Generate polynomial terms such as X², X³ and interaction terms when selected.",
                "3. Construct the expanded design matrix X*.",
                r"4. Estimate the coefficients by least squares: \hat\beta=(X^{*T}X^*)^{-1}X^{*T}Y when full rank.",
                "5. Calculate fitted values from the polynomial equation.",
                "6. Evaluate the fitted curve on validation/test observations.",
                "7. Compare degree choices using cross-validation rather than training fit alone.",
            ],
            "interpret":"The whole fitted curve is usually more meaningful than interpreting a high-order coefficient in isolation. A positive X² term creates upward curvature; a negative X² term creates downward curvature, but the derivative of the full equation determines the marginal relationship at a particular X.",
            "app":"The app creates degree-2 or degree-3 polynomial features, applies the common preprocessing pipeline, fits the regression and evaluates it with the same validation design as the other models.",
            "limits":["Higher degree can overfit small samples.","Polynomial features may be strongly correlated.","High-order coefficients are difficult to interpret alone.","Extrapolation can become extreme outside the observed range.","Degree should be validated rather than selected solely from training R²."],
            "source":"Polynomial Regression"
        },
        "Ridge Regression": {
            "purpose":"Reduce coefficient variance by penalising large coefficients, especially when predictors contain overlapping information.",
            "model":r"\hat\beta=\arg\min_\beta\left[\sum_i(y_i-x_i^T\beta)^2+\lambda\sum_{j=1}^{p}\beta_j^2\right]",
            "objective":r"(X^TX+\lambda I)\hat\beta=X^TY,\qquad \hat\beta=(X^TX+\lambda I)^{-1}X^TY",
            "calculation":[
                "1. Prepare and normally standardise predictors so the penalty is comparable across variables.",
                "2. Define the OLS squared-error loss.",
                r"3. Add the L2 penalty \lambda\sum_j\beta_j^2.",
                "4. Differentiate the penalised objective and set the gradient to zero.",
                r"5. Solve the penalised normal equations (XᵀX+λI)β=XᵀY.",
                "6. The positive diagonal penalty reduces the magnitude of unstable coefficients.",
                "7. Select the regularisation strength with cross-validation in a research-grade workflow.",
                "8. Evaluate on held-out data and report the chosen penalty value.",
            ],
            "interpret":"Ridge coefficients are shrunk versions of the unregularised coefficients. The main research question is whether the bias introduced by shrinkage produces better generalisation or more stable estimates.",
            "app":"The app fits Ridge inside the preprocessing pipeline and should be configured to tune the regularisation strength by cross-validation rather than relying on an arbitrary fixed value.",
            "limits":["The penalty introduces bias by design.","All predictors generally remain in the model.","Coefficient magnitudes depend on scaling and the selected penalty.","Ordinary OLS p-values should not simply be attached to Ridge coefficients."],
            "source":"Ridge Regression"
        },
        "Lasso Regression": {
            "purpose":"Regularise a linear model and perform sparse variable selection by encouraging some coefficients to become exactly zero.",
            "model":r"\hat\beta=\arg\min_\beta\left[\frac{1}{2n}\|Y-X\beta\|_2^2+\lambda\sum_j|\beta_j|\right]",
            "objective":r"\min_\beta\left[\text{least-squares loss}+\lambda\|\beta\|_1\right]",
            "calculation":[
                "1. Prepare/scale predictors.",
                "2. Define squared prediction loss.",
                "3. Add the L1 penalty λΣ|βj|.",
                "4. Because the absolute-value penalty is not differentiable at zero, a simple closed-form OLS inverse is not available for the whole problem.",
                "5. Optimisation methods such as coordinate descent update coefficients iteratively.",
                "6. The L1 penalty can push some coefficients exactly to zero.",
                "7. Tune λ by cross-validation.",
                "8. Report which coefficients remain non-zero and evaluate test performance.",
            ],
            "interpret":"A zero coefficient means the penalised solution has excluded that predictor from the fitted linear predictor at the selected penalty strength. With correlated predictors, selection can be unstable because several variables may carry similar information.",
            "app":"The app evaluates Lasso alongside OLS and Ridge and should report regularisation strength and the number of non-zero coefficients when tuning is enabled.",
            "limits":["Variable selection can be unstable with correlated predictors.","Coefficients are shrunk toward zero.","The chosen penalty strongly affects which variables remain active.","Standard OLS p-values are not directly transferable to post-selection Lasso coefficients."],
            "source":"Lasso Regression"
        },
        "Elastic Net Regression": {
            "purpose":"Combine L1 sparsity with L2 stabilisation, making it useful when there are many correlated predictors and a sparse but stable solution is desired.",
            "model":r"\min_\beta\left[\frac{1}{2n}\|Y-X\beta\|_2^2+\lambda\left(\rho\|\beta\|_1+\frac{1-\rho}{2}\|\beta\|_2^2\right)\right]",
            "objective":"Two tuning dimensions matter: the total regularisation strength and the L1/L2 mixture.",
            "calculation":[
                "1. Prepare and scale predictors.",
                "2. Define squared-error loss.",
                "3. Add an L1 component for sparsity.",
                "4. Add an L2 component for stabilisation and grouping behaviour.",
                "5. Optimise the combined objective with an iterative algorithm.",
                "6. Tune the penalty strength and L1 ratio using cross-validation.",
                "7. Obtain the final coefficient vector and inspect which coefficients are zero/non-zero.",
                "8. Evaluate generalisation on the untouched test set.",
            ],
            "interpret":"Elastic Net is useful when Lasso may be unstable because predictors are correlated. It can keep groups of related predictors while still encouraging sparsity.",
            "app":"The app compares Elastic Net with Ridge and Lasso so users can see whether the balance between shrinkage and selection is useful for the same dataset.",
            "limits":["Poor tuning can over- or under-regularise.","Coefficients depend on scaling and the penalty mixture.","Selected variables can change with tuning and resampling.","OLS p-values are not automatically valid for the regularised solution."],
            "source":"Elastic Net Regression"
        },
        "Random Forest Regression": {
            "purpose":"Learn nonlinear relationships and interactions by averaging predictions from many randomized regression trees.",
            "model":r"\hat f(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)",
            "objective":"There is no single OLS-style coefficient vector. The model grows many trees and aggregates their predictions.",
            "calculation":[
                "1. Draw a bootstrap sample for a tree.",
                "2. At each node, consider a random subset of predictors.",
                "3. Search candidate split rules that reduce the chosen regression impurity/error.",
                "4. Recursively grow the tree until stopping rules are reached.",
                "5. Record the prediction in each terminal leaf.",
                "6. Repeat for many trees using different resamples and random predictor subsets.",
                r"7. For a new observation, send it through every tree and average the B tree predictions.",
                "8. Use cross-validation and the test set to evaluate generalisation.",
            ],
            "interpret":"The prediction is the average of many tree predictions. Feature importance describes how much fitted prediction performance is associated with using features in the ensemble; it is not a causal coefficient.",
            "app":"The app preprocesses X, trains the requested forest, evaluates test/CV performance and exposes feature importance in Chart Studio.",
            "limits":["Less interpretable than a simple equation.","Poor extrapolation outside the training range.","Importance can be affected by correlated predictors.","Small datasets may produce unstable ensembles.","Hyperparameters such as tree count and depth affect bias/variance."],
            "source":"Random Forest"
        },
        "Gradient Boosting Regression": {
            "purpose":"Build an additive ensemble sequentially, with each new tree correcting the current model according to the loss gradient.",
            "model":r"F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta h_m(x)",
            "objective":"At each stage, fit a new weak learner to the negative gradient of the loss with respect to the current predictions.",
            "calculation":[
                "1. Start with an initial prediction F0(x), commonly related to the mean for squared-error loss.",
                "2. Calculate the current error/negative-gradient signal.",
                "3. Fit a shallow regression tree hm(x) to that signal.",
                r"4. Update the ensemble: Fm(x)=F_{m-1}(x)+η h_m(x).",
                "5. Recalculate the error signal after the update.",
                "6. Repeat for M stages.",
                "7. Evaluate the final additive model with cross-validation and the untouched test set.",
            ],
            "interpret":"The final prediction is the sum of many small tree-based corrections. Tree count, depth and learning rate control how quickly the model learns and how flexible it becomes.",
            "app":"The app fits GradientBoostingRegressor with the configured tree count, learning rate and depth, then reports test/CV metrics and feature importance.",
            "limits":["Can overfit when too flexible.","Sensitive to hyperparameter choices.","Less transparent than coefficient-based regression.","Feature importance is not a causal effect.","Extrapolation beyond the training range is limited."],
            "source":"Gradient Boosting"
        },
        "Statistical–AI Residual Hybrid": {
            "purpose":"Preserve an interpretable statistical base while allowing AI to learn predictable structure remaining in the residuals.",
            "model":r"Y=m(X)+\varepsilon,\qquad \hat m_H(X)=\hat g(X)+\hat h(X)",
            "objective":"The research question is whether the residual component contains systematic, generalisable information beyond the statistical model.",
            "calculation":[
                "1. Fit the statistical base model inside the training folds.",
                "2. Produce out-of-fold predictions so each training observation receives a base prediction from a model that did not train on that observation.",
                r"3. Calculate OOF residuals e_i^{OOF}=Y_i-\hat g^{(-k)}(X_i).",
                "4. Train the AI residual learner h(X) on the OOF residuals.",
                "5. Refit the statistical base model on the complete training data.",
                r"6. For a new observation, calculate \hat Y_H=\hat g(X)+\hat h(X).",
                "7. Evaluate the hybrid on the untouched test set.",
                "8. Compare the hybrid against the statistical base model under the same validation design.",
            ],
            "interpret":"A test-set improvement in RMSE/MAE or R² provides evidence of incremental predictive value under that validation design. It does not prove the hybrid will improve on other populations or datasets.",
            "app":"The app uses out-of-fold residual logic and compares the hybrid with the linear base model and other candidate models on the same test set.",
            "limits":["Residual leakage can make performance look artificially good.","The AI correction can overfit residual noise.","More complexity does not guarantee better prediction.","Uncertainty is harder to quantify than for one OLS model.","Results depend on the base model, residual learner and validation design."],
            "source":"Linear + AI Residual Hybrid"
        }
    }

    if theory_topic=="Prediction metrics, validation and inference":
        theory_header("Prediction metrics, validation and statistical inference", "These quantities answer different questions, so they should not be mixed together.")
        st.markdown("### RMSE")
        st.latex(r"RMSE=\sqrt{\frac{1}{n}\sum_i(Y_i-\hat Y_i)^2}")
        st.write("RMSE is in the units of Y and gives extra influence to large prediction errors. A lower value means smaller squared prediction error on the stated evaluation sample.")
        st.markdown("### MAE")
        st.latex(r"MAE=\frac{1}{n}\sum_i|Y_i-\hat Y_i|")
        st.write("MAE is the average absolute error in the units of Y. It is easier to explain directly and is less sensitive to a few very large errors than RMSE.")
        st.markdown("### R²")
        st.latex(r"R^2=1-\frac{SSE}{SST}")
        st.write("R² compares predictions with a mean-prediction baseline on the same evaluation sample. It is not a percentage accuracy score and it can be negative on test data.")
        st.markdown("### Cross-validation")
        st.write("K-fold cross-validation repeatedly fits on K−1 training folds and evaluates on the remaining fold. The mean and fold-to-fold variation show how stable performance is within the training data.")
        st.markdown("### Final test set")
        st.write("The untouched test set is reserved for the final out-of-sample evaluation after fitting and model-selection decisions. This is the main internal estimate of generalisation in this app.")
        st.markdown("### Inference versus prediction")
        st.write("OLS coefficients, standard errors, confidence intervals and p-values address inferential questions under model assumptions. Test RMSE, MAE and R² address prediction. One model can be useful for inference while another gives lower prediction error.")
    elif theory_topic in model_details:
        info=model_details[theory_topic]
        theory_header(theory_topic,info["purpose"],info["model"])
        st.markdown("### Mathematical estimation / optimisation")
        st.latex(info["objective"])
        st.markdown("### Step-by-step calculation")
        for item in info["calculation"]: st.markdown(f"**{item}**")
        st.markdown("### How to interpret the fitted model")
        st.write(info["interpret"])
        st.markdown("### How StatAI runs this model")
        st.write(info["app"])
        st.markdown("### ⚠️ Main limitations")
        for item in info["limits"]: st.markdown(f"- {item}")
        src=THEORY_SOURCES[info["source"]]
        st.markdown("### 📚 Recommended detailed learning")
        st.markdown(f"**Full theory:** [{src['theory_label']}]({src['theory']})")
        st.markdown(f"**Python example:** [{src['code_label']}]({src['code']})")
        if info["source"] in ["Random Forest","Gradient Boosting"]:
            st.caption("These models do not estimate an OLS-style coefficient vector. Their learned structure is represented by trees, split rules, leaf predictions and the resulting ensemble prediction.")
        elif info["source"] in ["Ridge Regression","Lasso Regression","Elastic Net Regression"]:
            st.caption("Regularisation changes coefficient estimation. Do not interpret these coefficients as ordinary OLS estimates or attach ordinary OLS p-values without an appropriate inference method.")

if not st.session_state.get("analysis_ready",False):
    with tabs[0]:
        st.info("Choose Y and X in **Data & EDA**, then click **Run / Refresh complete analysis**. Results tabs will populate after the analysis is run.")
    with tabs[2]: st.info("Click **🚀 Run / Refresh Complete Analysis** above the tabs.")
    with tabs[4]: st.info("Click **🚀 Run / Refresh Complete Analysis** above the tabs.")
    with tabs[5]: st.info("Click **🚀 Run / Refresh Complete Analysis** above the tabs.")
    with tabs[6]: st.info("Click **🚀 Run / Refresh Complete Analysis** above the tabs.")
    with tabs[7]: st.info("Click **🚀 Run / Refresh Complete Analysis** above the tabs. Dataset charts remain available in the Chart Studio after the analysis is run.")
    with tabs[8]: st.info("Click **🚀 Run / Refresh Complete Analysis** above the tabs.")
    with tabs[9]: st.info("Click **🚀 Run / Refresh Complete Analysis** above the tabs.")
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
    fig=px.bar(full_comparison,x="Model",y="R²",color="Type",title="Full-data R² comparison",text_auto=".3f"); fig.update_layout(template="plotly_white",height=500,xaxis_tickangle=-35); plot_with_guidance(fig)
    if not coef.empty:
        with st.expander("Full-data OLS coefficients",expanded=True): st.dataframe(coef,use_container_width=True,hide_index=True)

# Predictive results
with tabs[4]:
    st.subheader("📈 Predictive Results")
    a,b,c,d=st.columns(4); a.metric("Selected model",selected_model); b.metric("Test RMSE",f"{float(selrow['RMSE']):.4f}"); c.metric("Test MAE",f"{float(selrow['MAE']):.4f}"); d.metric("Test R²",f"{float(selrow['R²']):.4f}")
    st.caption(f"Selection criterion: {criterion}. Training observations: {len(ytr):,}; untouched test observations: {len(yte):,}.")
    st.dataframe(comparison,use_container_width=True,hide_index=True)
    fig=px.bar(comparison.sort_values("RMSE"),x="Model",y="RMSE",color="Type",title="Test RMSE by model",text_auto=".3f"); fig.update_layout(template="plotly_white",height=500,xaxis_tickangle=-35); plot_with_guidance(fig)
    pred_long=pd.DataFrame({"Actual":np.tile(yte.to_numpy(),len(preds)),"Predicted":np.concatenate(list(preds.values())),"Model":np.repeat(list(preds.keys()),len(yte))})
    fig=px.scatter(pred_long,x="Actual",y="Predicted",facet_col="Model",facet_col_wrap=2,title="Hold-out test predictions"); mn=min(pred_long.Actual.min(),pred_long.Predicted.min()); mx=max(pred_long.Actual.max(),pred_long.Predicted.max()); fig.add_shape(type="line",x0=mn,x1=mx,y0=mn,y1=mx,line_dash="dash",row="all",col="all"); fig.update_layout(template="plotly_white",height=850); plot_with_guidance(fig)

    with st.expander("🔬 Hybrid analysis — statistical model + AI residual correction",expanded=False):
        if "Linear Regression" in comparison.Model.values and "Linear + AI Residual Hybrid" in comparison.Model.values:
            lin=comparison[comparison.Model=="Linear Regression"].iloc[0]; hyb=comparison[comparison.Model=="Linear + AI Residual Hybrid"].iloc[0]
            imp=float(lin["RMSE"]-hyb["RMSE"]); pct=100*imp/float(lin["RMSE"])
            a,b,c=st.columns(3); a.metric("Linear test RMSE",f"{float(lin['RMSE']):.4f}"); b.metric("Hybrid test RMSE",f"{float(hyb['RMSE']):.4f}"); c.metric("Hybrid RMSE change",f"{pct:+.2f}%")
            st.write(f"RMSE difference (Linear − Hybrid): **{imp:.6f}**. Positive means lower hybrid error on this test split.")
        if len(oof):
            a,b,c=st.columns(3); a.metric("OOF residual SD",f"{np.std(oof):.4f}"); b.metric("OOF residual mean",f"{np.mean(oof):.4f}"); c.metric("OOF residual MAE",f"{np.mean(np.abs(oof)):.4f}")
            oof_fig=px.histogram(x=oof,nbins=30,title="Out-of-fold statistical residuals")
            plot_with_guidance(oof_fig)
        st.markdown("### What this result means")
        st.write("The hybrid asks whether an AI learner can predict systematic residual structure left after the statistical base model. A lower test error is evidence of incremental predictive value under this validation design, not a universal claim about hybrid models.")
        st.warning("⚠️ The residual learner must use out-of-fold residuals and an untouched test set; otherwise the apparent improvement can be overly optimistic.")

with tabs[5]:
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

with tabs[6]:
    st.subheader("💡 Results Interpretation & Model Limitations")
    st.caption("Everything is explained on one page so the user does not need to move between several interpretation tabs. Use the section headings and expanders to read only what you need.")

    st.markdown("### 1. Overall result")
    best_test=comparison.loc[comparison.RMSE.idxmin()]
    best_cv=comparison.loc[comparison["CV RMSE"].idxmin()]
    st.write(f"Under the current **{criterion}** selection rule, the displayed model is **{selected_model}**. On the current test split, **{best_test.Model}** has RMSE {best_test.RMSE:.4f}, while the lowest mean cross-validation RMSE is reported for **{best_cv.Model}** at {best_cv['CV RMSE']:.4f}.")
    st.warning("⚠️ These are conditional results for this dataset, variable specification, preprocessing and validation design. They are not universal rankings of regression algorithms.")

    st.markdown("### 2. Model-by-model predictive interpretation")
    for _,r in comparison.iterrows():
        gap=float(r.get("Generalization Gap R²",np.nan))
        with st.expander(str(r["Model"]),expanded=False):
            st.write(f"**Test RMSE:** {r['RMSE']:.4f} | **Test MAE:** {r['MAE']:.4f} | **Test R²:** {r['R²']:.4f} | **CV RMSE:** {r.get('CV RMSE',np.nan):.4f}")
            st.write("Lower RMSE/MAE indicate smaller prediction errors. Higher R² indicates better performance relative to the mean baseline on the same evaluation sample.")
            if np.isfinite(gap):
                if gap>.15: st.warning("The training-to-test R² gap is relatively large. Investigate possible overfitting, model complexity and validation stability.")
                elif gap>.05: st.info("There is a noticeable training-to-test decline. Compare this with cross-validation variability and residual diagnostics.")
                else: st.success("The training-to-test R² gap is comparatively small in this split; still confirm stability with cross-validation.")

    st.markdown("### 3. Error metrics in plain language")
    with st.expander("RMSE — what does it tell me?",expanded=False):
        st.latex(r"RMSE=\sqrt{\frac{1}{n}\sum_i(Y_i-\hat Y_i)^2}")
        st.write("RMSE is the typical error scale in the units of Y, with large errors receiving extra weight. It is useful when large prediction mistakes matter more.")
    with st.expander("MAE — what does it tell me?",expanded=False):
        st.latex(r"MAE=\frac{1}{n}\sum_i|Y_i-\hat Y_i|")
        st.write("MAE is the average absolute prediction error in the units of Y. It is often the easiest metric to explain to a non-specialist.")
    with st.expander("R² — what does it tell me?",expanded=False):
        st.latex(r"R^2=1-\frac{SSE}{SST}")
        st.write("R² compares the model with a mean-prediction baseline on the same evaluation sample. It is not a percent accuracy measure. Test R² can be negative.")

    st.markdown("### 4. OLS result interpretation")
    if not coef.empty:
        for _,r in coef.iterrows():
            v=str(r['Variable']); b=float(r['Coefficient']); p=float(r['p-value']); lo=float(r['95% CI Lower']); hi=float(r['95% CI Upper'])
            if v.lower() in {'const','intercept'}:
                continue
            direction="increase" if b>0 else "decrease" if b<0 else "change near zero"
            evidence=("strong statistical evidence" if p<.001 else "statistical evidence" if p<.05 else "limited statistical evidence")
            with st.expander(v,expanded=False):
                st.write(f"A one-unit increase in **{v}** is associated with an estimated **{abs(b):.4f}-unit {direction}** in the conditional mean of Y, holding the other included predictors constant.")
                st.write(f"Coefficient = {b:.4f}; 95% CI = [{lo:.4f}, {hi:.4f}]; p = {p:.4g}. Under the fitted-model assumptions, this provides {evidence} against a zero coefficient.")
                st.warning("This is a conditional association, not proof of causality. Consider study design, omitted variables, measurement quality and the confidence interval.")
    else:
        st.info("OLS coefficient interpretation is unavailable for this specification.")

    st.markdown("### 5. Hybrid result interpretation")
    if "Linear Regression" in comparison.Model.values and "Linear + AI Residual Hybrid" in comparison.Model.values:
        lin=comparison[comparison.Model=="Linear Regression"].iloc[0]; hyb=comparison[comparison.Model=="Linear + AI Residual Hybrid"].iloc[0]
        rmse_change=float(lin['RMSE']-hyb['RMSE']); r2_change=float(hyb['R²']-lin['R²'])
        st.write(f"The hybrid changes test RMSE by **{rmse_change:+.4f}** and test R² by **{r2_change:+.4f}** relative to Linear Regression.")
        if rmse_change>0: st.success("The hybrid has lower test RMSE than the linear base model on this test split, which is evidence of incremental predictive value under the present validation design.")
        elif rmse_change<0: st.warning("The hybrid has higher test RMSE than the linear base model on this test split; the AI correction did not improve that metric here.")
        else: st.info("The displayed test RMSE is the same to the shown precision.")
    else:
        st.info("Run both Linear Regression and the residual hybrid to obtain the incremental comparison.")

    st.markdown("### 6. Graph interpretation guide")
    graph_guide={
        "Actual vs Predicted":"Observed values are the actual outcomes; the predicted series is the model estimate. Small differences suggest smaller prediction errors.",
        "Identity plot":"The diagonal 1:1 line represents perfect prediction. The closer the test points are to that line, the smaller the prediction errors.",
        "Residuals vs Predicted":"Residual = Actual − Predicted. A roughly patternless cloud around zero is generally reassuring. Curvature, funnels or clusters deserve investigation.",
        "Residual distribution":"The centre shows average signed error; the spread and tails show how variable the errors are. Strong skewness or extreme tails may indicate unusual observations or asymmetric errors.",
        "Correlation heatmap":"The cells show pairwise linear association. Strong predictor-predictor correlations can indicate multicollinearity, but correlation is not causation and may miss nonlinear relationships.",
        "Feature importance":"Importance indicates how the fitted tree model used features for prediction. It is not a regression coefficient and does not prove that a feature causes Y.",
    }
    for title,desc in graph_guide.items():
        with st.expander(title,expanded=False):
            st.write(desc)
            st.warning("Interpret the graph together with the numerical metric table and the research context. A visually attractive pattern is not by itself evidence of causation or generalisation.")
            st.write("**Suggested next check:** compare the graph across training/test data, inspect residuals, and verify whether the observed pattern is consistent with the domain knowledge.")

    st.markdown("### 7. Model-specific limitations")
    selected_limit_model=st.selectbox("Choose a model to read its limitations",MODEL_NAMES,key="limit_model_v17")
    limits={
        "Linear Regression":["Functional-form misspecification can distort inference and prediction.","Multicollinearity can make coefficients unstable.","Influential observations can materially change the fit.","Heteroscedasticity affects conventional standard errors.","Extrapolation outside the observed predictor range can be unreliable."],
        "Polynomial Regression":["Higher degree can overfit.","Polynomial terms can be highly correlated.","Higher-order coefficients are difficult to interpret individually.","Extrapolation can become extreme and unrealistic."],
        "Ridge Regression":["The penalty introduces shrinkage bias by design.","Penalty strength should be tuned rather than fixed arbitrarily.","All predictors generally remain in the model.","Ordinary OLS p-values should not be attached to Ridge coefficients without suitable inference."],
        "Lasso Regression":["Selection can be unstable with correlated predictors.","Coefficients are shrunk toward zero.","Penalty strength strongly affects selected variables.","Post-selection OLS p-values are not automatically valid."],
        "Elastic Net":["Two tuning dimensions must be chosen.","Coefficient interpretation depends on correlated predictors and the selected penalty mixture.","Scaling matters.","Selected variables can change with tuning and resampling."],
        "Random Forest":["The model is less transparent than a coefficient equation.","Feature importance is not a causal effect.","Extrapolation outside the training range is weak.","Small datasets can produce unstable tree ensembles."],
        "Gradient Boosting":["Excessive depth, tree count or learning rate can overfit.","Performance is sensitive to hyperparameters.","Feature importance is not a causal effect.","Extrapolation is limited."],
        "Linear + AI Residual Hybrid":["Residual leakage can create overly optimistic performance.","The residual learner can overfit noise.","Additional complexity may not improve generalisation.","Uncertainty estimation is more complicated than for OLS.","Results depend strongly on the base model, residual learner and validation design."],
    }
    for item in limits[selected_limit_model]: st.markdown(f"- {item}")

    st.markdown("### 8. Common warnings for every regression")
    warnings=[
        ("Data leakage","Do not use information that would not be available at prediction time."),
        ("Small sample","Small samples can make coefficients, p-values, cross-validation and AI predictions unstable."),
        ("Missing data","Understand why data are missing before deleting observations."),
        ("Outliers / influence","A few unusual observations can strongly affect coefficients and error metrics."),
        ("Multicollinearity","High predictor correlation can destabilise coefficient interpretation."),
        ("Causality","Regression association is not automatically a causal effect."),
        ("Extrapolation","A model validated within the observed range may fail outside it."),
        ("Distribution shift","Performance can change when the future population differs from the training population."),
    ]
    for t,d in warnings: st.markdown(f"**{t}:** {d}")

    st.markdown("### 9. Research reporting suggestion")
    st.info("A PhD-level report should state separately: (1) data and preprocessing, (2) statistical inference, (3) cross-validation, (4) untouched-test prediction, (5) hybrid incremental value, (6) diagnostics, and (7) limitations. Avoid reducing the entire analysis to a single 'best model' number.")

with tabs[7]:
    st.subheader("🎨 Chart Studio — Simple, Clear & Interpreted")
    st.caption("Start with a simple chart. The app explains what the graph describes, what to look for, what can be misleading, and what to check next.")

    nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cats=[c for c in df.columns if c not in nums]
    chart_type=st.selectbox(
        "Choose a chart",
        [
            "Scatter plot",
            "Histogram",
            "Box plot",
            "Bar chart",
            "Line / trend",
            "Correlation heatmap",
            "Actual vs Predicted",
            "Residuals vs Predicted",
            "Residual distribution",
            "Feature importance",
        ],
        key="simple_chart_type_v17"
    )
    fig=None
    chart_note=""

    if chart_type=="Scatter plot":
        if len(nums)<2: st.warning("At least two numeric variables are needed.")
        else:
            c1,c2,c3=st.columns(3)
            xx=c1.selectbox("X variable",nums,key="simple_sc_x")
            yy=c2.selectbox("Y variable",nums,index=min(1,len(nums)-1),key="simple_sc_y")
            use_group=c3.selectbox("Colour by",["None"]+cats,key="simple_sc_group")
            point_colour=st.color_picker("Point colour", "#2E86DE", key="simple_sc_colour") if use_group=="None" else None
            temp=df[[xx,yy]+([] if use_group=="None" else [use_group])].dropna()
            fig=px.scatter(temp,x=xx,y=yy,color=None if use_group=="None" else use_group,trendline="ols" if len(temp)>=10 else None,title=f"Scatter plot: {yy} vs {xx}")
            if use_group=="None": fig.update_traces(marker=dict(color=point_colour))
            chart_note="Use this to see whether two numeric variables move together, whether the relationship is roughly linear, and whether unusual observations stand out. A trendline is descriptive, not proof of causation."

    elif chart_type=="Histogram":
        v=st.selectbox("Numeric variable",nums,key="simple_hist_var")
        c1,c2=st.columns(2); colour=c1.color_picker("Histogram colour","#4C78A8",key="simple_hist_colour"); bins=c2.slider("Bins",10,60,25,key="simple_hist_bins")
        fig=go.Figure(go.Histogram(x=df[v].dropna(),nbinsx=bins,marker_color=colour,name=v))
        fig.update_layout(title=f"Distribution of {v}",xaxis_title=v,yaxis_title="Frequency")
        chart_note="This shows how often values occur. Look for centre, spread, skewness, multiple peaks and extreme values. A skewed distribution is not automatically a problem; its importance depends on the model and research question."

    elif chart_type=="Box plot":
        c1,c2,c3=st.columns(3)
        v=c1.selectbox("Numeric variable",nums,key="simple_box_var")
        g=c2.selectbox("Group",["None"]+cats,key="simple_box_group")
        colour=c3.color_picker("Box colour","#F58518",key="simple_box_colour")
        fig=px.box(df,y=v,x=None if g=="None" else g,points="outliers",title=f"Box plot of {v}" if g=="None" else f"{v} by {g}")
        if g=="None": fig.update_traces(marker_color=colour,line_color=colour)
        chart_note="The box shows the middle 50% of values, the centre line marks the median, and plotted outliers are observations far from the central range. Outliers should be investigated, not automatically deleted."

    elif chart_type=="Bar chart":
        if not cats: st.warning("A categorical/grouping variable is needed for a simple bar chart.")
        else:
            c1,c2,c3=st.columns(3)
            g=c1.selectbox("Group",cats,key="simple_bar_group")
            v=c2.selectbox("Numeric measure",nums,key="simple_bar_value")
            agg=c3.selectbox("Summary",["Mean","Median","Count"],key="simple_bar_agg")
            if agg=="Mean": p=df.groupby(g,dropna=False)[v].mean().reset_index(name="Value")
            elif agg=="Median": p=df.groupby(g,dropna=False)[v].median().reset_index(name="Value")
            else: p=df.groupby(g,dropna=False)[v].count().reset_index(name="Value")
            bar_colour=st.color_picker("Bar colour","#59A14F",key="simple_bar_colour")
            fig=px.bar(p,x=g,y="Value",text_auto=".2f",title=f"{agg} of {v} by {g}",color_discrete_sequence=[bar_colour])
            chart_note="Bars compare group summaries. Before interpreting differences, check group sizes and variability. A difference in means is descriptive and does not by itself show a causal group effect."

    elif chart_type=="Line / trend":
        c1,c2,c3=st.columns(3)
        xx=c1.selectbox("Order / time variable",df.columns.tolist(),key="simple_line_x")
        v=c2.selectbox("Numeric measure",nums,key="simple_line_v")
        marker=c3.checkbox("Show markers",value=True,key="simple_line_marker")
        line_colour=st.color_picker("Line colour","#E45756",key="simple_line_colour")
        temp=df[[xx,v]].dropna().copy()
        parsed=pd.to_datetime(temp[xx],errors="coerce")
        if parsed.notna().mean()>=.8:
            temp["__date"]=parsed; temp=temp.sort_values("__date"); temp[xx]=temp["__date"].dt.strftime("%Y-%m-%d")
        fig=px.line(temp,x=xx,y=v,markers=marker,title=f"{v} across {xx}")
        fig.update_traces(line_color=line_colour)
        chart_note="A line chart shows how a measure changes across an ordered x-axis. A visible trend can indicate temporal or ordered association, but seasonality, grouping and external factors should be considered before causal interpretation."

    elif chart_type=="Correlation heatmap":
        if len(nums)<2: st.warning("At least two numeric variables are required.")
        else:
            corr=df[nums].corr(numeric_only=True)
            colors=st.selectbox("Colour scale",["RdBu_r","Viridis","Cividis"],key="simple_corr_scale")
            fig=px.imshow(corr,text_auto=".2f",aspect="auto",title="Pearson correlation heatmap",color_continuous_scale=colors,zmin=-1,zmax=1)
            chart_note="Values near +1 indicate strong positive linear association; values near −1 indicate strong negative linear association; values near 0 indicate weak linear association. Correlation does not establish causation and may miss nonlinear patterns."

    elif chart_type=="Actual vs Predicted":
        model_for_chart=st.selectbox("Model",list(preds.keys()),key="simple_actual_pred_model")
        c1,c2=st.columns(2)
        actual_colour=c1.color_picker("Actual point colour","#2CA02C",key="simple_actual_colour")
        predicted_colour=c2.color_picker("Predicted line colour","#1F77B4",key="simple_pred_colour")
        actual=yte.to_numpy(dtype=float); pred=np.asarray(preds[model_for_chart],dtype=float); obs=np.arange(1,len(actual)+1)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=obs,y=actual,mode="markers",name="Actual — observed points",marker=dict(color=actual_colour,size=8)))
        fig.add_trace(go.Scatter(x=obs,y=pred,mode="lines",name="Predicted — model line",line=dict(color=predicted_colour,width=3)))
        fig.update_layout(title=f"Actual vs Predicted — {model_for_chart}",xaxis_title="Test observation order",yaxis_title=y_col,legend_title="What is shown")
        chart_note="The points are the actual observed Y values. The line is the model's predicted Y values for those same test observations in their displayed order. The line is not a regression trendline. Close point-to-line agreement indicates smaller individual errors."

    elif chart_type=="Residuals vs Predicted":
        model_for_chart=st.selectbox("Model",list(preds.keys()),key="simple_resid_model")
        point_colour=st.color_picker("Residual point colour","#9467BD",key="simple_resid_colour")
        pred=np.asarray(preds[model_for_chart],dtype=float); residual=yte.to_numpy(dtype=float)-pred
        temp=pd.DataFrame({"Predicted":pred,"Residual":residual})
        fig=px.scatter(temp,x="Predicted",y="Residual",title=f"Residuals vs Predicted — {model_for_chart}",color_discrete_sequence=[point_colour])
        fig.add_hline(y=0,line_dash="dash",line_width=2)
        chart_note="Each point shows prediction versus error, where residual = Actual − Predicted. A roughly random cloud around zero is desirable; curvature, a funnel shape or clusters suggest that the model may be missing structure or has non-constant error variance."

    elif chart_type=="Residual distribution":
        model_for_chart=st.selectbox("Model",list(preds.keys()),key="simple_resid_dist_model")
        hist_colour=st.color_picker("Residual colour","#4E79A7",key="simple_resid_dist_colour")
        residual=yte.to_numpy(dtype=float)-np.asarray(preds[model_for_chart],dtype=float)
        fig=go.Figure(go.Histogram(x=residual,nbinsx=30,marker_color=hist_colour,name="Residuals"))
        fig.update_layout(title=f"Residual distribution — {model_for_chart}",xaxis_title="Residual = Actual − Predicted",yaxis_title="Frequency")
        chart_note="The centre of the distribution indicates systematic over- or under-prediction on average; the spread shows how variable the errors are; the tails show unusually large errors. Normal-looking errors are not automatically required for good prediction, but are relevant to some inferential procedures."

    else:  # Feature importance
        choices=[m for m in ["Random Forest","Gradient Boosting"] if m in specs]
        if not choices: st.info("Feature importance is available when a tree-based model has been run.")
        else:
            model_for_chart=st.selectbox("Tree model",choices,key="simple_importance_model")
            ai_fit=make_clone(specs[model_for_chart]); ai_fit.fit(Xtr,ytr)
            prep=ai_fit.named_steps["prep"]; mdl=ai_fit.named_steps["model"]
            names=list(prep.get_feature_names_out()); vals=getattr(mdl,"feature_importances_",None)
            if vals is not None:
                fi=pd.DataFrame({"Feature":names,"Importance":vals}).sort_values("Importance",ascending=False).head(20)
                imp_colour=st.color_picker("Bar colour","#59A14F",key="simple_importance_colour")
                fig=px.bar(fi.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title=f"Feature importance — {model_for_chart}",color_discrete_sequence=[imp_colour])
                chart_note="This ranks features by their contribution to the fitted tree ensemble's predictive splitting structure. It is not a regression coefficient, not a p-value, and not proof that the feature causes the outcome."

    if fig is not None:
        fig.update_layout(template="plotly_white",height=560,margin=dict(l=30,r=30,t=80,b=50),hovermode="closest")
        st.plotly_chart(fig,use_container_width=True)
        with st.expander("🧭 What does this graph describe?",expanded=True):
            st.markdown(f"**What you are seeing:** {chart_note}")
            gtitle=str(fig.layout.title.text or "")
            interpretation,warning,suggestion=chart_guidance(gtitle)
            if chart_note:
                st.markdown(f"**Plain-language meaning:** {chart_note}")
            st.markdown(f"**⚠️ Interpretation warning:** {warning}")
            st.markdown(f"**✅ What to do next:** {suggestion}")

with tabs[8]:
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

st.divider(); st.caption("StatAI v17 • Statistical–AI Hybrid Modelling Platform • Research and educational use")
