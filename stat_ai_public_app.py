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
    st.divider(); st.caption("Research note: full-data fit, statistical inference and out-of-sample prediction are reported separately.")

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

with st.sidebar:
    y_col=st.selectbox("Outcome / dependent variable (Y)",numeric)

available=[c for c in df.columns if c!=y_col]
non_id=[c for c in available if str(c).strip().lower().replace(" ","_") not in ID_NAMES]
default=[c for c in non_id if c in numeric][:5]
if not default:
    default=[c for c in available if c in numeric][:5]

x_cols=st.multiselect("Predictors / independent variables (X)",available,default=default)
if not x_cols:
    st.warning("Select at least one predictor."); st.stop()

excluded_ids=[c for c in available if c not in x_cols and str(c).strip().lower().replace(" ","_") in ID_NAMES]
if excluded_ids:
    st.info("Identifier columns excluded from the default predictors: " + ", ".join(excluded_ids))

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
    st.subheader("📚 Theory, workflow & model flowcharts")
    theory_tabs=st.tabs(["🧭 Overall workflow","📐 Statistical models","🤖 AI models","🔗 Hybrid model","📊 Metrics & validation"])
    with theory_tabs[0]:
        workflow=[
            ("1","Data","Upload data and define the research outcome and predictors."),("2","Quality Check","Inspect missing values, types, identifiers and usable observations."),("3","EDA","Study distributions, correlations, outliers and possible nonlinear patterns."),("4","Train / Test","Reserve a final test set before model comparison."),("5","Statistical Models","Estimate interpretable linear and regularised relationships."),("6","AI Models","Learn nonlinearities and interactions with tree ensembles."),("7","Hybrid","Learn predictable residual structure and add it to the statistical base prediction."),("8","Cross-Validation","Repeat training/validation folds within the training data to estimate generalisation."),("9","Final Test","Evaluate on observations not used for fitting or model selection."),("10","Diagnostics","Check multicollinearity, residual behaviour and train-test gaps."),("11","Interpretation","Separate predictive evidence, statistical inference and hybrid evidence."),("12","Research Report","Record methods, results, limitations and reproducibility information.")]
        cards=[]
        for i,(n,t,d) in enumerate(workflow):
            cards.append(f'<div class="wf-card"><div class="wf-num">Step {n}</div><div class="wf-title">{t}</div><div class="wf-desc">{d}</div></div>')
            if i<len(workflow)-1: cards.append('<div class="wf-arrow">→</div>')
        st.markdown('<div class="wf-viewport"><div class="wf-track">'+''.join(cards)+'</div></div><div class="note"><b>Read left → right.</b> The workflow scrolls horizontally instead of shrinking the cards, so the labels do not overlap.</div>',unsafe_allow_html=True)
        st.markdown("### How information moves")
        for n,title,desc in workflow:
            st.markdown(f"**{n}. {title}** — {desc}")
    with theory_tabs[1]:
        st.markdown("### Linear Regression"); st.latex(r"Y=\beta_0+\beta_1X_1+\cdots+\beta_pX_p+\varepsilon"); st.write("Least squares chooses coefficients that minimise the sum of squared residuals. The coefficients provide an interpretable conditional relationship under the specified model and assumptions.")
        st.markdown("### Polynomial Regression"); st.write("Polynomial features add powers such as X² and X³ so a linear coefficient model can represent curvature. Higher degree increases flexibility and can increase overfitting risk.")
        st.markdown("### Ridge, Lasso and Elastic Net"); st.write("These regularised models add penalties to coefficient size. Ridge shrinks coefficients, Lasso can set some coefficients to zero, and Elastic Net combines L1 and L2 penalties.")
    with theory_tabs[2]:
        st.markdown("### Random Forest"); st.write("Random Forest builds many decision trees using resampled observations and random subsets of predictors. Each tree captures nonlinear splits and interactions; predictions are aggregated across trees.")
        st.markdown("### Gradient Boosting"); st.write("Gradient Boosting builds trees sequentially. Each new tree focuses on remaining prediction error, gradually improving the ensemble. This flexibility can capture nonlinear patterns but requires validation to control overfitting.")
    with theory_tabs[3]:
        st.markdown("### Residual hybrid modelling"); st.latex(r"\hat m_H(X)=\hat g(X)+\hat h(X)"); st.write("The statistical model provides an interpretable base structure. Out-of-fold residuals are then calculated so the AI learner does not simply memorise residuals from predictions generated on the same observations used to fit the base model. The AI component learns predictable residual structure. Final prediction is the base prediction plus the learned correction.")
        st.markdown("### Hybrid workflow"); st.markdown("**Base model → Out-of-fold predictions → Residuals → AI residual learner → Refit base model → Base prediction + AI correction → Validation**")
    with theory_tabs[4]:
        st.markdown("### Metrics"); st.latex(r"RMSE=\sqrt{\frac{1}{n}\sum_i(Y_i-\hat Y_i)^2}"); st.latex(r"MAE=\frac{1}{n}\sum_i|Y_i-\hat Y_i|"); st.latex(r"R^2=1-\frac{SSE}{SST}")
        st.write("Lower RMSE/MAE indicate smaller prediction errors. R² measures improvement relative to a mean-baseline reference on the same evaluation sample.")
        st.markdown("### Validation logic"); st.write("Training data are used for fitting. Cross-validation estimates performance within the training data. The untouched test set is reserved for final out-of-sample evaluation. Full-data OLS inference is a separate analysis and should not be confused with test-set prediction.")

# Analysis button
run=st.button("🚀 Run / Refresh complete analysis",type="primary",use_container_width=True)
if not run:
    st.info("Select your variables and click **Run / Refresh complete analysis** to calculate the regression, AI, hybrid, diagnostics and reports.")
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
    st.subheader("🎨 Chart Studio")
    nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]; cats=[c for c in df.columns if c not in nums]
    chart=st.selectbox("Chart type",["Scatter plot","Histogram","Box plot","Bar chart","Line / trend","Correlation heatmap","Prediction error distribution","CV RMSE with uncertainty"])
    if chart=="Scatter plot":
        xx=st.selectbox("X",nums); yy=st.selectbox("Y",nums,index=min(1,len(nums)-1)); fig=px.scatter(df,x=xx,y=yy,title=f"{yy} vs {xx}")
    elif chart=="Histogram":
        v=st.selectbox("Variable",nums); fig=px.histogram(df,x=v,nbins=30,title=f"Distribution of {v}")
    elif chart=="Box plot":
        v=st.selectbox("Numeric variable",nums); g=st.selectbox("Group",["None"]+cats); fig=px.box(df,y=v,x=None if g=="None" else g,points="outliers",title=f"Box plot of {v}")
    elif chart=="Bar chart":
        g=st.selectbox("Category",cats or df.columns.tolist()); v=st.selectbox("Numeric measure",nums); p=df.groupby(g,dropna=False)[v].mean().reset_index(); fig=px.bar(p,x=g,y=v,title=f"Mean {v} by {g}")
    elif chart=="Line / trend":
        v=st.selectbox("Numeric variable",nums); fig=px.line(df.reset_index(),x="index",y=v,title=f"{v} across row order")
    elif chart=="Correlation heatmap":
        fig=px.imshow(df[nums].corr(),text_auto=".2f",aspect="auto",title="Pearson correlation matrix",color_continuous_scale="RdBu_r")
    elif chart=="Prediction error distribution":
        err=pd.DataFrame({m:yte.to_numpy()-p for m,p in preds.items()}); lm=err.melt(var_name="Model",value_name="Error"); fig=px.histogram(lm,x="Error",color="Model",barmode="overlay",nbins=35,title="Test prediction error distributions")
    else:
        p=comparison.sort_values("CV RMSE"); fig=go.Figure(go.Bar(x=p.Model,y=p["CV RMSE"],error_y=dict(type="data",array=p["CV RMSE SD"].fillna(0)))); fig.update_layout(title="Mean cross-validation RMSE ± SD")
    fig.update_layout(template="plotly_white",height=600,margin=dict(l=20,r=20,t=70,b=30)); st.plotly_chart(fig,use_container_width=True)

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

st.divider(); st.caption("StatAI v8 • Statistical–AI Hybrid Modelling Platform • Research and educational use")
