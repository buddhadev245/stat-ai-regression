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
st.set_page_config(page_title="StatAI | Statistical–AI Hybrid Modelling", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.6rem;padding-bottom:3rem}
.hero{padding:1.7rem 1.8rem;border:1px solid #cfe2e8;border-radius:22px;background:linear-gradient(135deg,#edf8fb,#ffffff);margin-bottom:1rem}
.kicker{font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:#176b87}.title{font-size:clamp(2rem,4vw,3.2rem);font-weight:850;letter-spacing:-.045em;color:#123746;line-height:1.05}.subtitle{font-size:1.05rem;color:#5b6d74;max-width:1000px;margin-top:.5rem}
.card{padding:1.15rem 1.3rem;border:1px solid #dce8ec;border-radius:16px;background:#fff;box-shadow:0 4px 18px rgba(18,55,70,.05);margin:.6rem 0 1rem}
.muted{color:#64757c}.pill{display:inline-block;padding:.35rem .7rem;border-radius:999px;background:#f1f8fa;border:1px solid #d7e8ed;color:#28576a;font-weight:700;font-size:.8rem;margin:.15rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#f4f8fa,#edf5f7)}
div[data-testid="stMetric"]{background:#f6fafb;border:1px solid #dcecef;padding:.8rem 1rem;border-radius:13px}
[data-testid="stTabs"] button{font-weight:750}
.small{font-size:.88rem;color:#6a7a81}.formula{padding:1rem 1.1rem;background:#f7fbfc;border-left:4px solid #176b87;border-radius:8px;font-family:serif;font-size:1.05rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="kicker">Research • prediction • inference • reproducibility</div><div class="title">📊 Statistical–AI Hybrid Modelling Platform</div><div class="subtitle">A professional browser-based workspace for exploratory analysis, statistical regression, machine-learning comparison, residual hybrid modelling, diagnostics, interpretation and reproducible reporting.</div></div>', unsafe_allow_html=True)

# ----------------------------- helpers -----------------------------
def read_uploaded_file(uploaded):
    n=uploaded.name.lower()
    if n.endswith('.csv'): return pd.read_csv(uploaded)
    if n.endswith(('.xlsx','.xls')): return pd.read_excel(uploaded)
    raise ValueError('Please upload CSV or Excel.')

def coerce_numeric_like(df):
    out=df.copy()
    for c in out.columns:
        if out[c].dtype=='object':
            conv=pd.to_numeric(out[c].astype(str).str.replace(',','',regex=False).str.strip(),errors='coerce')
            nm=out[c].notna().sum()
            if nm and conv.notna().sum()/nm>=.85: out[c]=conv
    return out

def infer_columns(df):
    nums=[]; cats=[]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): nums.append(c)
        else:
            conv=pd.to_numeric(df[c].astype(str).str.replace(',','',regex=False).str.strip(),errors='coerce')
            nm=df[c].notna().sum()
            (nums if nm and conv.notna().sum()/nm>=.85 else cats).append(c)
    return nums,cats

def clean_target(df,c): return pd.to_numeric(df[c].astype(str).str.replace(',','',regex=False).str.strip(),errors='coerce')

def make_preprocessor(X):
    num=list(X.select_dtypes(include=[np.number]).columns); cat=[c for c in X.columns if c not in num]; tr=[]
    if num: tr.append(('num',Pipeline([('imputer',SimpleImputer(strategy='median'))]),num))
    if cat: tr.append(('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
    return ColumnTransformer(tr,remainder='drop')

def pipe(model,X): return Pipeline([('preprocess',make_preprocessor(X)),('model',model)])

def metrics(y,p): return {'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MAE':float(mean_absolute_error(y,p)),'R²':float(r2_score(y,p))}

def hybrid_fit(base,residual,Xtr,ytr,Xte,k,seed):
    Xtr=Xtr.reset_index(drop=True); ytr=ytr.reset_index(drop=True); oof=np.zeros(len(ytr)); kf=KFold(k,shuffle=True,random_state=seed)
    for ti,vi in kf.split(Xtr):
        b=clone(base); b.fit(Xtr.iloc[ti],ytr.iloc[ti]); oof[vi]=ytr.iloc[vi].to_numpy()-b.predict(Xtr.iloc[vi])
    rm=clone(residual); rm.fit(Xtr,oof); bf=clone(base); bf.fit(Xtr,ytr)
    return bf,rm,bf.predict(Xte)+rm.predict(Xte),oof

def safe_rank(series,ascending): return series.rank(method='min',ascending=ascending).astype(int)

# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.header('⚙️ Analysis controls')
    test_size=st.slider('Test-set proportion',.15,.40,.25,.05)
    seed=st.number_input('Random seed',1,9999,42,1)
    folds=st.slider('Cross-validation folds',3,10,5,1)
    st.divider(); st.subheader('Model settings')
    rf_trees=st.slider('Random Forest trees',100,800,300,50)
    gb_trees=st.slider('Gradient Boosting trees',50,500,200,25)
    poly_degree=st.selectbox('Polynomial degree',[2,3],index=0)
    st.divider(); st.subheader('Model selection')
    criterion=st.selectbox('Selection criterion',['Cross-validation RMSE','Cross-validation MAE','Cross-validation R²','Test RMSE','Test MAE','Test R²'])
    selection_mode=st.radio('Final model display',['Automatically select','Choose manually'])
    st.divider(); st.caption('Privacy: uploaded data are processed for the current analysis session. Avoid confidential or personally identifiable data unless appropriate for your hosting environment.')

# ----------------------------- upload -----------------------------
up=st.file_uploader('📥 Upload CSV or Excel dataset',type=['csv','xlsx','xls'],help='Use one numeric outcome and one or more predictors.')
if up is None:
    st.info('Upload a dataset to activate the analysis workspace.')
    st.markdown('<div class="card"><h3>Research workflow</h3><span class="pill">01 Data</span><span class="pill">02 EDA</span><span class="pill">03 Models</span><span class="pill">04 Hybrid</span><span class="pill">05 Diagnostics</span><span class="pill">06 Interpretation</span><span class="pill">07 Report</span><p class="muted">The platform separates predictive evaluation from statistical inference. It does not assume that AI or hybridisation is always superior.</p></div>',unsafe_allow_html=True)
    st.stop()
try: df=coerce_numeric_like(read_uploaded_file(up)).replace([np.inf,-np.inf],np.nan)
except Exception as e: st.error(f'Could not read the dataset: {e}'); st.stop()
if df.empty: st.error('The uploaded dataset is empty.'); st.stop()
nums,cats=infer_columns(df)
if not nums: st.error('No numeric outcome was detected. Regression requires a numeric outcome.'); st.stop()

with st.sidebar:
    st.subheader('Variables')
y_col=st.selectbox('Outcome / dependent variable (Y)',nums)
avail=[c for c in df.columns if c!=y_col]
default=[c for c in avail if c in nums][:5] or avail[:min(5,len(avail))]
x_cols=st.multiselect('Predictors / independent variables (X)',avail,default=default)
if not x_cols: st.warning('Select at least one predictor.'); st.stop()
y=clean_target(df,y_col); valid=y.notna(); X=df.loc[valid,x_cols].reset_index(drop=True); y=y.loc[valid].reset_index(drop=True)
if len(y)<max(30,len(x_cols)+10): st.error('Too few usable observations for a stable regression workflow.'); st.stop()
if any(X[c].nunique(dropna=True)<2 for c in x_cols): st.error('At least one predictor has fewer than two distinct values.'); st.stop()

st.success(f'Loaded **{len(df):,} rows × {len(df.columns):,} columns** • **{len(y):,} usable outcomes**')

# ----------------------------- tabs -----------------------------
tabs=st.tabs(['🏠 Overview','📁 Data & EDA','📚 Theory & Model Flowcharts','📈 Predictive Results','🔬 Hybrid Analysis','🩺 Diagnostics & Inference','💡 Results Interpretation','🎨 Chart Studio','📄 Report & Export'])

with tabs[0]:
    st.markdown('<div class="card"><h2>Analysis overview</h2><p>This dashboard follows a transparent workflow: inspect the data, understand relationships, fit statistical and AI models, evaluate out-of-sample performance, examine the hybrid residual component, check statistical diagnostics, and interpret the complete evidence.</p></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4); a.metric('Observations',f'{len(y):,}'); b.metric('Predictors',f'{len(x_cols):,}'); c.metric('Numeric predictors',f'{sum(pd.api.types.is_numeric_dtype(X[c]) for c in x_cols):,}'); d.metric('Categorical predictors',f'{sum(not pd.api.types.is_numeric_dtype(X[c]) for c in x_cols):,}')
    st.markdown('### Current specification')
    st.write(f'**Outcome (Y):** `{y_col}`')
    st.write('**Predictors (X):** '+', '.join(f'`{c}`' for c in x_cols))
    st.markdown('### What the platform does not assume')
    st.info('A flexible AI model is not automatically better. A hybrid model is not automatically better. Statistical significance is not the same as predictive accuracy. The final interpretation should consider validation, diagnostics, uncertainty and the scientific context.')

with tabs[1]:
    st.subheader('📁 Data preview and quality')
    t1,t2,t3=st.tabs(['Preview','Quality','Descriptive statistics'])
    with t1: st.dataframe(df.head(25),use_container_width=True,hide_index=True)
    with t2:
        q=pd.DataFrame({'Variable':df.columns,'Type':[str(df[c].dtype) for c in df.columns],'Missing':[int(df[c].isna().sum()) for c in df.columns],'Unique':[int(df[c].nunique(dropna=True)) for c in df.columns]})
        st.dataframe(q,use_container_width=True,hide_index=True)
    with t3: st.dataframe(df[x_cols+[y_col]].describe(include='all').T,use_container_width=True)

with tabs[2]:
    st.subheader('📚 Theory, mathematical foundation & model flowcharts')
    st.caption('This tab explains what each model is doing, why it is used, and how information moves from data to prediction.')

    theory_tabs = st.tabs(['🧭 Overall workflow','📐 Statistical models','🤖 AI models','🔗 Hybrid model','📊 Metrics & validation'])

    with theory_tabs[0]:
        st.markdown("""
        <div class="card">
        <h3>Research modelling workflow</h3>
        <p><b>Data → Quality Check → EDA → Train/Test Split → Statistical Models → AI Models → Hybrid Model → Cross-Validation → Final Test → Diagnostics → Interpretation → Report</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Visual flowchart
        # Horizontal workflow: every stage is arranged left-to-right to reduce visual complexity.
        nodes = [
            ("1\nData",0,1),
            ("2\nQuality\ncheck",1,1),
            ("3\nEDA",2,1),
            ("4\nTrain / Test\nsplit",3,1),
            ("5\nStatistical\nmodels",4,1),
            ("6\nAI\nmodels",5,1),
            ("7\nHybrid\nmodel",6,1),
            ("8\nCross-\nvalidation",7,1),
            ("9\nFinal\ntest",8,1),
            ("10\nDiagnostics",9,1),
            ("11\nInterpretation",10,1),
            ("12\nResearch\nreport",11,1)
        ]
        edge_pairs=[(i,i+1) for i in range(len(nodes)-1)]
        fig=go.Figure()
        for a,b in edge_pairs:
            x0,y0=nodes[a][1],nodes[a][2]; x1,y1=nodes[b][1],nodes[b][2]
            fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0,xref='x',yref='y',axref='x',ayref='y',
                               showarrow=True,arrowhead=3,arrowsize=1,arrowwidth=2,arrowcolor='#7b8f98')
        for label,xv,yv in nodes:
            fig.add_trace(go.Scatter(x=[xv],y=[yv],mode='markers+text',
                marker=dict(size=54,line=dict(width=2,color='#176b87'),color='#eaf6f9'),
                text=[label],textposition='middle center',hoverinfo='skip',showlegend=False))
        fig.update_xaxes(visible=False,range=[-.7,11.7]); fig.update_yaxes(visible=False,range=[.2,1.8])
        fig.update_layout(height=260,template='plotly_white',margin=dict(l=15,r=15,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)
        st.info('The final test set is kept separate from model fitting and model-selection decisions. Cross-validation is performed within the training data.')

    with theory_tabs[1]:
        st.markdown("### 1. Linear Regression")
        st.latex(r"Y = \beta_0+\beta_1X_1+\cdots+\beta_pX_p+\varepsilon")
        st.write("Estimates a systematic linear relationship between predictors and the outcome. Coefficients describe the expected change in the outcome associated with a one-unit change in a predictor, conditional on the other included variables.")
        st.markdown("**Flow:** `X variables → linear equation → estimated coefficients → prediction`")

        st.markdown("### 2. Polynomial Regression")
        st.latex(r"Y=\beta_0+\beta_1X+\beta_2X^2+\cdots+\varepsilon")
        st.write("Extends a linear model by adding powers of predictors so that curved relationships can be represented.")
        st.markdown("**Flow:** `X → polynomial terms → linear estimation → nonlinear-shaped prediction`")

        st.markdown("### 3. Ridge Regression")
        st.latex(r"\min_{\beta}\left[\sum_i(y_i-\hat y_i)^2+\lambda\sum_j\beta_j^2\right]")
        st.write("Adds an L2 penalty to shrink coefficients. It can reduce coefficient instability when predictors are correlated or when many predictors are included.")
        st.markdown("**Flow:** `X → linear model + L2 penalty → shrunk coefficients → prediction`")

        st.markdown("### 4. Lasso Regression")
        st.latex(r"\min_{\beta}\left[\sum_i(y_i-\hat y_i)^2+\lambda\sum_j|\beta_j|\right]")
        st.write("Adds an L1 penalty. Some coefficients can be shrunk exactly to zero, giving a form of variable selection.")
        st.markdown("**Flow:** `X → linear model + L1 penalty → sparse coefficients → prediction`")

        st.markdown("### 5. Elastic Net")
        st.latex(r"\min_{\beta}\left[\sum_i(y_i-\hat y_i)^2+\lambda_1\sum_j|\beta_j|+\lambda_2\sum_j\beta_j^2\right]")
        st.write("Combines L1 and L2 regularisation and can be useful when predictors are numerous and/or correlated.")
        st.markdown("**Flow:** `X → L1 + L2 regularisation → stable/sparse solution → prediction`")

    with theory_tabs[2]:
        st.markdown("### 6. Random Forest")
        st.write("Builds many decision trees using resampled observations and random subsets of predictors, then aggregates their predictions.")
        st.markdown("**Flow:** `X → many decision trees → aggregate tree predictions → final prediction`")
        st.markdown("### 7. Gradient Boosting")
        st.write("Builds trees sequentially. Each new tree is fitted to improve the current ensemble by focusing on remaining prediction errors.")
        st.markdown("**Flow:** `X → initial prediction → residual/error correction → repeated trees → final prediction`")
        st.warning("Flexible AI models can capture nonlinearities and interactions, but high flexibility can also increase overfitting risk. Their performance must therefore be evaluated out of sample.")

    with theory_tabs[3]:
        st.markdown("### 8. Linear + AI Residual Hybrid")
        st.latex(r"Y=m(X)+\varepsilon")
        st.latex(r"\hat m_H(X)=\hat g(X)+\hat h(X)")
        st.write("The statistical model first captures an interpretable structural component. An AI learner then attempts to learn systematic information remaining in the statistical residuals.")
        st.markdown("#### Step-by-step")
        st.markdown("""
        1. Fit the statistical base model on training data.
        2. Generate **out-of-fold predictions** for the training observations.
        3. Calculate residuals from those out-of-fold predictions.
        4. Train the AI residual learner on those residuals.
        5. Refit the statistical base model on the complete training portion.
        6. Predict the test observations with the base model.
        7. Predict the residual correction with the AI learner.
        8. Add the two components.
        """)
        st.latex(r"e_i^{OOF}=Y_i-\hat g^{(-k(i))}(X_i)")
        st.latex(r"\hat Y_H=\hat g(X)+\hat h(X)")
        st.markdown("**Key research idea:** the hybrid is useful only if the residuals contain systematic, generalisable information rather than mainly random noise.")
        st.info("Residual hybridisation is an established modelling strategy. The research question is when and under what conditions it adds incremental out-of-sample information.")

    with theory_tabs[4]:
        st.markdown("### Performance measures")
        st.latex(r"RMSE=\sqrt{\frac{1}{n}\sum_i(Y_i-\hat Y_i)^2}")
        st.latex(r"MAE=\frac{1}{n}\sum_i|Y_i-\hat Y_i|")
        st.latex(r"R^2=1-\frac{SSE}{SST}")
        st.write("RMSE and MAE are prediction-error measures; lower values indicate smaller errors. R² describes performance relative to a mean-baseline reference; higher values are generally better on the same evaluation sample.")

        st.markdown("### Validation logic")
        st.markdown("""
        **Training data:** fit models and perform internal tuning/cross-validation.

        **Cross-validation:** repeatedly divides the training data into fitting and validation folds to estimate expected performance.

        **Untouched test set:** used only for final out-of-sample confirmation.

        **Inference layer:** full-data OLS diagnostics answer a different question from hold-out prediction. Coefficients, standard errors, confidence intervals and hypothesis tests should not be inferred automatically for AI components.
        """)

with tabs[3]:
    st.subheader('📈 Predictive results')
    a,b,c,d=st.columns(4); a.metric('Selected model',selected_model); b.metric('Test RMSE',f"{selrow.RMSE:.4f}"); c.metric('Test MAE',f"{selrow.MAE:.4f}"); d.metric('Test R²',f"{selrow['R²']:.4f}")
    st.caption(f'Model selection criterion: {criterion}. Training observations: {len(ytr):,}; untouched test observations: {len(yte):,}.')
    st.dataframe(comparison[['Model','Type','Train RMSE','Train MAE','Train R²','CV RMSE','CV RMSE SD','CV MAE','CV MAE SD','CV R²','CV R² SD','RMSE','MAE','R²','Generalization Gap R²','Average Test Rank']].style.format({c:'{:.4f}' for c in comparison.columns if c not in ['Model','Type'] and 'Rank' not in c}),use_container_width=True,hide_index=True)
    fig=px.bar(comparison.sort_values('RMSE'),x='Model',y='RMSE',title='Test RMSE by model',text_auto='.3f'); fig.update_layout(template='plotly_white',height=500,xaxis_tickangle=-35); st.plotly_chart(fig,use_container_width=True)
    st.markdown('### Actual vs predicted')
    long=pd.DataFrame({'Actual':np.tile(yte.to_numpy(),len(preds)),'Predicted':np.concatenate(list(preds.values())),'Model':np.repeat(list(preds.keys()),len(yte))})
    fig=px.scatter(long,x='Actual',y='Predicted',facet_col='Model',facet_col_wrap=2,trendline=None,title='Hold-out test predictions'); mn=min(long.Actual.min(),long.Predicted.min()); mx=max(long.Actual.max(),long.Predicted.max()); fig.add_shape(type='line',x0=mn,x1=mx,y0=mn,y1=mx,line_dash='dash',row='all',col='all'); fig.update_layout(template='plotly_white',height=850); st.plotly_chart(fig,use_container_width=True)

with tabs[4]:
    st.subheader('🔬 Hybrid model analysis')
    lin=comparison[comparison.Model=='Linear Regression'].iloc[0]; hyb=comparison[comparison.Model=='Linear + AI Residual Hybrid'].iloc[0]
    rmse_imp=lin.RMSE-hyb.RMSE; mae_imp=lin.MAE-hyb.MAE; pct=100*rmse_imp/lin.RMSE
    a,b,c=st.columns(3); a.metric('Linear test RMSE',f'{lin.RMSE:.4f}'); b.metric('Hybrid test RMSE',f'{hyb.RMSE:.4f}'); c.metric('Hybrid RMSE change',f'{pct:+.2f}%')
    st.write(f'RMSE difference (Linear − Hybrid): **{rmse_imp:.6f}**. MAE difference: **{mae_imp:.6f}**.')
    st.info('Positive improvement means the hybrid had lower error than the linear model on this particular hold-out split. It does not establish universal superiority.')
    resid=oof
    r1,r2,r3=st.columns(3); r1.metric('OOF residual SD',f'{np.std(resid):.4f}'); r2.metric('OOF residual mean',f'{np.mean(resid):.4f}'); r3.metric('OOF residual MAE',f'{np.mean(np.abs(resid)):.4f}')
    fig=px.histogram(x=resid,nbins=30,title='Out-of-fold statistical residuals'); fig.update_layout(template='plotly_white'); st.plotly_chart(fig,use_container_width=True)
    st.markdown('### What this tells you')
    st.write('If residuals retain systematic structure that the AI learner can generalise, the hybrid can add predictive information. If residuals are mostly random noise, the AI component has little genuine signal to learn and may overfit.')

# ----------------------------- diagnostics -----------------------------
Xdiag=pd.DataFrame(index=X.index)
for c in x_cols:
    if pd.api.types.is_numeric_dtype(X[c]): Xdiag[c]=pd.to_numeric(X[c],errors='coerce')
    else: Xdiag=pd.concat([Xdiag,pd.get_dummies(X[c].astype('string').fillna('Missing'),prefix=c,drop_first=True,dtype=float)],axis=1)
Xdiag=Xdiag.replace([np.inf,-np.inf],np.nan).fillna(Xdiag.median(numeric_only=True)); Xdiag=Xdiag.loc[:,Xdiag.nunique()>1]
try:
    osm=sm.OLS(y,sm.add_constant(Xdiag.astype(float),has_constant='add')).fit(); conf=osm.conf_int(); coef=pd.DataFrame({'Variable':osm.params.index,'Coefficient':osm.params.values,'Std. Error':osm.bse.values,'t':osm.tvalues.values,'p-value':osm.pvalues.values,'95% CI Lower':conf[0].values,'95% CI Upper':conf[1].values})
    sst=float(np.sum((y-y.mean())**2)); ssr=float(np.sum(osm.resid**2)); verified=1-ssr/sst
    anova=pd.DataFrame({'Source':['Regression','Residual','Total'],'Sum of Squares':[sst-ssr,ssr,sst],'df':[osm.df_model,osm.df_resid,osm.df_model+osm.df_resid]})
    bp=het_breuschpagan(osm.resid,osm.model.exog); bp_p=bp[1]
except Exception as e:
    osm=None; coef=pd.DataFrame(); anova=pd.DataFrame(); verified=np.nan; bp_p=np.nan

with tabs[5]:
    st.subheader('🩺 Diagnostics & statistical inference')
    if osm is None: st.warning('OLS diagnostics could not be calculated for this specification.');
    else:
        a,b,c,d,e=st.columns(5); a.metric('Full-data OLS R²',f'{osm.rsquared:.4f}'); b.metric('Adjusted R²',f'{osm.rsquared_adj:.4f}'); c.metric('F-statistic',f'{osm.fvalue:.3f}'); d.metric('Model p-value',f'{osm.f_pvalue:.3g}'); e.metric('OLS observations',f'{int(osm.nobs):,}')
        st.markdown('### OLS result verification')
        v1,v2,v3=st.columns(3); v1.metric('Statsmodels R²',f'{osm.rsquared:.6f}'); v2.metric('Reproduced R²',f'{verified:.6f}'); v3.metric('Absolute difference',f'{abs(osm.rsquared-verified):.2e}')
        st.success('✓ R² verification passed: the independently reproduced 1 − SSR/SST value agrees with the OLS result to numerical precision.') if abs(osm.rsquared-verified)<1e-10 else st.warning('R² verification requires review.')
        with st.expander('OLS coefficients'): st.dataframe(coef,use_container_width=True,hide_index=True)
        with st.expander('ANOVA'): st.dataframe(anova,use_container_width=True,hide_index=True)
        st.markdown('### Multicollinearity')
        if Xdiag.shape[1]>=2:
            vr=[]
            for i,c in enumerate(Xdiag.columns):
                try:v=float(variance_inflation_factor(Xdiag.astype(float).values,i))
                except:v=np.inf
                vr.append({'Variable':c,'VIF':v,'Interpretation':'Very high' if v>=10 else ('High / potentially problematic' if v>=5 else 'No strong VIF indication')})
            vif=pd.DataFrame(vr).sort_values('VIF',ascending=False); st.dataframe(vif,use_container_width=True,hide_index=True); st.caption('VIF mainly concerns coefficient stability and standard errors. It does not by itself determine predictive performance.')
        else: vif=pd.DataFrame(); st.info('VIF requires at least two usable predictor columns.')
        st.markdown('### Heteroscedasticity check')
        st.metric('Breusch–Pagan p-value',f'{bp_p:.4g}' if np.isfinite(bp_p) else 'N/A'); st.caption('A small p-value can indicate non-constant error variance. This test is diagnostic, not a complete model-validity proof.')
        with st.expander('Why OLS R² and predictive R² do not need to match',expanded=True):
            st.write(f'OLS inference uses all {len(y):,} usable observations. Predictive evaluation fits models on {len(ytr):,} training observations and evaluates them on {len(yte):,} untouched test observations. They answer different questions, so different R² values are expected.')

with tabs[6]:
    st.subheader('💡 Results interpretation — what the numbers mean')
    st.caption('Interpretation is generated from the current dataset and validation results. It is conditional on the chosen variables, preprocessing, model settings and validation design.')

    int_tabs = st.tabs(['🎯 Predictive results','📐 Statistical inference','🔗 Hybrid results','⚠️ Diagnostics & limitations','📝 Research conclusion'])

    best_rmse=comparison.loc[comparison.RMSE.idxmin()]
    best_cv=comparison.loc[comparison['CV RMSE'].idxmin()]
    gap=float(selrow['Generalization Gap R²'])

    with int_tabs[0]:
        st.markdown("### Test-set performance")
        st.write(f"On the current untouched test set, **{best_rmse.Model}** has the lowest RMSE (**{best_rmse.RMSE:.4f}**) among the models included in this run.")
        st.write(f"The model with the lowest mean cross-validation RMSE is **{best_cv.Model}** (**{best_cv['CV RMSE']:.4f}**).")
        st.info("These statements describe this dataset and this validation design. They should not be interpreted as a universal ranking of algorithms.")
        a,b,c=st.columns(3)
        a.metric("Selected model",selected_model)
        b.metric("Test RMSE",f"{selrow.RMSE:.4f}")
        c.metric("Test R²",f"{selrow['R²']:.4f}")
        st.markdown("**How to read the metrics:**")
        st.markdown("""
        - **RMSE:** typical error on the outcome scale, with larger errors receiving more weight.
        - **MAE:** average absolute prediction error and generally less sensitive to extreme errors than RMSE.
        - **R²:** relative explanatory/predictive performance on the evaluation sample.
        - **CV SD:** variation in performance across validation folds; larger values indicate less stable fold-to-fold performance.
        """)

    with int_tabs[1]:
        st.markdown("### Full-data OLS interpretation")
        if osm is not None:
            sig=coef[(coef['Variable']!='const') & (coef['p-value']<.05)]
            st.write(f"The full-data OLS model has **R² = {osm.rsquared:.4f}** and adjusted R² = **{osm.rsquared_adj:.4f}**.")
            st.write(f"In this fitted specification, **{len(sig)}** displayed non-intercept terms have p < 0.05.")
            st.dataframe(coef,use_container_width=True,hide_index=True)
            st.info("A statistically significant coefficient is evidence against a zero coefficient under the fitted model and its assumptions. It is not, by itself, proof of causality, practical importance, or superior prediction.")
            st.markdown("### Why this can differ from test R²")
            st.write(f"OLS inference uses all {len(y):,} usable observations, while predictive evaluation uses {len(ytr):,} training observations and {len(yte):,} untouched test observations. Different R² values are therefore expected.")
        else:
            st.warning("OLS inference was not available for this specification.")

    with int_tabs[2]:
        st.markdown("### Does the AI residual component add information?")
        lin=comparison[comparison.Model=='Linear Regression'].iloc[0]
        hyb=comparison[comparison.Model=='Linear + AI Residual Hybrid'].iloc[0]
        rmse_imp=float(lin.RMSE-hyb.RMSE)
        mae_imp=float(lin.MAE-hyb.MAE)
        pct=float(100*rmse_imp/lin.RMSE) if lin.RMSE else np.nan
        a,b,c=st.columns(3)
        a.metric("Linear RMSE",f"{lin.RMSE:.4f}")
        b.metric("Hybrid RMSE",f"{hyb.RMSE:.4f}")
        c.metric("RMSE change",f"{pct:+.2f}%")
        if rmse_imp>0:
            st.success(f"On this hold-out split, the hybrid has lower RMSE than linear regression by {rmse_imp:.4f}. This is evidence of incremental value for this evaluation, not a universal conclusion.")
        elif rmse_imp<0:
            st.warning(f"On this hold-out split, the hybrid has higher RMSE than linear regression by {-rmse_imp:.4f}. The residual learner did not improve this particular evaluation.")
        else:
            st.info("The hybrid and linear model have essentially the same test RMSE on this split.")
        st.write(f"The OOF residual mean is **{np.mean(oof):.4f}** and OOF residual SD is **{np.std(oof):.4f}**.")
        st.markdown("**Research interpretation:** the central question is whether residual structure is systematic and generalisable. A large residual variance alone does not prove that an AI learner can predict it.")

    with int_tabs[3]:
        st.markdown("### Generalisation")
        st.write(f"The selected model has a train-to-test R² gap of **{gap:.4f}**.")
        if gap>0.15:
            st.warning("The train-to-test gap is relatively large for this run and warrants investigation for overfitting, data heterogeneity, or validation instability.")
        elif gap>0.05:
            st.info("There is a noticeable train-to-test gap. Compare CV variability and diagnostics before drawing conclusions.")
        else:
            st.success("The train-to-test R² gap is relatively small in this run. This is encouraging, but it is not a formal proof of generalisation.")
        if 'vif' in locals() and not vif.empty:
            high=vif[vif.VIF>=5]['Variable'].tolist()
            st.markdown("### Multicollinearity")
            st.write("Variables with VIF ≥ 5: **"+(", ".join(high) if high else "none detected by this threshold.")+"**.")
            st.caption("Multicollinearity mainly affects coefficient stability and standard errors. It does not automatically mean that predictive performance is poor.")
        if np.isfinite(bp_p):
            st.markdown("### Breusch–Pagan diagnostic")
            st.write(f"Breusch–Pagan p-value: **{bp_p:.4g}**.")
            st.caption("A small p-value can indicate non-constant error variance. This is one diagnostic and should be considered with residual plots and the modelling context.")

    with int_tabs[4]:
        st.markdown("### Research-level conclusion")
        st.write(
            f"The current analysis compares interpretable statistical models, regularised statistical models, flexible AI models and a residual hybrid under the selected validation design. "
            f"The selected model by **{criterion}** is **{selected_model}**, but the evidence remains conditional on this dataset and analysis design."
        )
        st.markdown("""
        **For a research report, state separately:**
        1. What the predictive metrics show.
        2. What the OLS coefficients and inferential diagnostics show.
        3. Whether the hybrid changes out-of-sample performance.
        4. Whether diagnostics reveal possible model weaknesses.
        5. What limitations remain.

        **Do not conclude that an algorithm is universally best from one dataset or one train/test split.**
        """)
        st.info("The platform is designed to support transparent comparison rather than a blind 'best model' decision.")

with tabs[7]:
    st.subheader('🎨 Chart Studio')
    st.caption('Choose what you want to visualise. This section is independent from the model-selection criterion.')
    chart=st.selectbox('Chart preference',['Scatter plot','Histogram','Box plot','Bar chart','Line / trend','Correlation heatmap','Prediction error distribution','CV RMSE with uncertainty'])
    numeric=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]; categorical=[c for c in df.columns if c not in numeric]
    if chart=='Scatter plot':
        c1,c2,c3=st.columns(3); xx=c1.selectbox('X',numeric,index=0); yy=c2.selectbox('Y',numeric,index=min(1,len(numeric)-1)); cc=c3.selectbox('Colour',['None']+categorical)
        p=df[[xx,yy]+([] if cc=='None' else [cc])].dropna(); fig=px.scatter(p,x=xx,y=yy,color=None if cc=='None' else cc,trendline='ols' if len(p)>=10 else None,title=f'{yy} vs {xx}')
    elif chart=='Histogram':
        v=st.selectbox('Variable',numeric); fig=px.histogram(df,x=v,nbins=30,marginal='box',title=f'Distribution of {v}')
    elif chart=='Box plot':
        c1,c2=st.columns(2); v=c1.selectbox('Numeric variable',numeric); g=c2.selectbox('Group',['None']+categorical); fig=px.box(df,y=v,x=None if g=='None' else g,points='outliers',title=f'Box plot of {v}')
    elif chart=='Bar chart':
        c1,c2=st.columns(2); g=c1.selectbox('Category',categorical or df.columns.tolist()); v=c2.selectbox('Numeric measure',numeric); p=df.groupby(g,dropna=False)[v].mean().reset_index(); fig=px.bar(p,x=g,y=v,title=f'Mean {v} by {g}')
    elif chart=='Line / trend':
        v=st.selectbox('Numeric variable',numeric); fig=px.line(df.reset_index(),x='index',y=v,title=f'{v} across row order')
    elif chart=='Correlation heatmap':
        corr=df[numeric].corr(numeric_only=True); fig=px.imshow(corr,text_auto='.2f',aspect='auto',title='Pearson correlation matrix')
    elif chart=='Prediction error distribution':
        err=pd.DataFrame({m:yte.to_numpy()-p for m,p in preds.items()}); lm=err.melt(var_name='Model',value_name='Error'); fig=px.histogram(lm,x='Error',color='Model',barmode='overlay',nbins=35,title='Test prediction error distributions')
    else:
        p=comparison.sort_values('CV RMSE'); fig=go.Figure(go.Bar(x=p.Model,y=p['CV RMSE'],error_y=dict(type='data',array=p['CV RMSE SD'].fillna(0)))); fig.update_layout(title='Mean cross-validation RMSE ± SD',template='plotly_white')
    fig.update_layout(template='plotly_white',height=600,margin=dict(l=20,r=20,t=70,b=30)); st.plotly_chart(fig,use_container_width=True)

with tabs[8]:
    st.subheader('📄 Research report & export')
    pred_table=Xte.reset_index(drop=True).copy(); pred_table.insert(0,'Actual',yte.reset_index(drop=True));
    for m,p in preds.items(): pred_table[f'Predicted — {m}']=p
    report_text=f'''STATISTICAL–AI HYBRID MODELLING PLATFORM\n\nOutcome: {y_col}\nPredictors: {", ".join(x_cols)}\nObservations: {len(y)}\nTraining observations: {len(ytr)}\nTest observations: {len(yte)}\nTest proportion: {test_size}\nRandom seed: {seed}\nCV folds: {folds}\nSelection criterion: {criterion}\nSelected model: {selected_model}\n\nMODEL PERFORMANCE\n{comparison.to_string(index=False)}\n\nINTERPRETATION\nResults are conditional on this dataset, preprocessing, model settings and validation design. Full-data OLS inference and hold-out predictive evaluation are separate analyses. Hybrid incremental value should be assessed out-of-sample and should not be assumed a priori.\n'''
    st.text_area('Report preview',report_text,height=420)
    c1,c2,c3=st.columns(3)
    c1.download_button('⬇️ Download comparison CSV',comparison.to_csv(index=False).encode(),file_name='model_comparison.csv',mime='text/csv')
    c2.download_button('⬇️ Download predictions CSV',pred_table.to_csv(index=False).encode(),file_name='test_predictions.csv',mime='text/csv')
    c3.download_button('⬇️ Download research report TXT',report_text.encode(),file_name='statistical_ai_report.txt',mime='text/plain')
    excel=io.BytesIO()
    with pd.ExcelWriter(excel,engine='openpyxl') as w:
        df.to_excel(w,'Data',index=False); comparison.to_excel(w,'Model Comparison',index=False); pred_table.to_excel(w,'Test Predictions',index=False); coef.to_excel(w,'OLS Coefficients',index=False); anova.to_excel(w,'ANOVA',index=False)
        if 'vif' in locals() and not vif.empty: vif.to_excel(w,'VIF',index=False)
    st.download_button('📘 Download complete Excel report',excel.getvalue(),file_name='statistical_ai_complete_report.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.markdown('### Reproducibility note')
    st.write('The report records the dataset dimensions, selected variables, validation settings, model settings, model-comparison metrics and inference results. For publication-grade work, retain the original dataset, preprocessing decisions, software versions and the final analysis script alongside the exported report.')

st.divider(); st.caption('StatAI • Statistical–AI Hybrid Modelling Platform • Theory → Modelling → Validation → Diagnostics → Interpretation • Research and educational use')
