import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import pickle
import base64

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import (IsolationForest, RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, r2_score, classification_report,
                             confusion_matrix, mean_absolute_error, mean_squared_error,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Diamond Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Cinzel:wght@400;600;700&display=swap');

:root {
    --bg: #06080f;
    --surface: #0b0e1a;
    --surface2: #0f1220;
    --surface3: #131828;
    --border: #1e2440;
    --border-bright: #2d3560;

    --diamond-white: #e8f0ff;
    --diamond-ice: #c5d8ff;
    --diamond-blue: #7aa0ff;
    --diamond-deep: #4060d4;
    --diamond-shimmer: #a8c8ff;
    --diamond-fire: #ff9d6e;
    --diamond-pink: #e87fff;
    --diamond-teal: #5fffd4;

    --gold: #d4af6e;
    --gold-light: #f0d090;
    --gold-dim: #a08040;

    --text: #dce8ff;
    --text-dim: #8090b0;
    --text-muted: #445070;

    --font-display: 'Cinzel', serif;
    --font-body: 'Cormorant Garamond', serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 17px !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* subtle noise texture overlay */
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0; opacity: 0.4;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    letter-spacing: .08em !important;
    color: var(--diamond-white) !important;
}
h4, h5, h6 {
    font-family: var(--font-display) !important;
    letter-spacing: .06em !important;
    color: var(--diamond-ice) !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
}

/* ── Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--diamond-blue) !important;
    box-shadow: 0 0 0 2px rgba(122,160,255,0.15) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] { background: var(--diamond-blue) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--diamond-blue) !important; }

/* ── Radio ── */
[data-testid="stRadio"] label { color: var(--text) !important; font-family: var(--font-mono) !important; font-size: 13px !important; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--diamond-deep) 0%, var(--diamond-blue) 100%) !important;
    color: #fff !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 10px 24px !important;
    transition: all .2s ease !important;
    box-shadow: 0 4px 20px rgba(64,96,212,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #5070e8 0%, #99bfff 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(100,160,255,0.4) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border-bright) !important;
    border-radius: 8px !important;
    background: var(--surface2) !important;
    padding: 16px !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--diamond-ice) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-family: var(--font-mono) !important; font-size: 11px !important; letter-spacing: .1em !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    border: 1px solid var(--border-bright) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-muted) !important;
    font-family: var(--font-display) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .1em !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    white-space: nowrap !important;
    transition: all .15s ease !important;
    text-transform: uppercase !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--diamond-ice) !important; }
.stTabs [aria-selected="true"] {
    color: var(--diamond-shimmer) !important;
    border-bottom: 2px solid var(--diamond-blue) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 28px !important; }

hr { border-color: var(--border) !important; }

/* ── Custom Components ── */
.diamond-hero {
    text-align: center;
    padding: 56px 0 40px;
    position: relative;
}
.diamond-icon {
    font-size: 52px;
    line-height: 1;
    margin-bottom: 16px;
    filter: drop-shadow(0 0 24px rgba(122,160,255,0.6));
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.hero-title {
    font-family: var(--font-display) !important;
    font-size: clamp(28px, 5vw, 52px);
    font-weight: 700;
    letter-spacing: .1em;
    color: var(--diamond-white);
    line-height: 1.1;
}
.hero-title .accent { color: var(--diamond-blue); }
.hero-sub {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 10px;
    letter-spacing: .15em;
    text-transform: uppercase;
}

.step-badge {
    display: flex;
    align-items: center;
    gap: 14px;
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 600;
    color: var(--diamond-white);
    margin-bottom: 28px;
    letter-spacing: .06em;
}
.step-num {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--diamond-deep), var(--diamond-blue));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
    font-family: var(--font-display);
    color: white;
    box-shadow: 0 4px 16px rgba(64,96,212,0.4);
    flex-shrink: 0;
}

.pipeline-rail {
    display: flex;
    margin-bottom: 36px;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
}
.rail-step {
    flex: 1;
    padding: 10px 4px;
    text-align: center;
    font-size: 9px;
    font-family: var(--font-display);
    color: var(--text-muted);
    background: var(--surface);
    border-right: 1px solid var(--border);
    letter-spacing: .08em;
    text-transform: uppercase;
    transition: all .2s;
}
.rail-step:last-child { border-right: none; }
.rail-step.done {
    background: rgba(122,160,255,0.07);
    color: var(--diamond-shimmer);
}
.rail-step.active {
    background: linear-gradient(135deg, var(--diamond-deep), var(--diamond-blue));
    color: #fff;
    font-weight: 600;
}

.gem-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}
.gem-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--diamond-blue), transparent);
}
.gem-card .gc-label {
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: .15em;
    text-transform: uppercase;
    font-family: var(--font-mono);
    margin-bottom: 6px;
}
.gem-card .gc-value {
    font-family: var(--font-display);
    font-size: 30px;
    font-weight: 700;
    color: var(--diamond-ice);
    letter-spacing: .02em;
}

.cut-pill {
    display: inline-block;
    background: rgba(122,160,255,0.1);
    border: 1px solid rgba(122,160,255,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 11px;
    color: var(--diamond-shimmer);
    font-family: var(--font-mono);
    margin: 2px;
    letter-spacing: .04em;
}
.cut-pill.gold {
    background: rgba(212,175,110,0.1);
    border-color: rgba(212,175,110,0.3);
    color: var(--gold-light);
}

.section-rule {
    border: none;
    border-top: 1px solid var(--border);
    margin: 32px 0;
}

.download-gem {
    display: inline-block;
    background: linear-gradient(135deg, rgba(122,160,255,0.15), rgba(64,96,212,0.15));
    border: 1px solid var(--diamond-blue);
    color: var(--diamond-ice);
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 11px;
    letter-spacing: .12em;
    text-transform: uppercase;
    border-radius: 3px;
    padding: 10px 24px;
    text-decoration: none;
    margin-top: 8px;
    transition: all .2s;
}
.download-gem:hover { background: linear-gradient(135deg, rgba(122,160,255,0.3), rgba(64,96,212,0.3)); }

/* Diamond facts sidebar */
.fact-box {
    background: linear-gradient(135deg, rgba(122,160,255,0.06), rgba(64,96,212,0.06));
    border: 1px solid var(--border-bright);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.fact-box .fact-label {
    font-size: 10px;
    color: var(--gold);
    letter-spacing: .15em;
    text-transform: uppercase;
    font-family: var(--font-mono);
    margin-bottom: 8px;
}
.fact-box .fact-text {
    font-family: var(--font-body);
    font-size: 15px;
    color: var(--text-dim);
    font-style: italic;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    'df': None, 'problem_type': 'Regression', 'target_col': None,
    'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
    'best_model': None, 'selected_features': None,
    'pipeline_stage': 0, 'model_comparison': {},
    '_scaler': None, '_Xte_scaled': None, '_yte': None,
    '_Xtr_scaled': None, '_ytr': None, 'tuned_model': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
# Diamond-tuned Plotly theme
DTHEME = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(11,14,26,0.8)',
    font=dict(family='DM Mono, monospace', color='#8090b0', size=11),
    xaxis=dict(gridcolor='#1e2440', linecolor='#1e2440', zerolinecolor='#1e2440'),
    yaxis=dict(gridcolor='#1e2440', linecolor='#1e2440', zerolinecolor='#1e2440'),
)
def dtheme(**kw):
    d = DTHEME.copy(); d.update(kw)
    return d

D_ICE    = '#c5d8ff'
D_BLUE   = '#7aa0ff'
D_DEEP   = '#4060d4'
D_FIRE   = '#ff9d6e'
D_PINK   = '#e87fff'
D_TEAL   = '#5fffd4'
D_GOLD   = '#d4af6e'
D_PALETTE = [D_BLUE, D_TEAL, D_FIRE, D_PINK, D_GOLD, D_ICE]

# Color scales
CS_BLUE  = [[0,'#06080f'],[0.5,'#2040a0'],[1,'#c5d8ff']]
CS_FIRE  = [[0,'#06080f'],[0.5,'#a04020'],[1,'#ff9d6e']]
CS_TEAL  = [[0,'#06080f'],[0.5,'#207060'],[1,'#5fffd4']]

def step_badge(n, title):
    st.markdown(f'<div class="step-badge"><span class="step-num">{n}</span>{title}</div>', unsafe_allow_html=True)

def section_hr():
    st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)

def get_download_link(obj, filename, label):
    buf = pickle.dumps(obj)
    b64 = base64.b64encode(buf).decode()
    return f'<a class="download-gem" href="data:application/octet-stream;base64,{b64}" download="{filename}">⬧  {label}</a>'

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────
REGRESSION_MODELS = {
    "Linear Regression":    LinearRegression(),
    "Ridge Regression":     Ridge(alpha=1.0),
    "SVR (RBF)":            SVR(kernel='rbf'),
    "KNN Regressor":        KNeighborsRegressor(n_neighbors=5),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
}
CLASSIFICATION_MODELS = {
    "Logistic Regression":  LogisticRegression(max_iter=2000, n_jobs=-1),
    "SVC (RBF)":            SVC(kernel='rbf', probability=True),
    "KNN Classifier":       KNeighborsClassifier(n_neighbors=5),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
}
GRIDS = {
    "SVR":                      {'C':[0.01,0.1,1,10],'kernel':['rbf'],'epsilon':[0.01,0.1,0.5]},
    "SVC":                      {'C':[0.01,0.1,1,10,100],'kernel':['rbf'],'gamma':['scale','auto']},
    "LogisticRegression":       {'C':[0.001,0.01,0.1,1,10,100],'penalty':['l2']},
    "LinearRegression":         {'fit_intercept':[True,False]},
    "Ridge":                    {'alpha':[0.001,0.01,0.1,1,10,100]},
    "KNeighborsClassifier":     {'n_neighbors':[3,5,7,9,11],'weights':['uniform','distance']},
    "KNeighborsRegressor":      {'n_neighbors':[3,5,7,9,11],'weights':['uniform','distance']},
    "RandomForestRegressor":    {'n_estimators':[50,100,200],'max_depth':[None,5,10,20],'min_samples_split':[2,5,10]},
    "RandomForestClassifier":   {'n_estimators':[50,100,200],'max_depth':[None,5,10,20]},
    "GradientBoostingRegressor":{'n_estimators':[50,100,200],'learning_rate':[0.01,0.05,0.1],'max_depth':[3,5,7]},
    "GradientBoostingClassifier":{'n_estimators':[50,100],'learning_rate':[0.05,0.1,0.2],'max_depth':[3,5]},
}

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="diamond-hero">
  <div class="diamond-icon">💎</div>
  <div class="hero-title">Auto<span class="accent">ML</span> Studio</div>
  <div class="hero-sub">Gemstone Price Intelligence · Predict · Analyse · Refine</div>
</div>
""", unsafe_allow_html=True)

# Pipeline Progress Rail
stage   = st.session_state.pipeline_stage
labels  = ['Setup','EDA','Clean','Features','Split','Compare','Train','Tune']
def rail_cls(i): return 'done' if i < stage else ('active' if i == stage else '')
rail_html = ''.join([f'<div class="rail-step {rail_cls(i)}">{l}</div>' for i,l in enumerate(labels)])
st.markdown(f'<div class="pipeline-rail">{rail_html}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "💎  Setup",
    "🔬  EDA",
    "🧹  Clean",
    "⚖️  Features",
    "✂️  Split",
    "📊  Compare",
    "🧠  Train",
    "⚙️  Tune",
])

# ══════════════════════════════════════════════
# TAB 1 — SETUP
# ══════════════════════════════════════════════
with tabs[0]:
    step_badge(1, "Problem Setup & Data Ingestion")

    col_left, col_main = st.columns([1, 2.5], gap="large")
    with col_left:
        st.markdown("##### Task Type")
        st.session_state.problem_type = st.radio(
            "Select task:", ("Regression", "Classification"),
            index=0 if st.session_state.problem_type == "Regression" else 1,
            label_visibility="collapsed"
        )
        st.markdown(f'<div class="cut-pill gold">{st.session_state.problem_type}</div>', unsafe_allow_html=True)

    with col_main:
        st.markdown("##### Upload Diamond Dataset (CSV)")
        uploaded = st.file_uploader("Drop diamonds.csv here", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df = pd.read_csv(uploaded)
            # clean unnamed cols
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            st.session_state.df = df
            if st.session_state.pipeline_stage == 0:
                st.session_state.pipeline_stage = 1
            st.success(f"✦  Loaded **{uploaded.name}** — {df.shape[0]:,} stones × {df.shape[1]} attributes")

    if st.session_state.df is not None:
        df = st.session_state.df
        section_hr()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Stones", f"{df.shape[0]:,}")
        c2.metric("Attributes", f"{df.shape[1]}")
        c3.metric("Numeric", f"{df.select_dtypes(include=np.number).shape[1]}")
        c4.metric("Categorical", f"{df.select_dtypes(include='object').shape[1]}")

        st.markdown("##### Data Preview")
        st.dataframe(df.head(10), use_container_width=True, height=280)

        section_hr()
        col_t, col_pca = st.columns([1, 2], gap="large")
        with col_t:
            st.markdown("##### Target Variable")
            target_col = st.selectbox("Select target:", df.columns.tolist(),
                                       index=list(df.columns).index('price') if 'price' in df.columns else len(df.columns)-1,
                                       label_visibility="collapsed")
            st.session_state.target_col = target_col
            st.markdown(f'<div class="cut-pill gold">Target: {target_col}</div>', unsafe_allow_html=True)

        with col_pca:
            st.markdown("##### PCA — 2D Gemstone Map")
            num_feats = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=np.number).columns.tolist()
            sel_feats = st.multiselect("Features for PCA:", num_feats,
                                        default=num_feats[:min(6, len(num_feats))],
                                        label_visibility="collapsed")
            if st.button("⬧  Project to 2D"):
                if len(sel_feats) >= 2:
                    Xp = df[sel_feats].dropna()
                    comps = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Xp))
                    pca_df = pd.DataFrame(comps, columns=['PC 1', 'PC 2'])
                    if target_col in df.columns:
                        pca_df['Target'] = df[target_col].dropna().iloc[:len(pca_df)].values
                        fig = px.scatter(pca_df, x='PC 1', y='PC 2', color='Target',
                                         color_continuous_scale=CS_BLUE,
                                         title='PCA — Diamond Manifold Projection',
                                         opacity=0.6)
                    else:
                        fig = px.scatter(pca_df, x='PC 1', y='PC 2',
                                         color_discrete_sequence=[D_BLUE], opacity=0.6)
                    fig.update_layout(**dtheme(height=360))
                    fig.update_traces(marker=dict(size=3, opacity=0.6))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Select at least 2 features.")

# ══════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════
with tabs[1]:
    step_badge(2, "Exploratory Data Analysis")
    if st.session_state.df is None:
        st.info("Load a dataset in Setup first.")
    else:
        df = st.session_state.df
        num_df = df.select_dtypes(include=np.number)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("##### Descriptive Statistics")
            st.dataframe(df.describe().T.style.format(precision=3)
                         .background_gradient(cmap='Blues', subset=['mean', 'std']),
                         use_container_width=True, height=300)
        with c2:
            st.markdown("##### Missing Values")
            miss = df.isnull().sum()
            miss_df = pd.DataFrame({'Column': miss.index, 'Missing': miss.values,
                                    'Pct (%)': (miss.values / len(df) * 100).round(2)})
            miss_df = miss_df[miss_df['Missing'] > 0]
            if miss_df.empty:
                st.success("✦  No missing values — perfect clarity.")
            else:
                st.dataframe(miss_df, use_container_width=True)

        section_hr()
        st.markdown("##### Correlation Matrix — The 4Cs & Dimensions")
        if not num_df.empty:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale=CS_BLUE)
            fig.update_layout(**dtheme(height=440))
            st.plotly_chart(fig, use_container_width=True)

        section_hr()
        cl, cr = st.columns(2, gap="large")
        with cl:
            st.markdown("##### Feature Distribution")
            feat_dist = st.selectbox("Feature:", df.columns.tolist(), key="dist_feat")
            fig2 = px.histogram(df, x=feat_dist, marginal="box", nbins=60,
                                color_discrete_sequence=[D_BLUE])
            fig2.update_layout(**dtheme(height=340, bargap=0.04))
            st.plotly_chart(fig2, use_container_width=True)

        with cr:
            st.markdown("##### Feature vs Target Scatter")
            if st.session_state.target_col and pd.api.types.is_numeric_dtype(df.get(st.session_state.target_col, pd.Series())):
                num_feats2 = [c for c in num_df.columns if c != st.session_state.target_col]
                if num_feats2:
                    x_feat = st.selectbox("X-axis:", num_feats2,
                                           index=num_feats2.index('carat') if 'carat' in num_feats2 else 0,
                                           key="scatter_feat")
                    color_by = None
                    if 'cut' in df.columns:
                        color_by = 'cut'
                    fig3 = px.scatter(df, x=x_feat, y=st.session_state.target_col,
                                      color=color_by, opacity=0.35,
                                      color_discrete_sequence=D_PALETTE)
                    fig3.update_layout(**dtheme(height=340))
                    fig3.update_traces(marker=dict(size=3))
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Select a numeric target in Setup.")

        section_hr()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            st.markdown("##### Categorical Distributions — Cut · Color · Clarity")
            cat_sel = st.selectbox("Column:", cat_cols, key="cat_vc")
            vc = df[cat_sel].value_counts().reset_index()
            vc.columns = [cat_sel, 'count']
            fig4 = px.bar(vc, x=cat_sel, y='count', color='count',
                          color_continuous_scale=CS_BLUE)
            fig4.update_layout(**dtheme(height=320))
            st.plotly_chart(fig4, use_container_width=True)

            # Price by category
            if st.session_state.target_col and st.session_state.target_col in df.columns:
                st.markdown(f"##### Price Distribution by {cat_sel.title()}")
                fig5 = px.box(df, x=cat_sel, y=st.session_state.target_col,
                              color=cat_sel, color_discrete_sequence=D_PALETTE)
                fig5.update_layout(**dtheme(height=360, showlegend=False))
                st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — CLEANING
# ══════════════════════════════════════════════
with tabs[2]:
    step_badge(3, "Data Cleansing & Feature Engineering")
    if st.session_state.df is None:
        st.info("Load a dataset first.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.markdown("##### Imputation")
        miss_cols = df.columns[df.isnull().any()].tolist()
        if miss_cols:
            ci1, ci2, ci3 = st.columns([2, 2, 1], gap="medium")
            with ci1: col_imp = st.selectbox("Column:", miss_cols)
            with ci2: strategy = st.selectbox("Strategy:", ["Mean", "Median", "Mode", "Drop Rows"])
            with ci3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Apply"):
                    s = st.session_state.df
                    if strategy == "Mean" and col_imp in num_cols:    s[col_imp].fillna(s[col_imp].mean(), inplace=True)
                    elif strategy == "Median" and col_imp in num_cols: s[col_imp].fillna(s[col_imp].median(), inplace=True)
                    elif strategy == "Mode":  s[col_imp].fillna(s[col_imp].mode()[0], inplace=True)
                    else: st.session_state.df = s.dropna(subset=[col_imp]).reset_index(drop=True)
                    st.success(f"✦  Imputed `{col_imp}` via {strategy}.")
        else:
            st.success("✦  No missing values — flawless clarity.")

        section_hr()
        st.markdown("##### Remove Columns")
        drop_cols = st.multiselect("Columns to drop:",
                                    [c for c in st.session_state.df.columns if c != st.session_state.target_col])
        if st.button("Drop Selected") and drop_cols:
            removed_cols = drop_cols.copy()
            st.session_state.df = st.session_state.df.drop(columns=drop_cols)
            st.success(f"✦  Dropped {len(removed_cols)} column(s).")
            with st.expander("📋 Details", expanded=True):
                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.markdown("**Removed Columns:**")
                    for col in removed_cols:
                        st.markdown(f"  • `{col}`")
                with col_det2:
                    st.markdown("**Updated Dataset:**")
                    st.markdown(f"  • Rows: `{st.session_state.df.shape[0]:,}`")
                    st.markdown(f"  • Columns: `{st.session_state.df.shape[1]}` (was {st.session_state.df.shape[1] + len(removed_cols)})")
            st.dataframe(st.session_state.df.head(5), use_container_width=True)

        section_hr()
        st.markdown("##### Outlier Detection")
        num_cols2 = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        co1, co2 = st.columns([1, 2], gap="medium")
        with co1: outlier_method = st.selectbox("Algorithm:", ["— None —", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        with co2:
            default_out_feats = [c for c in ['carat','depth','table','x','y','z'] if c in num_cols2]
            outlier_feats = st.multiselect("Dimensions:", num_cols2,
                                            default=default_out_feats or num_cols2[:min(4, len(num_cols2))])

        if outlier_method != "— None —" and outlier_feats:
            df_curr = st.session_state.df
            tmp = df_curr[outlier_feats].dropna()
            outliers = pd.Series(False, index=tmp.index)
            if outlier_method == "IQR":
                Q1, Q3 = tmp.quantile(.25), tmp.quantile(.75)
                IQR = Q3 - Q1
                outliers = ((tmp < (Q1 - 1.5 * IQR)) | (tmp > (Q3 + 1.5 * IQR))).any(axis=1)
            elif outlier_method == "Isolation Forest":
                contamination = st.slider("Contamination", .01, .2, .05)
                outliers = pd.Series(IsolationForest(contamination=contamination, random_state=42).fit_predict(tmp) == -1, index=tmp.index)
            elif outlier_method == "DBSCAN":
                eps_val = st.slider("ε (eps)", .1, 3.0, .7)
                preds = DBSCAN(eps=eps_val, min_samples=5).fit_predict(StandardScaler().fit_transform(tmp))
                outliers = pd.Series(preds == -1, index=tmp.index)
            elif outlier_method == "OPTICS":
                preds = OPTICS(min_samples=5).fit_predict(StandardScaler().fit_transform(tmp))
                outliers = pd.Series(preds == -1, index=tmp.index)

            n_out = int(outliers.sum())
            if n_out > 0:
                if len(outlier_feats) >= 2:
                    viz_df = tmp.copy()
                    viz_df['Type'] = outliers.map({True: 'Outlier', False: 'Normal'})
                    fig_out = px.scatter(viz_df, x=outlier_feats[0], y=outlier_feats[1],
                                        color='Type', color_discrete_map={'Normal': D_BLUE, 'Outlier': D_FIRE},
                                        opacity=0.5)
                    fig_out.update_layout(**dtheme(height=320))
                    fig_out.update_traces(marker=dict(size=4))
                    st.plotly_chart(fig_out, use_container_width=True)
                st.warning(f"⚠  {n_out:,} outlier stones detected via {outlier_method}.")
                if st.button(f"Remove {n_out:,} Outliers"):
                    st.session_state.df = st.session_state.df.drop(tmp.index[outliers]).reset_index(drop=True)
                    st.success(f"✦  Removed {n_out:,} stones. Dataset now {len(st.session_state.df):,} rows.")
            else:
                st.success(f"✦  No outliers found via {outlier_method}.")

        section_hr()
        st.markdown("##### Categorical Encoding")
        cat_cols_enc = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols_enc:
            enc_c1, enc_c2 = st.columns([2, 1])
            with enc_c1: enc_sel = st.multiselect("Columns:", cat_cols_enc, default=cat_cols_enc)
            with enc_c2: enc_method = st.selectbox("Method:", ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"])
            if st.button("Encode"):
                df_enc = st.session_state.df.copy()
                cols_before = df_enc.shape[1]
                if enc_method == "Label Encoding":
                    le = LabelEncoder()
                    for col in enc_sel:
                        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                elif enc_method == "One-Hot Encoding":
                    df_enc = pd.get_dummies(df_enc, columns=enc_sel, drop_first=False)
                elif enc_method == "Ordinal Encoding":
                    oe = OrdinalEncoder()
                    df_enc[enc_sel] = oe.fit_transform(df_enc[enc_sel].astype(str))
                cols_after = df_enc.shape[1]
                st.session_state.df = df_enc
                st.success(f"✦  {enc_method} applied to {len(enc_sel)} column(s).")
                with st.expander("📋 Encoding Results", expanded=True):
                    col_enc1, col_enc2 = st.columns(2)
                    with col_enc1:
                        st.markdown("**Encoded Columns:**")
                        for col in enc_sel:
                            st.markdown(f"  • `{col}`")
                    with col_enc2:
                        st.markdown("**Transformation Summary:**")
                        st.markdown(f"  • Method: `{enc_method}`")
                        st.markdown(f"  • Columns Before: `{cols_before}`")
                        st.markdown(f"  • Columns After: `{cols_after}`")
                st.markdown("**Encoded Dataset Preview:**")
                st.dataframe(st.session_state.df.head(5), use_container_width=True)
        else:
            st.info("No categorical columns — ready for modelling.")

        section_hr()
        st.markdown("##### Feature Engineering")
        num_cols_fe = [c for c in st.session_state.df.select_dtypes(include=np.number).columns
                       if c != st.session_state.target_col]
        if num_cols_fe:
            fe1, fe2, fe3 = st.columns([2, 2, 1])
            with fe1: fe_col = st.selectbox("Source column:", num_cols_fe)
            with fe2: fe_transform = st.selectbox("Transform:", ["Log (log1p)", "Square", "Square Root", "Abs", "Interaction (col1 × col2)"])
            with fe3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add Feature"):
                    df_fe = st.session_state.df.copy()
                    try:
                        if fe_transform == "Log (log1p)":
                            df_fe[f"{fe_col}_log"] = np.log1p(df_fe[fe_col].clip(lower=0))
                        elif fe_transform == "Square":
                            df_fe[f"{fe_col}_sq"] = df_fe[fe_col] ** 2
                        elif fe_transform == "Square Root":
                            df_fe[f"{fe_col}_sqrt"] = np.sqrt(df_fe[fe_col].clip(lower=0))
                        elif fe_transform == "Abs":
                            df_fe[f"{fe_col}_abs"] = df_fe[fe_col].abs()
                        elif fe_transform == "Interaction (col1 × col2)":
                            if len(num_cols_fe) >= 2:
                                col2 = [c for c in num_cols_fe if c != fe_col][0]
                                df_fe[f"{fe_col}_x_{col2}"] = df_fe[fe_col] * df_fe[col2]
                        st.session_state.df = df_fe
                        st.success(f"✦  Feature `{fe_transform}` on `{fe_col}` added.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        section_hr()
        st.markdown("##### Feature Scaling — Normalize the Dimensions")
        num_cols_scale = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if num_cols_scale:
            scale_c1, scale_c2 = st.columns([2, 1.5])
            with scale_c1:
                scale_cols = st.multiselect("Columns to scale:", num_cols_scale, 
                                             default=num_cols_scale,
                                             help="Select numeric columns to normalize")
            with scale_c2:
                scale_method = st.selectbox("Scaler:", [
                    "StandardScaler",
                    "MinMaxScaler", 
                    "RobustScaler",
                    "MaxAbsScaler"
                ], help="• StandardScaler: Zero mean, unit variance\n• MinMaxScaler: Scale to [0,1]\n• RobustScaler: Robust to outliers\n• MaxAbsScaler: Scale to [-1,1]")
            
            if st.button("⬧  Apply Scaling"):
                if scale_cols:
                    df_scaled = st.session_state.df.copy()
                    try:
                        if scale_method == "StandardScaler":
                            scaler = StandardScaler()
                        elif scale_method == "MinMaxScaler":
                            scaler = MinMaxScaler()
                        elif scale_method == "RobustScaler":
                            scaler = RobustScaler()
                        else:  # MaxAbsScaler
                            scaler = MaxAbsScaler()
                        
                        # Fit and transform
                        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
                        st.session_state.df = df_scaled
                        
                        # Show results
                        st.success(f"✦  {scale_method} applied to {len(scale_cols)} column(s).")
                        
                        with st.expander("📊 Scaling Statistics", expanded=True):
                            stat_c1, stat_c2 = st.columns(2)
                            with stat_c1:
                                st.markdown("**Scaled Columns:**")
                                for col in scale_cols:
                                    st.markdown(f"  • `{col}`")
                            with stat_c2:
                                st.markdown("**Statistics:**")
                                st.markdown(f"  • Method: `{scale_method}`")
                                st.markdown(f"  • Min value: `{df_scaled[scale_cols].min().min():.4f}`")
                                st.markdown(f"  • Max value: `{df_scaled[scale_cols].max().max():.4f}`")
                                st.markdown(f"  • Mean: `{df_scaled[scale_cols].mean().mean():.4f}`")
                        
                        # Show before/after comparison for one column
                        if len(scale_cols) > 0:
                            st.markdown("**Scaled Distribution (First Column):**")
                            col_compare = scale_cols[0]
                            
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                st.markdown("**Distribution:**")
                                fig_scaled = px.histogram(df_scaled, x=col_compare, nbins=40,
                                                         color_discrete_sequence=[D_BLUE])
                                fig_scaled.update_layout(**dtheme(height=280, bargap=0.04))
                                st.plotly_chart(fig_scaled, use_container_width=True)
                            with col_v2:
                                st.markdown("**Scaled Statistics:**")
                                stats_text = f"""
                                **Column: `{col_compare}`**
                                • Min: `{df_scaled[col_compare].min():.4f}`
                                • Max: `{df_scaled[col_compare].max():.4f}`
                                • Mean: `{df_scaled[col_compare].mean():.4f}`
                                • Std: `{df_scaled[col_compare].std():.4f}`
                                • Median: `{df_scaled[col_compare].median():.4f}`
                                """
                                st.markdown(stats_text)
                        
                        st.markdown("**Scaled Dataset Preview:**")
                        st.dataframe(st.session_state.df[scale_cols].head(8), use_container_width=True)
                    except Exception as e:
                        st.error(f"Scaling error: {str(e)}")
                else:
                    st.warning("Select at least one column to scale.")
        else:
            st.info("No numeric columns to scale.")

        section_hr()
        st.markdown("##### Current Dataset")
        st.dataframe(st.session_state.df.head(6), use_container_width=True)
        st.markdown(f'<div class="cut-pill">{st.session_state.df.shape[0]:,} rows</div>'
                    f'<div class="cut-pill">{st.session_state.df.shape[1]} cols</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — FEATURE SELECTION
# ══════════════════════════════════════════════
with tabs[3]:
    step_badge(4, "Feature Selection — Grading the Attributes")
    if st.session_state.df is None or st.session_state.target_col is None:
        st.info("Complete Setup & Clean steps first.")
    else:
        df = st.session_state.df.dropna()
        target = st.session_state.target_col
        if target not in df.columns:
            st.error("Target column not found. Re-select in Setup.")
        else:
            X = df.drop(columns=[target])
            y = df[target]
            Xn = X.select_dtypes(include=np.number)
            if Xn.empty:
                st.error("No numeric features. Encode categorical columns in Clean step first.")
            else:
                method = st.selectbox("Selection Method:", [
                    "Variance Threshold", "Correlation with Target", "Mutual Information (Info Gain)"])
                section_hr()

                if method == "Variance Threshold":
                    thr = st.slider("Minimum Variance", 0.0, 2.0, 0.05, step=0.01)
                    if st.button("⬧  Apply Filter"):
                        sel = VarianceThreshold(threshold=thr)
                        sel.fit(Xn)
                        kept = Xn.columns[sel.get_support()].tolist()
                        var_df = pd.DataFrame({'Feature': Xn.columns, 'Variance': Xn.var().values})
                        fig = px.bar(var_df.sort_values('Variance', ascending=False),
                                     x='Feature', y='Variance', color='Variance', color_continuous_scale=CS_BLUE)
                        fig.add_hline(y=thr, line_dash='dash', line_color=D_FIRE, annotation_text='Threshold')
                        fig.update_layout(**dtheme(height=340))
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"✦  Retained {len(kept)} features: {', '.join(kept)}")
                        st.session_state.selected_features = kept

                elif method == "Correlation with Target":
                    if not pd.api.types.is_numeric_dtype(y):
                        st.warning("Target must be numeric for correlation.")
                    else:
                        cthr = st.slider("Min |Correlation|", 0.0, 1.0, 0.3)
                        if st.button("⬧  Filter by Correlation"):
                            corrs = pd.concat([Xn, y], axis=1).corr()[target].drop(target).abs()
                            kept = corrs[corrs >= cthr].index.tolist()
                            fig = px.bar(corrs.sort_values(ascending=False).reset_index(),
                                         x='index', y=target, color=target,
                                         color_continuous_scale=CS_TEAL,
                                         labels={'index': 'Feature', target: '|Correlation|'})
                            fig.add_hline(y=cthr, line_dash='dash', line_color=D_FIRE, annotation_text='Threshold')
                            fig.update_layout(**dtheme(height=340))
                            st.plotly_chart(fig, use_container_width=True)
                            st.success(f"✦  Retained {len(kept)} features: {', '.join(kept)}")
                            st.session_state.selected_features = kept

                elif method == "Mutual Information (Info Gain)":
                    k = st.slider("Top-K Features", 1, len(Xn.columns), min(8, len(Xn.columns)))
                    if st.button("⬧  Compute MI"):
                        with st.spinner("Calculating information gain…"):
                            mi = mutual_info_classif(Xn, y, random_state=42) if st.session_state.problem_type == "Classification" \
                                 else mutual_info_regression(Xn, y, random_state=42)
                            mi_s = pd.Series(mi, index=Xn.columns).sort_values(ascending=False)
                            fig = px.bar(mi_s.reset_index(), x='index', y=0, color=0,
                                         color_continuous_scale=CS_BLUE,
                                         labels={'index': 'Feature', 0: 'MI Score'})
                            fig.update_layout(**dtheme(height=340))
                            st.plotly_chart(fig, use_container_width=True)
                            kept = mi_s.head(k).index.tolist()
                            st.success(f"✦  Top-{k}: {', '.join(kept)}")
                            st.session_state.selected_features = kept

                if st.session_state.selected_features:
                    st.markdown(f'<div class="cut-pill gold">Selected: {len(st.session_state.selected_features)} facets</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 — SPLIT
# ══════════════════════════════════════════════
with tabs[4]:
    step_badge(5, "Train / Test Split — Partition the Gems")
    if st.session_state.df is None or st.session_state.target_col is None:
        st.info("Complete earlier steps first.")
    else:
        cs1, cs2, cs3 = st.columns([2, 2, 1], gap="large")
        with cs1: test_ratio = st.slider("Test Set (%)", 10, 40, 20) / 100
        with cs2: rand_state = st.number_input("Random Seed", value=42, step=1)
        with cs3: stratify_split = st.checkbox("Stratify", value=False)

        section_hr()
        if st.button("⬧  Execute Split", use_container_width=True):
            df_sp = st.session_state.df.dropna()
            if st.session_state.selected_features:
                avail = [f for f in st.session_state.selected_features if f in df_sp.columns]
                Xsp = df_sp[avail]
            else:
                Xsp = df_sp.drop(columns=[st.session_state.target_col]).select_dtypes(include=np.number)
            ysp = df_sp[st.session_state.target_col]
            strat = ysp if (stratify_split and st.session_state.problem_type == "Classification") else None
            try:
                Xtr, Xte, ytr, yte = train_test_split(Xsp, ysp, test_size=test_ratio,
                                                        random_state=int(rand_state), stratify=strat)
            except Exception as e:
                Xtr, Xte, ytr, yte = train_test_split(Xsp, ysp, test_size=test_ratio, random_state=int(rand_state))
                st.warning(f"Stratify skipped: {e}")

            st.session_state.X_train = Xtr; st.session_state.X_test = Xte
            st.session_state.y_train = ytr; st.session_state.y_test = yte
            if st.session_state.pipeline_stage < 5: st.session_state.pipeline_stage = 5

            ca, cb, cc = st.columns(3)
            ca.metric("Training Stones", f"{Xtr.shape[0]:,}")
            cb.metric("Test Stones", f"{Xte.shape[0]:,}")
            cc.metric("Features", f"{Xtr.shape[1]}")

            fig_split = go.Figure(go.Pie(
                labels=['Train', 'Test'], values=[Xtr.shape[0], Xte.shape[0]],
                hole=0.65, marker=dict(colors=[D_BLUE, D_FIRE]),
                textfont=dict(family='DM Mono, monospace', size=12)
            ))
            fig_split.update_layout(**dtheme(height=280, showlegend=True,
                                             margin=dict(t=20, b=20)))
            st.plotly_chart(fig_split, use_container_width=True)
            st.success("✦  Dataset partitioned. Ready for model comparison.")

# ══════════════════════════════════════════════
# TAB 6 — COMPARE
# ══════════════════════════════════════════════
with tabs[5]:
    step_badge(6, "Model Comparison — The Leaderboard")
    if st.session_state.X_train is None:
        st.info("Complete the Split step first.")
    else:
        model_pool = REGRESSION_MODELS if st.session_state.problem_type == "Regression" else CLASSIFICATION_MODELS
        section_hr()
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            cv_k_cmp = st.slider("CV Folds", 2, 10, 5, key="cmp_cv")
        with cc2:
            scale_cmp = st.checkbox("Scale features", value=True, key="cmp_scale")
        models_to_compare = st.multiselect("Models to compare:", list(model_pool.keys()),
                                            default=list(model_pool.keys()))

        if st.button("🏁  Run Comparison", use_container_width=True):
            Xtr = st.session_state.X_train.values
            ytr = st.session_state.y_train.values
            if scale_cmp:
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr)
            scorer = 'r2' if st.session_state.problem_type == "Regression" else 'accuracy'
            results = []
            progress = st.progress(0)
            status = st.empty()
            for i, mname in enumerate(models_to_compare):
                status.markdown(f'<div class="cut-pill">⬧ Evaluating: {mname}</div>', unsafe_allow_html=True)
                m = model_pool[mname]
                scores = cross_val_score(m, Xtr, ytr,
                                         cv=KFold(n_splits=cv_k_cmp, shuffle=True, random_state=42),
                                         scoring=scorer, n_jobs=-1)
                results.append({'Model': mname, 'CV Mean': scores.mean(), 'CV Std': scores.std(),
                                 'CV Min': scores.min(), 'CV Max': scores.max()})
                progress.progress((i + 1) / len(models_to_compare))
            status.empty(); progress.empty()

            res_df = pd.DataFrame(results).sort_values('CV Mean', ascending=False).reset_index(drop=True)
            st.session_state.model_comparison = res_df.to_dict()

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=res_df['Model'], y=res_df['CV Mean'],
                error_y=dict(type='data', array=res_df['CV Std'].values, visible=True, color=D_FIRE),
                marker=dict(color=res_df['CV Mean'].values, colorscale=CS_BLUE,
                            line=dict(color=D_BLUE, width=1))
            ))
            fig_cmp.update_layout(**dtheme(height=380, title='Cross-Validated Model Leaderboard',
                                           xaxis_title='Model', yaxis_title=f'CV {scorer.upper()}'))
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.markdown("##### Results")
            st.dataframe(res_df.style.format({'CV Mean': '{:.4f}', 'CV Std': '{:.4f}',
                                               'CV Min': '{:.4f}', 'CV Max': '{:.4f}'})
                         .background_gradient(cmap='Blues', subset=['CV Mean']),
                         use_container_width=True)
            best_name = res_df.iloc[0]['Model']
            st.success(f"🏆  Best: **{best_name}** — CV {scorer.upper()} = {res_df.iloc[0]['CV Mean']:.4f}")
            st.markdown(f'<div class="cut-pill gold">⬧ Recommended: {best_name}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 7 — TRAIN
# ══════════════════════════════════════════════
with tabs[6]:
    step_badge(7, "Model Training & Validation")
    if st.session_state.X_train is None:
        st.info("Complete the Split step first.")
    else:
        model_pool = REGRESSION_MODELS if st.session_state.problem_type == "Regression" else CLASSIFICATION_MODELS
        cm1, cm2, cm3 = st.columns([2, 2, 1], gap="large")
        with cm1: model_name = st.selectbox("Model:", list(model_pool.keys()))
        with cm2: k_folds = st.number_input("K-Fold Splits", min_value=2, max_value=20, value=5)
        with cm3: scale_data = st.checkbox("Scale", value=True)

        section_hr()
        if st.button("⬧  Launch Training", use_container_width=True):
            with st.spinner("Cutting facets…"):
                model = model_pool[model_name]
                Xtr = st.session_state.X_train.values
                Xte = st.session_state.X_test.values
                ytr = st.session_state.y_train.values
                yte = st.session_state.y_test.values

                sc = None
                if scale_data:
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(Xtr)
                    Xte = sc.transform(Xte)

                scorer = 'r2' if st.session_state.problem_type == "Regression" else 'accuracy'
                cv_scores = cross_val_score(model, Xtr, ytr,
                                            cv=KFold(n_splits=int(k_folds), shuffle=True, random_state=42),
                                            scoring=scorer, n_jobs=-1)

                st.markdown("##### Cross-Validation Scores")
                cv_df = pd.DataFrame({'Fold': range(1, int(k_folds) + 1), 'Score': cv_scores})
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Bar(x=cv_df['Fold'], y=cv_df['Score'],
                                        marker=dict(color=cv_scores, colorscale=CS_BLUE,
                                                     line=dict(color=D_BLUE, width=1))))
                fig_cv.add_hline(y=cv_scores.mean(), line_dash='dot', line_color=D_TEAL,
                                 annotation_text=f'Mean = {cv_scores.mean():.4f}',
                                 annotation_font_color=D_TEAL)
                fig_cv.update_layout(**dtheme(height=280, xaxis_title='Fold', yaxis_title=scorer.upper()))
                st.plotly_chart(fig_cv, use_container_width=True)

                col_a, col_b = st.columns(2)
                col_a.metric("CV Mean Score", f"{cv_scores.mean():.4f}")
                col_b.metric("CV Std Dev (±)", f"{cv_scores.std():.4f}")

                model.fit(Xtr, ytr)
                ytr_pred = model.predict(Xtr)
                yte_pred = model.predict(Xte)

                st.session_state.best_model = model
                st.session_state._scaler = sc
                st.session_state._Xte_scaled = Xte
                st.session_state._yte = yte
                st.session_state._Xtr_scaled = Xtr
                st.session_state._ytr = ytr
                if st.session_state.pipeline_stage < 7:
                    st.session_state.pipeline_stage = 7

                section_hr()
                st.markdown("##### Performance Diagnostics")

                if st.session_state.problem_type == "Regression":
                    tr_r2   = r2_score(ytr, ytr_pred)
                    te_r2   = r2_score(yte, yte_pred)
                    te_mae  = mean_absolute_error(yte, yte_pred)
                    te_rmse = rmse(yte, yte_pred)
                    te_mape = mape(yte, yte_pred)

                    d1, d2, d3, d4, d5 = st.columns(5)
                    d1.metric("Train R²",  f"{tr_r2:.4f}")
                    d2.metric("Test R²",   f"{te_r2:.4f}")
                    d3.metric("MAE",       f"${te_mae:,.0f}")
                    d4.metric("RMSE",      f"${te_rmse:,.0f}")
                    d5.metric("MAPE (%)",  f"{te_mape:.2f}%")

                    fig_sc = px.scatter(x=yte, y=yte_pred, opacity=.45,
                                        labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
                                        color_discrete_sequence=[D_BLUE])
                    mn = float(min(yte.min(), yte_pred.min()))
                    mx = float(max(yte.max(), yte_pred.max()))
                    fig_sc.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                                     line=dict(dash='dash', color=D_GOLD, width=2))
                    fig_sc.update_layout(**dtheme(height=380, title='Actual vs Predicted Price'))
                    fig_sc.update_traces(marker=dict(size=3))
                    st.plotly_chart(fig_sc, use_container_width=True)

                    residuals = yte - yte_pred
                    fig_res = px.histogram(x=residuals, nbins=60, color_discrete_sequence=[D_TEAL])
                    fig_res.update_layout(**dtheme(height=260, title='Residual Distribution', xaxis_title='Residual ($)'))
                    st.plotly_chart(fig_res, use_container_width=True)

                    delta = tr_r2 - te_r2
                    if delta > 0.15: st.error("⚠  Overfitting — Train R² significantly exceeds Test R².")
                    elif te_r2 < 0.5: st.warning("⚠  Underfitting — consider a more expressive model.")
                    else: st.success("✦  Model generalises well.")

                else:
                    tr_acc = accuracy_score(ytr, ytr_pred)
                    te_acc = accuracy_score(yte, yte_pred)
                    n_classes = len(np.unique(yte))
                    roc_auc = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            if n_classes == 2:
                                proba = model.predict_proba(Xte)[:, 1]
                                roc_auc = roc_auc_score(yte, proba)
                            else:
                                proba = model.predict_proba(Xte)
                                roc_auc = roc_auc_score(yte, proba, multi_class='ovr', average='macro')
                        except Exception: pass

                    d1, d2, d3 = st.columns(3)
                    d1.metric("Train Accuracy", f"{tr_acc:.4f}")
                    d2.metric("Test Accuracy",  f"{te_acc:.4f}")
                    if roc_auc: d3.metric("ROC-AUC", f"{roc_auc:.4f}")
                    else:       d3.metric("Acc Gap", f"{tr_acc - te_acc:.4f}")

                    cm_mat = confusion_matrix(yte, yte_pred)
                    fig_cm = px.imshow(cm_mat, text_auto=True, color_continuous_scale=CS_BLUE)
                    fig_cm.update_layout(**dtheme(height=380, title='Confusion Matrix'))
                    st.plotly_chart(fig_cm, use_container_width=True)

                    if n_classes == 2 and hasattr(model, 'predict_proba'):
                        try:
                            fpr, tpr, _ = roc_curve(yte, model.predict_proba(Xte)[:, 1])
                            fig_roc = go.Figure()
                            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                          line=dict(color=D_BLUE, width=2),
                                                          name=f'AUC={roc_auc:.3f}'))
                            fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                                              line=dict(dash='dash', color=D_FIRE))
                            fig_roc.update_layout(**dtheme(height=320, title='ROC Curve'))
                            st.plotly_chart(fig_roc, use_container_width=True)
                        except Exception: pass

                    rpt = classification_report(yte, yte_pred, output_dict=True)
                    rpt_df = pd.DataFrame(rpt).T.iloc[:-3]
                    st.dataframe(rpt_df.style.format(precision=3)
                                 .background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                                 use_container_width=True)

                    if tr_acc - te_acc > 0.15: st.error("⚠  Overfitting detected.")
                    elif tr_acc < 0.6 and te_acc < 0.6: st.warning("⚠  Underfitting.")
                    else: st.success("✦  Model generalises well.")

                # Feature importances
                if hasattr(model, 'feature_importances_'):
                    section_hr()
                    st.markdown("##### Feature Importances — Which Cs Matter Most?")
                    feat_names = st.session_state.X_train.columns.tolist()
                    fi = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
                    fig_fi = px.bar(fi.reset_index(), x='index', y=0, color=0,
                                    color_continuous_scale=CS_BLUE,
                                    labels={'index': 'Feature', 0: 'Importance'})
                    fig_fi.update_layout(**dtheme(height=300))
                    st.plotly_chart(fig_fi, use_container_width=True)

                section_hr()
                st.markdown("##### Export Model")
                export_obj = {'model': model, 'scaler': sc,
                               'features': st.session_state.X_train.columns.tolist(),
                               'problem_type': st.session_state.problem_type}
                st.markdown(get_download_link(export_obj, f"{model_name.replace(' ','_')}_diamond_model.pkl",
                                               "Download Model (.pkl)"), unsafe_allow_html=True)
                st.caption("Load: `import pickle; obj = pickle.load(open('diamond_model.pkl','rb'))`")

# ══════════════════════════════════════════════
# TAB 8 — TUNE
# ══════════════════════════════════════════════
with tabs[7]:
    step_badge(8, "Hyperparameter Tuning — The Final Polish")
    if st.session_state.best_model is None:
        st.info("Train a model in Step 7 first.")
    else:
        model = st.session_state.best_model
        mtype = type(model).__name__
        st.markdown(f'<div class="cut-pill gold">⬧ Active engine: {mtype}</div>', unsafe_allow_html=True)

        if mtype not in GRIDS or not GRIDS[mtype]:
            st.warning(f"`{mtype}` has no tunable hyperparameters configured.")
        else:
            section_hr()
            st.markdown("##### Parameter Search Space")
            st.json(GRIDS[mtype])
            section_hr()

            ct1, ct2 = st.columns(2, gap="large")
            with ct1:
                search_method = st.radio("Search Strategy:", [
                    "Grid Search (Exhaustive)", "Random Search (Stochastic)"])
            with ct2:
                cv_k = st.number_input("CV Folds", min_value=2, max_value=10, value=3)
                if "Random" in search_method:
                    n_iter = st.number_input("Iterations", min_value=5, max_value=200, value=30)

            if st.button("⬧  Start Tuning", use_container_width=True):
                with st.spinner("Polishing facets…"):
                    Xtr_t = st.session_state._Xtr_scaled
                    ytr_t = st.session_state._ytr
                    scorer = 'r2' if st.session_state.problem_type == "Regression" else 'accuracy'
                    if "Grid" in search_method:
                        searcher = GridSearchCV(model, GRIDS[mtype], cv=int(cv_k),
                                                scoring=scorer, n_jobs=-1)
                    else:
                        searcher = RandomizedSearchCV(model, GRIDS[mtype], cv=int(cv_k),
                                                      n_iter=int(n_iter), scoring=scorer,
                                                      random_state=42, n_jobs=-1)
                    searcher.fit(Xtr_t, ytr_t)

                section_hr()
                st.markdown("##### Tuning Results")
                d1, d2 = st.columns(2)
                d1.metric("Best CV Score", f"{searcher.best_score_:.4f}")
                d2.metric("Combinations Tested", f"{len(searcher.cv_results_['params'])}")
                st.markdown("**Optimal Parameters:**")
                st.json(searcher.best_params_)

                cv_res = pd.DataFrame(searcher.cv_results_)
                top_n = min(20, len(cv_res))
                cv_res_top = cv_res.nlargest(top_n, 'mean_test_score').reset_index(drop=True)
                cv_res_top['combo'] = cv_res_top.index.astype(str)

                fig_tune = go.Figure()
                fig_tune.add_trace(go.Bar(
                    x=cv_res_top['combo'], y=cv_res_top['mean_test_score'],
                    error_y=dict(type='data', array=cv_res_top['std_test_score'].values,
                                 visible=True, color=D_FIRE),
                    marker=dict(color=cv_res_top['mean_test_score'].values,
                                colorscale=CS_BLUE, line=dict(color=D_BLUE, width=1))
                ))
                fig_tune.add_hline(y=searcher.best_score_, line_dash='dot', line_color=D_TEAL,
                                   annotation_text=f'Best = {searcher.best_score_:.4f}',
                                   annotation_font_color=D_TEAL)
                fig_tune.update_layout(**dtheme(height=320, xaxis_title='Rank', yaxis_title='CV Score'))
                st.plotly_chart(fig_tune, use_container_width=True)

                best_est = searcher.best_estimator_
                yte_tuned = best_est.predict(st.session_state._Xte_scaled)
                yte_true  = st.session_state._yte
                st.session_state.tuned_model = best_est

                section_hr()
                st.markdown("##### Tuned Model — Test Set")
                if st.session_state.problem_type == "Regression":
                    te_r2   = r2_score(yte_true, yte_tuned)
                    te_rmse = rmse(yte_true, yte_tuned)
                    te_mae  = mean_absolute_error(yte_true, yte_tuned)
                    ta, tb, tc = st.columns(3)
                    ta.metric("Tuned R²",   f"{te_r2:.4f}")
                    tb.metric("Tuned RMSE", f"${te_rmse:,.0f}")
                    tc.metric("Tuned MAE",  f"${te_mae:,.0f}")
                else:
                    te_acc = accuracy_score(yte_true, yte_tuned)
                    ta, tb = st.columns(2)
                    ta.metric("Tuned Accuracy", f"{te_acc:.4f}")
                    tb.metric("Strategy", search_method.split()[0])

                st.success("✦  Hyperparameter tuning complete. Your model is perfectly cut.")

                section_hr()
                st.markdown("##### Export Tuned Model")
                export_tuned = {'model': best_est, 'scaler': st.session_state._scaler,
                                'features': st.session_state.X_train.columns.tolist(),
                                'best_params': searcher.best_params_,
                                'problem_type': st.session_state.problem_type}
                st.markdown(get_download_link(export_tuned, f"tuned_{mtype}_diamond.pkl",
                                               "Download Tuned Model (.pkl)"), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#1e2440; font-family: DM Mono, monospace; font-size:11px; letter-spacing:.1em;'>
  💎 &nbsp; AUTOML STUDIO &nbsp;·&nbsp; Built with Streamlit, scikit-learn & Plotly &nbsp;·&nbsp; Exhibition 2025
</p>
""", unsafe_allow_html=True)