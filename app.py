import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Urban Traffic Bottleneck Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #08111f; color: #e8eef8; }
[data-testid="stSidebar"] { background: #0d192a; border-right: 1px solid #1c3048; }
.hero {
  padding: 28px 30px; border: 1px solid #24415c; border-radius: 20px;
  background: radial-gradient(circle at 85% 15%, #163c53 0, transparent 34%),
              linear-gradient(120deg, #10263b, #0c1728 70%);
  margin-bottom: 22px;
}
.eyebrow { color: #42d6bd; font-size: 12px; font-weight: 700; letter-spacing: 2px; }
.hero h1 { font-family: 'Space Grotesk'; color: #f5f8ff; font-size: 34px; margin: 8px 0; }
.hero p { color: #a9bad0; max-width: 780px; margin-bottom: 0; }
.metric {
  background: #101f32; border: 1px solid #203a55; border-radius: 15px;
  padding: 17px 18px; min-height: 105px;
}
.metric-label { color: #9db0c8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #f5f8ff; font-family: 'Space Grotesk'; font-size: 27px; font-weight: 700; margin-top: 7px; }
.section-title { font-family: 'Space Grotesk'; font-size: 22px; color: #f5f8ff; margin: 10px 0 4px; }
.small-muted { color: #9db0c8; font-size: 13px; }
div.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #17bda9, #2d9fce); color: #06121d;
  border: 0; border-radius: 10px; font-weight: 700; min-height: 44px;
}
.stButton > button { border-radius: 10px; }
div[data-testid="stAlert"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
REQUIRED_DEFAULTS = ["vehicle_count", "average_speed", "hour", "day_of_week", "road_id", "severity"]

def metric_card(label, value):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

def build_pipeline(model_name, numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", categorical_pipe, categorical_features))
    preprocess = ColumnTransformer(transformers)
    if model_name == "KNN":
        model = KNeighborsClassifier()
        params = {"model__n_neighbors": [3, 5, 7]}
    else:
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        params = {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]}
    return Pipeline([("prep", preprocess), ("model", model)]), params

def safe_cv(y):
    smallest = int(y.value_counts().min())
    return min(5, smallest)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">SMART MOBILITY · MACHINE LEARNING</div>
  <h1>Urban Traffic Bottleneck Prediction</h1>
  <p>Analyze historical road-traffic patterns, compare KNN and Random Forest, and estimate congestion severity across four levels: Low, Moderate, High and Severe.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🚦 Control panel")
    st.caption("Project: SIH26124")
    page = st.radio("Navigate", ["Overview", "Train & Evaluate", "Predict Severity", "About the Project"])
    st.divider()
    st.markdown("**Project team**")
    st.write("Avinash Kumar · 12518763")
    st.write("Mohan Kumar · 12522200")
    st.caption("Models are trained on the dataset you upload. No live traffic feed is connected.")

if "models" not in st.session_state:
    st.session_state.models = {}
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []
if "data" not in st.session_state:
    st.session_state.data = None
if "target" not in st.session_state:
    st.session_state.target = "severity"

# ---------- Overview ----------
if page == "Overview":
    st.markdown('<div class="section-title">Traffic intelligence at a glance</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">A machine-learning workflow for early identification of congested road segments.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Prediction classes", "04")
    with c2: metric_card("ML algorithms", "02")
    with c3: metric_card("Tuning method", "GridSearchCV")
    with c4: metric_card("Evaluation", "5 metrics")
    st.write("")
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("### Why predict bottlenecks?")
        st.write("Urban congestion causes travel delays, fuel wastage and air pollution. Early severity prediction can help traffic authorities identify problematic road segments and make better-informed traffic-management decisions.")
        st.markdown("### Project workflow")
        steps = [
            ("01", "Collect traffic data", "Upload historical traffic records in CSV format."),
            ("02", "Clean & prepare", "Remove duplicates, impute missing values and scale numeric inputs."),
            ("03", "Train models", "Train KNN and Random Forest classifiers."),
            ("04", "Tune & compare", "Use GridSearchCV and evaluate model performance."),
            ("05", "Predict severity", "Estimate Low, Moderate, High or Severe congestion.")
        ]
        for n, title, desc in steps:
            st.markdown(f"**{n}  {title}**  \n<span class='small-muted'>{desc}</span>", unsafe_allow_html=True)
    with right:
        st.markdown("### Severity levels")
        levels = [("🟢", "Low", "Traffic is flowing freely"), ("🟡", "Moderate", "Some delay or slower movement"), ("🟠", "High", "Heavy traffic and notable delays"), ("🔴", "Severe", "Very heavy congestion")]
        for icon, name, desc in levels:
            st.markdown(f"**{icon} {name}**  \n<span class='small-muted'>{desc}</span>", unsafe_allow_html=True)
        st.info("Start at **Train & Evaluate** and upload a labeled CSV dataset to train the models.")

# ---------- Train ----------
elif page == "Train & Evaluate":
    st.markdown('<div class="section-title">Train and evaluate models</div>', unsafe_allow_html=True)
    st.write("Upload a labeled CSV file. The default schema follows the project appendix, but you can select your own target and feature columns.")
    uploaded = st.file_uploader("Upload traffic dataset (.csv)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.data = df
            st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
            with st.expander("Preview dataset"):
                st.dataframe(df.head(20), use_container_width=True)
                st.write("Columns:", ", ".join(map(str, df.columns)))
            target = st.selectbox("Select target column (congestion severity label)", df.columns.tolist(),
                                  index=df.columns.get_loc("severity") if "severity" in df.columns else len(df.columns)-1)
            feature_options = [c for c in df.columns if c != target]
            defaults = [c for c in ["vehicle_count", "average_speed", "hour", "day_of_week", "road_id"] if c in feature_options]
            features = st.multiselect("Select input features", feature_options, default=defaults or feature_options)
            if st.button("🚀 Train KNN and Random Forest", type="primary", use_container_width=True):
                if not features:
                    st.error("Select at least one feature column.")
                else:
                    work = df[features + [target]].dropna(subset=[target]).drop_duplicates()
                    X, y = work[features], work[target].astype(str)
                    if y.nunique() < 2:
                        st.error("The target column must contain at least two classes.")
                    elif len(work) < 10:
                        st.error("Please upload at least 10 labeled rows for a basic train/test split.")
                    else:
                        counts = y.value_counts()
                        stratify = y if counts.min() >= 2 else None
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=stratify
                        )
                        numeric = X.select_dtypes(include=np.number).columns.tolist()
                        categorical = [c for c in features if c not in numeric]
                        cv = safe_cv(y_train)
                        if cv < 2:
                            st.error("Not enough examples per class to run cross-validation. Add more examples for each severity class.")
                        else:
                            results, fitted = {}, {}
                            progress = st.progress(0, text="Preparing models...")
                            for idx, name in enumerate(["KNN", "Random Forest"]):
                                pipe, params = build_pipeline(name, numeric, categorical)
                                if name == "KNN":
                                    max_k = max(1, len(X_train))
                                    params["model__n_neighbors"] = [k for k in [3, 5, 7] if k <= max_k] or [1]
                                search = GridSearchCV(pipe, params, cv=cv, scoring="f1_weighted", n_jobs=-1)
                                search.fit(X_train, y_train)
                                pred = search.predict(X_test)
                                results[name] = {
                                    "Accuracy": accuracy_score(y_test, pred),
                                    "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                                    "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                                    "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0),
                                    "Best parameters": str(search.best_params_),
                                    "Report": classification_report(y_test, pred, zero_division=0)
                                }
                                fitted[name] = search.best_estimator_
                                progress.progress((idx+1)/2, text=f"Trained {name}...")
                            st.session_state.models = fitted
                            st.session_state.metrics = results
                            st.session_state.feature_cols = features
                            st.session_state.target = target
                            st.session_state.numeric_features = numeric
                            st.session_state.categorical_features = categorical
                            st.session_state.class_labels = sorted(y.unique().tolist())
                            st.success("Both models trained successfully.")
            if st.session_state.metrics:
                st.markdown("### Model comparison")
                metric_rows = []
                for name, vals in st.session_state.metrics.items():
                    metric_rows.append({"Model": name, **{m: round(vals[m], 4) for m in ["Accuracy", "Precision", "Recall", "F1-score"]}})
                st.dataframe(pd.DataFrame(metric_rows).set_index("Model"), use_container_width=True)
                best = max(st.session_state.metrics, key=lambda m: st.session_state.metrics[m]["F1-score"])
                st.success(f"Best model by weighted F1-score: {best}")
                for name, vals in st.session_state.metrics.items():
                    with st.expander(f"{name} — details"):
                        st.write("Best parameters:", vals["Best parameters"])
                        st.code(vals["Report"])
                model_bytes = {name: joblib.dumps(model) if hasattr(joblib, "dumps") else None for name, model in st.session_state.models.items()}
                st.caption("Models are held in this Streamlit session. Train again after refreshing or restarting the app.")
        except Exception as e:
            st.error(f"Could not process this CSV: {e}")
    else:
        st.info("Expected example columns: vehicle_count, average_speed, hour, day_of_week, road_id, severity. The exact dataset schema is not included in the project PDF; choose the matching columns from your actual CSV.")

# ---------- Predict ----------
elif page == "Predict Severity":
    st.markdown('<div class="section-title">Predict congestion severity</div>', unsafe_allow_html=True)
    if not st.session_state.models:
        st.warning("No trained model found in this session. Go to **Train & Evaluate** and train the models first.")
    else:
        model_name = st.selectbox("Choose prediction model", list(st.session_state.models.keys()))
        st.write("Enter the road and traffic conditions:")
        values = {}
        cols = st.columns(2)
        for i, feature in enumerate(st.session_state.feature_cols):
            with cols[i % 2]:
                if feature in st.session_state.numeric_features:
                    series = st.session_state.data[feature]
                    med = float(pd.to_numeric(series, errors="coerce").median()) if pd.to_numeric(series, errors="coerce").notna().any() else 0.0
                    if feature == "hour":
                        values[feature] = st.slider("Hour of day", 0, 23, int(np.clip(med, 0, 23)))
                    elif feature == "day_of_week":
                        values[feature] = st.selectbox("Day of week (0=Mon ... 6=Sun)", list(range(7)), index=int(np.clip(med, 0, 6)))
                    elif feature == "vehicle_count":
                        values[feature] = st.number_input("Vehicle count", min_value=0, value=max(0, int(med)))
                    elif feature == "average_speed":
                        values[feature] = st.number_input("Average speed (km/h)", min_value=0.0, value=max(0.0, med))
                    else:
                        values[feature] = st.number_input(feature.replace("_", " ").title(), value=med)
                else:
                    opts = sorted(st.session_state.data[feature].dropna().astype(str).unique().tolist())
                    values[feature] = st.selectbox(feature.replace("_", " ").title(), opts) if opts else st.text_input(feature)
        if st.button("🔍 Predict severity", type="primary", use_container_width=True):
            sample = pd.DataFrame([values], columns=st.session_state.feature_cols)
            prediction = st.session_state.models[model_name].predict(sample)[0]
            color = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Severe": "🔴"}.get(str(prediction), "🚦")
            st.markdown(f"## {color} Predicted congestion: **{prediction}**")
            st.caption(f"Prediction generated by the {model_name} model trained on your uploaded dataset.")
            if hasattr(st.session_state.models[model_name], "predict_proba"):
                try:
                    probabilities = st.session_state.models[model_name].predict_proba(sample)[0]
                    classes = st.session_state.models[model_name].classes_
                    chart = pd.DataFrame({"Severity": classes, "Probability": probabilities}).set_index("Severity")
                    st.bar_chart(chart)
                except Exception:
                    pass

# ---------- About ----------
elif page == "About the Project":
    st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
    st.write("The Urban Traffic Bottleneck Prediction and Severity Analysis System uses historical traffic records to classify congestion severity into Low, Moderate, High and Severe.")
    st.markdown("### Technology stack")
    tech = st.columns(4)
    for col, title, desc in zip(tech, ["Python", "Pandas", "Scikit-learn", "Streamlit"], ["Core programming", "Data cleaning & analysis", "KNN, Random Forest & tuning", "Interactive web dashboard"]):
        with col:
            st.markdown(f'<div class="metric"><div class="metric-value" style="font-size:20px">{title}</div><div class="small-muted">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("### Methodology")
    st.write("Data collection → Data cleaning → Feature engineering → Severity classification → KNN and Random Forest training → GridSearchCV → Model evaluation → Prediction dashboard.")
    st.markdown("### Limitations")
    st.write("- Predictions depend on the quality and coverage of the uploaded historical dataset.")
    st.write("- This app does not receive real-time traffic or GPS data.")
    st.write("- Unexpected incidents may not be represented in historical training records.")
    st.markdown("### Future scope")
    st.write("- Integrate live traffic data from GPS, cameras and IoT sensors.")
    st.write("- Add road-segment maps and weather or accident information.")
    st.write("- Explore deep-learning models and smart traffic-signal control.")
    st.caption("Project team: Avinash Kumar (12518763) · Mohan Kumar (12522200)")
