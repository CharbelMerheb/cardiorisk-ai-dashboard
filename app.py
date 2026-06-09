from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "models" / "artifacts"


@dataclass(frozen=True)
class PatientProfile:
    sex: str
    age: int
    bmi: float
    smoking_status: str
    cigarettes_per_day: int
    systolic_bp: int
    diastolic_bp: int
    resting_heart_rate: int
    cholesterol: int
    glucose: int
    bp_meds: int
    prevalent_stroke: int
    prevalent_hypertension: int
    diabetes: int
    chest_pain_type: int
    resting_ecg: int
    max_heart_rate: int
    exercise_angina: int
    oldpeak: float
    st_slope: int
    residence_urban: int
    work_type: str


RISK_CONFIG = {
    "chd": {
        "title": "Coronary heart disease",
        "subtitle": "10-year CHD probability",
        "threshold": 0.58,
        "accent": "#c2410c",
    },
    "heart": {
        "title": "Heart disease",
        "subtitle": "Heart disease classifier output",
        "threshold": 0.50,
        "accent": "#be123c",
    },
    "stroke": {
        "title": "Stroke",
        "subtitle": "Stroke occurrence probability",
        "threshold": 0.58,
        "accent": "#0f766e",
    },
}


def page_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #14213d;
            --muted: #5f6c7b;
            --line: #d8dee9;
            --panel: #ffffff;
            --wash: #f6f8fb;
            --navy: #152238;
            --teal: #0f766e;
            --rose: #be123c;
            --amber: #b45309;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(246, 248, 251, 0.96), rgba(255, 255, 255, 0.98)),
                repeating-linear-gradient(90deg, rgba(20, 33, 61, 0.04) 0, rgba(20, 33, 61, 0.04) 1px, transparent 1px, transparent 88px);
            color: var(--ink);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label {
            color: #e5e7eb;
        }

        .hero {
            padding: 1.35rem 0 0.9rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.2rem;
        }

        .eyebrow {
            color: var(--teal);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero h1 {
            font-size: clamp(2.15rem, 5vw, 4.9rem);
            line-height: 0.96;
            margin: 0;
            color: var(--navy);
            letter-spacing: 0;
        }

        .hero p {
            max-width: 760px;
            color: var(--muted);
            font-size: 1.06rem;
            margin-top: 0.85rem;
            margin-bottom: 0;
        }

        .stat-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1.2rem 0 1.4rem;
        }

        .stat {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            min-height: 92px;
        }

        .stat span {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .stat strong {
            display: block;
            color: var(--navy);
            font-size: 1.55rem;
            margin-top: 0.25rem;
        }

        .section-label {
            color: var(--navy);
            font-size: 1.2rem;
            font-weight: 800;
            margin: 1.2rem 0 0.55rem;
        }

        .risk-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
        }

        .risk-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 12px 32px rgba(20, 33, 61, 0.08);
        }

        .risk-card h3 {
            color: var(--navy);
            font-size: 1.05rem;
            margin: 0;
        }

        .risk-card p {
            color: var(--muted);
            font-size: 0.88rem;
            margin: 0.2rem 0 0.8rem;
        }

        .score-row {
            align-items: end;
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .score {
            color: var(--navy);
            font-size: 2.65rem;
            font-weight: 850;
            line-height: 1;
        }

        .badge {
            border-radius: 999px;
            color: #fff;
            display: inline-flex;
            font-size: 0.75rem;
            font-weight: 800;
            padding: 0.28rem 0.6rem;
            white-space: nowrap;
        }

        .bar {
            background: #edf2f7;
            border-radius: 999px;
            height: 0.72rem;
            overflow: hidden;
        }

        .fill {
            border-radius: inherit;
            height: 100%;
        }

        .insight {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.85rem;
            padding-top: 0.75rem;
        }

        .notice {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            color: #7c2d12;
            padding: 0.85rem 1rem;
            margin: 1rem 0;
        }

        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 800;
            min-height: 3rem;
        }

        @media (max-width: 820px) {
            .stat-strip,
            .risk-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict[str, object]:
    required = [
        "chd_ml.joblib",
        "chd_nn.joblib",
        "heart_ml.joblib",
        "heart_nn.joblib",
        "stroke_ml.joblib",
        "stroke_nn.joblib",
    ]
    missing = [name for name in required if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Run `python scripts/train_models.py` first."
        )

    return {
        "Classical ML": {
            "chd": joblib.load(ARTIFACT_DIR / "chd_ml.joblib"),
            "heart": joblib.load(ARTIFACT_DIR / "heart_ml.joblib"),
            "stroke": joblib.load(ARTIFACT_DIR / "stroke_ml.joblib"),
        },
        "Neural Network": {
            "chd": joblib.load(ARTIFACT_DIR / "chd_nn.joblib"),
            "heart": joblib.load(ARTIFACT_DIR / "heart_nn.joblib"),
            "stroke": joblib.load(ARTIFACT_DIR / "stroke_nn.joblib"),
        },
    }


def yes_no(value: bool) -> int:
    return 1 if value else 0


def build_inputs(profile: PatientProfile) -> dict[str, pd.DataFrame]:
    is_male = 1 if profile.sex == "Male" else 0
    current_smoker = 1 if profile.smoking_status == "Current smoker" else 0
    former_smoker = 1 if profile.smoking_status == "Former smoker" else 0
    never_smoked = 1 if profile.smoking_status == "Never smoked" else 0
    ever_married = 1 if profile.age >= 25 else 0

    work_flags = {
        "Never worked": (1.0, 0.0, 0.0, 0.0),
        "Private": (0.0, 1.0, 0.0, 0.0),
        "Self-employed": (0.0, 0.0, 1.0, 0.0),
        "Children": (0.0, 0.0, 0.0, 1.0),
    }
    work_never, work_private, work_self, work_children = work_flags[profile.work_type]

    age_adult = int(18 <= profile.age < 50)
    age_middle = int(50 <= profile.age < 65)
    age_elderly = int(profile.age >= 65)
    bmi_normal = int(profile.bmi < 25)
    bmi_overweight = int(25 <= profile.bmi < 30)
    bmi_obese = int(profile.bmi >= 30)
    risk_score = (profile.age * 0.1) + (profile.glucose * 0.03) + (profile.bmi * 0.2) + (1.5 if profile.prevalent_hypertension else 0)

    chd = pd.DataFrame(
        [
            {
                "male": is_male,
                "age": profile.age,
                "education": 2.0,
                "currentSmoker": current_smoker,
                "cigsPerDay": profile.cigarettes_per_day,
                "BPMeds": float(profile.bp_meds),
                "prevalentStroke": profile.prevalent_stroke,
                "prevalentHyp": profile.prevalent_hypertension,
                "diabetes": profile.diabetes,
                "totChol": profile.cholesterol,
                "sysBP": profile.systolic_bp,
                "diaBP": profile.diastolic_bp,
                "BMI": profile.bmi,
                "heartRate": profile.resting_heart_rate,
                "glucose": profile.glucose,
            }
        ]
    )

    heart = pd.DataFrame(
        [
            {
                "Age": profile.age,
                "Sex": is_male,
                "ChestPainType": profile.chest_pain_type,
                "RestingBP": profile.systolic_bp,
                "Cholesterol": profile.cholesterol,
                "FastingBS": profile.diabetes,
                "RestingECG": profile.resting_ecg,
                "MaxHR": profile.max_heart_rate,
                "ExerciseAngina": profile.exercise_angina,
                "Oldpeak": profile.oldpeak,
                "ST_Slope": profile.st_slope,
            }
        ]
    )

    stroke = pd.DataFrame(
        [
            {
                "gender": float(is_male),
                "age": float(profile.age),
                "hypertension": profile.prevalent_hypertension,
                "heart_disease": profile.prevalent_stroke,
                "ever_married": ever_married,
                "avg_glucose_level": profile.glucose,
                "bmi": profile.bmi,
                "work_type_Never_worked": work_never,
                "work_type_Private": work_private,
                "work_type_Self-employed": work_self,
                "work_type_children": work_children,
                "Residence_type_Urban": float(profile.residence_urban),
                "smoking_status_formerly smoked": float(former_smoker),
                "smoking_status_never smoked": float(never_smoked),
                "smoking_status_smokes": float(current_smoker),
                "risk_score": risk_score,
                "age_group_Adult": age_adult,
                "age_group_Middle_Aged": age_middle,
                "age_group_Elderly": age_elderly,
                "bmi_category_Normal": bmi_normal,
                "bmi_category_Overweight": bmi_overweight,
                "bmi_category_Obese": bmi_obese,
            }
        ]
    )

    return {"chd": chd, "heart": heart, "stroke": stroke}


def predict_probability(model: object, data: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(data)[0][1])
    return float(model.predict(data)[0])


def risk_label(score: float, threshold: float) -> tuple[str, str]:
    if score >= threshold:
        return "Elevated", "#be123c"
    if score >= threshold * 0.65:
        return "Watch", "#b45309"
    return "Controlled", "#0f766e"


def render_risk_card(key: str, score: float) -> None:
    config = RISK_CONFIG[key]
    label, color = risk_label(score, config["threshold"])
    percent = max(0, min(100, score * 100))
    st.markdown(
        f"""
        <div class="risk-card">
            <h3>{config["title"]}</h3>
            <p>{config["subtitle"]}</p>
            <div class="score-row">
                <div class="score">{percent:.1f}%</div>
                <div class="badge" style="background:{color};">{label}</div>
            </div>
            <div class="bar">
                <div class="fill" style="width:{percent:.1f}%; background:{config["accent"]};"></div>
            </div>
            <div class="insight">Decision threshold: {config["threshold"] * 100:.0f}%. This output is a machine-learning screening estimate, not a clinical diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_engine() -> str:
    st.sidebar.markdown("### Model engine")
    return st.sidebar.radio(
        "Choose prediction engine",
        ["Classical ML", "Neural Network"],
        index=0,
        help="Classical ML uses logistic/tree-based baselines. Neural Network uses MLP classifiers trained from the processed datasets.",
    )


def collect_profile() -> PatientProfile:
    st.markdown('<div class="section-label">Patient profile</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.segmented_control("Sex", ["Male", "Female"], default="Male")
        age = st.slider("Age", 1, 100, 45)
        bmi = st.number_input("Body mass index", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
        smoking_status = st.selectbox("Smoking status", ["Never smoked", "Former smoker", "Current smoker"])
        cigarettes = st.slider("Cigarettes per day", 0, 100, 0, disabled=smoking_status != "Current smoker")

    with col2:
        systolic = st.slider("Systolic blood pressure", 80, 220, 120)
        diastolic = st.slider("Diastolic blood pressure", 50, 130, 80)
        resting_hr = st.slider("Resting heart rate", 40, 160, 72)
        max_hr = st.slider("Maximum heart rate", 60, 220, 150)
        cholesterol = st.number_input("Total cholesterol", min_value=80, max_value=650, value=200, step=1)
        glucose = st.number_input("Average glucose", min_value=50, max_value=400, value=90, step=1)

    with col3:
        bp_meds = yes_no(st.toggle("Blood pressure medication"))
        prevalent_stroke = yes_no(st.toggle("Prior stroke history"))
        prevalent_hyp = yes_no(st.toggle("Hypertension history"))
        diabetes = yes_no(st.toggle("Diabetes or fasting blood sugar"))
        chest_pain = st.selectbox("Chest pain type", [0, 1, 2, 3], format_func=lambda v: ["Asymptomatic", "Atypical angina", "Non-anginal", "Typical angina"][v])
        ecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda v: ["Normal", "ST-T abnormality", "Hypertrophy"][v])
        st_slope = st.selectbox("ST slope", [0, 1, 2], format_func=lambda v: ["Down", "Flat", "Up"][v])
        exercise_angina = yes_no(st.toggle("Exercise-induced angina"))
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=7.0, value=0.0, step=0.1)
        residence = yes_no(st.toggle("Urban residence", value=True))
        work_type = st.selectbox("Work type", ["Private", "Self-employed", "Never worked", "Children"])

    return PatientProfile(
        sex=sex or "Male",
        age=age,
        bmi=bmi,
        smoking_status=smoking_status,
        cigarettes_per_day=cigarettes if smoking_status == "Current smoker" else 0,
        systolic_bp=systolic,
        diastolic_bp=diastolic,
        resting_heart_rate=resting_hr,
        cholesterol=cholesterol,
        glucose=glucose,
        bp_meds=bp_meds,
        prevalent_stroke=prevalent_stroke,
        prevalent_hypertension=prevalent_hyp,
        diabetes=diabetes,
        chest_pain_type=chest_pain,
        resting_ecg=ecg,
        max_heart_rate=max_hr,
        exercise_angina=exercise_angina,
        oldpeak=oldpeak,
        st_slope=st_slope,
        residence_urban=residence,
        work_type=work_type,
    )


def main() -> None:
    st.set_page_config(
        page_title="CardioRisk AI",
        page_icon="heart",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    page_styles()

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Machine learning comparison dashboard</div>
            <h1>CardioRisk AI</h1>
            <p>An end-to-end data science project covering preprocessing, feature engineering, classical ML, neural-network classifiers, and model comparison across cardiovascular risk datasets.</p>
        </section>
        <div class="stat-strip">
            <div class="stat"><span>Core skills</span><strong>Preprocess</strong></div>
            <div class="stat"><span>Modeling</span><strong>ML vs NN</strong></div>
            <div class="stat"><span>Evaluation</span><strong>Compare</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    engine = sidebar_engine()
    st.sidebar.markdown("### Project notes")
    st.sidebar.caption("Portfolio project focused on data preprocessing, model training, comparison, and deployment. Not for clinical use.")

    try:
        artifacts = load_artifacts()
    except FileNotFoundError as exc:
        st.markdown(f'<div class="notice">{exc}</div>', unsafe_allow_html=True)
        st.stop()

    profile = collect_profile()
    run_prediction = st.button("Generate risk evaluation", type="primary", use_container_width=True)

    if run_prediction:
        model_inputs = build_inputs(profile)
        selected_models = artifacts[engine]
        scores = {key: predict_probability(selected_models[key], data) for key, data in model_inputs.items()}

        st.markdown(f'<div class="section-label">Risk evaluation - {engine}</div>', unsafe_allow_html=True)
        st.markdown('<div class="risk-grid">', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, key in zip(cols, ["chd", "heart", "stroke"], strict=True):
            with col:
                render_risk_card(key, scores[key])
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Review model-ready feature rows"):
            tab1, tab2, tab3 = st.tabs(["CHD", "Heart disease", "Stroke"])
            tab1.dataframe(model_inputs["chd"], use_container_width=True, hide_index=True)
            tab2.dataframe(model_inputs["heart"], use_container_width=True, hide_index=True)
            tab3.dataframe(model_inputs["stroke"], use_container_width=True, hide_index=True)
    else:
        st.info("Enter a profile and generate an evaluation to compare the selected model family.")


if __name__ == "__main__":
    main()
