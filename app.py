
import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_ann_model.keras"
SCALER_PATH = BASE_DIR / "diabetes_scaler.pkl"

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    /* Main page */
    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(34,211,238,.18), transparent 24%),
            radial-gradient(circle at 15% 90%, rgba(59,130,246,.12), transparent 25%),
            linear-gradient(135deg, #eaf7ff 0%, #f8fcff 48%, #e8f6ff 100%);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f7ff 0%, #dff2ff 100%);
        border-right: 1px solid #c7e4f7;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1.2rem 1rem;
    }

    .brand {
        padding: 8px 8px 20px 8px;
        border-bottom: 1px solid rgba(14,116,144,.15);
        margin-bottom: 16px;
    }

    .brand-icon {
        font-size: 40px;
        line-height: 1;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #0b2f63;
        margin-top: 7px;
    }

    .brand-subtitle {
        color: #147ca0;
        font-size: 13px;
        font-weight: 600;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border: none;
        background: transparent;
        color: #123b69;
        text-align: left;
        border-radius: 12px;
        min-height: 45px;
        font-size: 16px;
        font-weight: 650;
        margin: 3px 0;
        transition: .2s;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #cceeff;
        color: #075985;
        transform: translateX(2px);
    }

    .side-card {
        background: rgba(255,255,255,.72);
        border: 1px solid rgba(148,197,225,.35);
        border-radius: 18px;
        padding: 17px;
        margin-top: 15px;
        box-shadow: 0 8px 24px rgba(24,105,145,.07);
    }

    .side-card h4 {
        color: #075985;
        margin: 0 0 10px 0;
    }

    .side-point {
        margin: 12px 0;
        color: #21476e;
        font-size: 14px;
        line-height: 1.45;
    }

    .warning {
        background: #e9f5ff;
        border: 1px solid #9ed4f5;
        color: #075985;
        border-radius: 16px;
        padding: 14px;
        margin-top: 16px;
        font-size: 13px;
        line-height: 1.55;
    }

    /* Hero */
    .hero {
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(177,219,241,.75);
        border-radius: 24px;
        padding: 25px 30px;
        box-shadow: 0 12px 35px rgba(15,83,125,.10);
        position: relative;
        overflow: hidden;
        margin-bottom: 18px;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 270px;
        height: 270px;
        right: -100px;
        top: -130px;
        border-radius: 50%;
        background: rgba(34,211,238,.10);
    }

    .hero-title {
        font-size: clamp(30px, 4vw, 47px);
        font-weight: 850;
        color: #0b2f63;
        line-height: 1.05;
        margin: 0;
    }

    .hero-title span {
        color: #0891b2;
    }

    .hero-sub {
        color: #56718e;
        font-size: 15px;
        margin-top: 9px;
        margin-bottom: 13px;
    }

    .hero-pill {
        display: inline-block;
        background: #e5f5ff;
        color: #096b91;
        border: 1px solid #a9dcf6;
        border-radius: 30px;
        padding: 9px 17px;
        font-size: 14px;
        font-weight: 650;
    }

    .hero-right {
        text-align: center;
        font-size: 72px;
        padding-top: 10px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(182,218,236,.70);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 9px 28px rgba(15,83,125,.08);
        margin-bottom: 18px;
    }

    .card-title {
        color: #0b4f82;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .card-caption {
        color: #6b8298;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* Metrics */
    .metric-card {
        background: rgba(255,255,255,.88);
        border: 1px solid #cfe8f6;
        border-radius: 17px;
        padding: 16px;
        text-align: center;
        min-height: 108px;
        box-shadow: 0 6px 18px rgba(15,83,125,.06);
    }

    .metric-icon {
        font-size: 28px;
    }

    .metric-value {
        color: #0b4f82;
        font-size: 21px;
        font-weight: 800;
        margin-top: 4px;
    }

    .metric-label {
        color: #668099;
        font-size: 12px;
    }

    /* Inputs */
    div[data-testid="stNumberInput"] label {
        color: #173e67 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stNumberInput"] input {
        background: #f9fcff !important;
        color: #173e67 !important;
        border: 1px solid #bfdbea !important;
        border-radius: 11px !important;
        min-height: 43px !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #159ac0 !important;
        box-shadow: 0 0 0 2px rgba(21,154,192,.12) !important;
    }

    /* Predict button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #168bd0, #15b8bb) !important;
        color: white !important;
        border: none !important;
        border-radius: 13px !important;
        min-height: 54px !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        box-shadow: 0 7px 20px rgba(21,145,190,.23);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(21,145,190,.30);
    }

    /* Info / result */
    .result-positive {
        background: linear-gradient(135deg, #fff0f1, #fff8f8);
        border: 1px solid #ffb8bd;
        color: #a61b27;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }

    .result-negative {
        background: linear-gradient(135deg, #eafff6, #f6fffb);
        border: 1px solid #8ce0bd;
        color: #08724c;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }

    .result-title {
        font-size: 27px;
        font-weight: 850;
    }

    .probability {
        font-size: 38px;
        font-weight: 900;
        margin-top: 6px;
    }

    /* Right cards */
    .tip-card {
        background: linear-gradient(135deg, #effcff, #f9ffff);
        border: 1px solid #c7edf0;
        border-radius: 17px;
        padding: 17px;
        margin-bottom: 11px;
        min-height: 110px;
    }

    .tip-icon {
        font-size: 27px;
    }

    .tip-title {
        color: #0a5681;
        font-size: 15px;
        font-weight: 800;
        margin-top: 5px;
    }

    .tip-text {
        color: #658097;
        font-size: 12px;
        line-height: 1.4;
    }

    .quote {
        background: linear-gradient(135deg, #f4fbff, #ffffff);
        border-radius: 18px;
        border: 1px solid #cfe6f4;
        padding: 20px;
        color: #17618a;
        font-weight: 650;
        font-style: italic;
        line-height: 1.6;
    }

    /* Bottom strip */
    .bottom-strip {
        background: rgba(255,255,255,.9);
        border: 1px solid #cfe7f4;
        border-radius: 18px;
        padding: 17px;
        margin-top: 3px;
        box-shadow: 0 7px 20px rgba(15,83,125,.05);
    }

    .bottom-item {
        text-align: center;
        color: #164a75;
        font-weight: 750;
        font-size: 14px;
    }

    .bottom-item span {
        display: block;
        font-size: 28px;
        margin-bottom: 4px;
    }

    .footer {
        text-align: center;
        color: #6c879e;
        font-size: 12px;
        margin-top: 18px;
        padding-top: 13px;
        border-top: 1px solid #cde3ef;
    }

    /* Hide Streamlit menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 900px) {
        .hero-right { display: none; }
        .hero { padding: 22px; }
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


try:
    model, scaler = load_assets()
except Exception as e:
    st.error("❌ Model/Scaler load nahi ho pa raha.")
    st.code(str(e))
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "Health Monitor"

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction_prob" not in st.session_state:
    st.session_state.prediction_prob = None

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🩺</div>
        <div class="brand-title">Diabetes<br>Prediction</div>
        <div class="brand-subtitle">AI for a Healthier Tomorrow</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🏠  Health Monitor", key="nav_home"):
        st.session_state.page = "Health Monitor"
        st.rerun()

    if st.button("ℹ️  About Diabetes", key="nav_about"):
        st.session_state.page = "About Diabetes"
        st.rerun()

    if st.button("🔮  Prediction", key="nav_prediction"):
        st.session_state.page = "Prediction"
        st.rerun()

    if st.button("🌱  Healthy Tips", key="nav_tips"):
        st.session_state.page = "Healthy Tips"
        st.rerun()

    if st.button("🎯  Take Action", key="nav_action"):
        st.session_state.page = "Take Action"
        st.rerun()

    st.markdown("""
    <div class="side-card">
        <h4>💙 Why It Matters?</h4>
        <div class="side-point"><b>📊 Monitor Health</b><br>
        Track important health indicators.</div>
        <div class="side-point"><b>💡 Stay Aware</b><br>
        Early prediction can support better awareness.</div>
        <div class="side-point"><b>🌿 Live Better</b><br>
        Maintain healthy lifestyle habits.</div>
    </div>

   
    """, unsafe_allow_html=True)

# =========================================================
# COMMON HERO
# =========================================================
def show_hero():
    c1, c2 = st.columns([4.8, 1.4])

    with c1:
        st.markdown("""
        <div class="hero">
            <div class="hero-title">🩺 Diabetes <span>Prediction Dashboard</span></div>
            <div class="hero-sub">
                Artificial Neural Network based diabetes risk prediction system.
            </div>
            <div class="hero-pill">
                💓 Early Prediction &nbsp; | &nbsp; Better Care &nbsp; | &nbsp; Healthier Tomorrow
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="hero">
            <div class="hero-right">🩺<br>💙</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# HOME PAGE
# =========================================================
def home_page():
    show_hero()

    st.markdown("""
    <div class="card">
        <div class="card-title">👋 Welcome to Health Monitor</div>
        <div class="card-caption">
            A simple AI-powered dashboard for educational diabetes risk prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    cards = [
        ("🎯", "Early Detection", "Helps support timely awareness."),
        ("🛡️", "Better Care", "Supports healthier decisions."),
        ("👥", "Improved Life", "Awareness can encourage action."),
        ("💗", "Brighter Tomorrow", "Small steps can build healthier habits.")
    ]

    for col, (icon, title, text) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{title}</div>
                <div class="metric-label">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔬 How This System Works</div>
            <div class="card-caption">Simple 3-step prediction workflow</div>
        </div>
        """, unsafe_allow_html=True)

        a, b, c = st.columns(3)
        for col, icon, title, text in [
            (a, "📝", "1. Enter Data", "Provide patient health information."),
            (b, "🤖", "2. AI Analysis", "ANN model processes the inputs."),
            (c, "📊", "3. View Result", "See the predicted diabetes risk.")
        ]:
            with col:
                st.markdown(f"""
                <div class="tip-card">
                    <div class="tip-icon">{icon}</div>
                    <div class="tip-title">{title}</div>
                    <div class="tip-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="card">
            <div class="card-title">🌱 Live Health, Better Tomorrow</div>
            <div class="card-caption">Small steps, big impact!</div>
        </div>
        """, unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        tips = [
            ("🍎", "Healthy Food", "Eat balanced meals"),
            ("🏃", "Regular Exercise", "Stay active"),
            ("🌙", "Better Sleep", "Aim for 7–8 hours"),
            ("🪷", "Happier You", "Take care of yourself")
        ]

        for i, (icon, title, text) in enumerate(tips):
            with (t1 if i % 2 == 0 else t2):
                st.markdown(f"""
                <div class="tip-card">
                    <div class="tip-icon">{icon}</div>
                    <div class="tip-title">{title}</div>
                    <div class="tip-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <div class="quote">
        💬 “Take care of your body. It's the only place you have to live.”
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICTION FORM
# =========================================================
def prediction_page():
    show_hero()

    st.markdown("""
    <div class="card">
        <div class="card-title">👤 Patient Information</div>
        <div class="card-caption">
            Please enter the following eight values accurately. These inputs match the trained ANN model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("diabetes_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "🤰 Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
                help="Number of times pregnant."
            )

            glucose = st.number_input(
                "🩸 Glucose (mg/dL)",
                min_value=0.0,
                max_value=300.0,
                value=120.0,
                step=1.0,
                help="Plasma glucose concentration."
            )

            blood_pressure = st.number_input(
                "❤️ Blood Pressure (mm Hg)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
                help="Diastolic blood pressure."
            )

            skin_thickness = st.number_input(
                "🧍 Skin Thickness (mm)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Triceps skin fold thickness."
            )

        with col2:
            insulin = st.number_input(
                "💉 Insulin (µU/mL)",
                min_value=0.0,
                max_value=1000.0,
                value=80.0,
                step=1.0,
                help="Serum insulin level."
            )

            bmi = st.number_input(
                "⚖️ BMI (kg/m²)",
                min_value=0.0,
                max_value=80.0,
                value=25.0,
                step=0.1,
                help="Body Mass Index."
            )

            pedigree = st.number_input(
                "🧬 Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.47,
                step=0.01,
                help="Represents genetic influence."
            )

            age = st.number_input(
                "🎂 Age (years)",
                min_value=1,
                max_value=120,
                value=30,
                step=1,
                help="Patient age."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "✨  Predict Diabetes  →",
            type="primary",
            use_container_width=True
        )

    if submitted:
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age
        ]], dtype=float)

        try:
            scaled_data = scaler.transform(input_data)
            probability = float(model.predict(scaled_data, verbose=0)[0][0])

            st.session_state.prediction_done = True
            st.session_state.prediction_prob = probability

            if probability >= 0.5:
                st.markdown(f"""
                <div class="result-positive">
                    <div class="result-title">⚠️ Diabetes Predicted</div>
                    <div>Model estimated probability</div>
                    <div class="probability">{probability * 100:.1f}%</div>
                    <div>Please consult a qualified healthcare professional for proper evaluation.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div class="result-title">✅ No Diabetes Predicted</div>
                    <div>Model estimated probability</div>
                    <div class="probability">{probability * 100:.1f}%</div>
                    <div>Continue healthy habits and seek professional advice when appropriate.</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error("Prediction ke time error aaya.")
            st.exception(e)

    if st.session_state.prediction_done and st.session_state.prediction_prob is not None:
        p = st.session_state.prediction_prob

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Prediction Probability", f"{p*100:.1f}%")
        with c2:
            st.metric("Decision Threshold", "50%")
        with c3:
            st.metric("Model", "ANN")

# =========================================================
# ABOUT PAGE
# =========================================================
def about_page():
    show_hero()

    st.markdown("""
    <div class="card">
        <div class="card-title">ℹ️ About Diabetes</div>
        <div class="card-caption">Basic educational information</div>
        <p>
        Diabetes is a chronic condition associated with elevated blood glucose levels.
        This dashboard demonstrates how machine learning can be used for educational
        risk prediction from patient-related numerical features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔎 What the model uses</div>
            <div class="tip-card">🤰 <b>Pregnancies</b><br><span class="tip-text">Pregnancy count.</span></div>
            <div class="tip-card">🩸 <b>Glucose</b><br><span class="tip-text">Glucose concentration.</span></div>
            <div class="tip-card">❤️ <b>Blood Pressure</b><br><span class="tip-text">Diastolic blood pressure.</span></div>
            <div class="tip-card">⚖️ <b>BMI</b><br><span class="tip-text">Body Mass Index.</span></div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
            <div class="card-title">🤖 About the ANN</div>
            <div class="tip-card">🧠 <b>Artificial Neural Network</b><br><span class="tip-text">Learns patterns from the training dataset.</span></div>
            <div class="tip-card">📏 <b>Feature Scaling</b><br><span class="tip-text">The saved scaler prepares inputs in the same way as training.</span></div>
            <div class="tip-card">📊 <b>Binary Prediction</b><br><span class="tip-text">The output is interpreted using a 0.5 probability threshold.</span></div>
        </div>
        """, unsafe_allow_html=True)

   

# =========================================================
# HEALTHY TIPS PAGE
# =========================================================
def tips_page():
    show_hero()

    st.markdown("""
    <div class="card">
        <div class="card-title">🌱 Healthy Tips</div>
        <div class="card-caption">Simple habits that can support overall health</div>
    </div>
    """, unsafe_allow_html=True)

    tips = [
        ("🍎", "Balanced Food", "Choose a balanced diet and pay attention to portion sizes."),
        ("🏃", "Regular Activity", "Include regular physical activity according to your ability."),
        ("💧", "Stay Hydrated", "Drink adequate water and maintain healthy daily routines."),
        ("🌙", "Quality Sleep", "Maintain a consistent sleep schedule."),
        ("🧘", "Manage Stress", "Use healthy ways to relax and manage everyday stress."),
        ("🩺", "Regular Checkups", "Discuss health concerns and screening with a healthcare professional.")
    ]

    cols = st.columns(3)
    for i, (icon, title, text) in enumerate(tips):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="tip-card">
                <div class="tip-icon">{icon}</div>
                <div class="tip-title">{title}</div>
                <div class="tip-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# ACTION PAGE
# =========================================================
def action_page():
    show_hero()

    st.markdown("""
    <div class="card">
        <div class="card-title">🎯 Take Action</div>
        <div class="card-caption">What you can do after using the educational prediction tool</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">✅ If you want to learn more</div>
            <div class="side-point">1. Review the patient input values.</div>
            <div class="side-point">2. Understand what each feature represents.</div>
            <div class="side-point">3. Compare the prediction with professional medical guidance.</div>
            <div class="side-point">4. Maintain healthy lifestyle habits.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🩺 When to seek professional help</div>
            <div class="side-point">• For persistent or concerning symptoms.</div>
            <div class="side-point">• For interpretation of medical test results.</div>
            <div class="side-point">• Before making major health or treatment decisions.</div>
            <div class="side-point">• For personalized diabetes screening and care.</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔮 Go to Prediction", type="primary", use_container_width=True):
        st.session_state.page = "Prediction"
        st.rerun()

# =========================================================
# PAGE ROUTER
# =========================================================
if st.session_state.page == "Health Monitor":
    home_page()
elif st.session_state.page == "About Diabetes":
    about_page()
elif st.session_state.page == "Prediction":
    prediction_page()
elif st.session_state.page == "Healthy Tips":
    tips_page()
elif st.session_state.page == "Take Action":
    action_page()

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    🩺 Diabetes Prediction System &nbsp; | &nbsp;
    Powered by Artificial Neural Network &nbsp; | &nbsp;
    Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
