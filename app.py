import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration de la page
# -----------------------------------------------------------------------------
BASE = Path(__file__).parent
st.set_page_config(
    page_title="MediRisk | Priorisation Clinique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS avancé (Thème Dark/Light Slate Clinical)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    .main-header h1 {
        color: #f8fafc !important;
        font-weight: 700;
        font-size: 2rem !important;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        margin-top: 0.4rem;
        font-size: 0.95rem;
    }

    /* Cartes de métriques & Résultats */
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    
    /* Badges de risque personnalisé */
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-high { background-color: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
    .badge-mid { background-color: #fef3c7; color: #d97706; border: 1px solid #fcd34d; }
    .badge-low { background-color: #dcfce7; color: #16a34a; border: 1px solid #86efac; }

    /* Amélioration des Onglets (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        font-weight: 500;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Bouton d'évaluation */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load():
    model = joblib.load(BASE / "models" / "rf_model.joblib")
    scaler = joblib.load(BASE / "models" / "scaler_clustering.joblib")
    kmeans = joblib.load(BASE / "models" / "kmeans_k5.joblib")
    cfg = json.loads((BASE / "models" / "config.json").read_text(encoding="utf-8"))
    return model, scaler, kmeans, cfg


model, scaler, kmeans, cfg = load()

CLUST_COLS = [
    "age_num", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "number_outpatient",
    "number_emergency", "number_inpatient", "n_meds", "n_med_changes"
]


def clustering_features(d):
    Z = d[CLUST_COLS].copy()
    Z["change"] = (d["change"] == "Ch").astype(int)
    Z["diabetesMed"] = (d["diabetesMed"] == "Yes").astype(int)
    return Z


def risk_level(p):
    t = cfg["thresholds"]
    return "Élevé" if p >= t["high"] else ("Moyen" if p >= t["mid"] else "Faible")


DEFAULTS = {
    "race": "Caucasian", "gender": "Female", "medical_specialty": "Unknown",
    "diag_1": "Circulatory", "diag_2": "Circulatory", "diag_3": "Unknown",
    "A1Cresult": "Not_tested", "max_glu_serum": "Not_tested",
    "change": "No", "diabetesMed": "Yes", "discharge_group": "Home",
    "admission_type": "Emergency", "admission_source": "Emergency_room"
}


def sel(label, key):
    opts = cfg["cat_options"][key]
    idx = opts.index(DEFAULTS[key]) if DEFAULTS.get(key) in opts else 0
    return st.selectbox(label, opts, index=idx)


def num(label, key):
    lo, hi, med = (int(v) for v in cfg["num_ranges"][key])
    return st.slider(label, lo, hi, med)


MED_LEVELS = ["No", "Steady", "Up", "Down"]

# -----------------------------------------------------------------------------
# En-tête de l'application
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏥 MediRisk — Support de Décision Clinique</h1>
    <p>Évaluation prédictive du risque de réadmission à 30 jours pour patients diabétiques (Dataset Diabetes 130-US Hospitals)</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Barre latérale (Profil de base du patient)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("OIP (2).webp", width=70)
    st.title("Profil Données")
    st.caption("Information Démographiques")
    
    age_num = st.select_slider("Âge (centre de tranche)", options=list(range(5, 100, 10)), value=65)
    gender = sel("Genre", "gender")
    race = sel("Origine", "race")
    
    st.divider()
    st.info("💡 **Conseil** : Complétez les onglets principaux puis cliquez sur **Lancer l'Évaluation**.")

# -----------------------------------------------------------------------------
# Formulaire principal interactif
# -----------------------------------------------------------------------------
with st.form("patient_form", border=False):
    tab1, tab2, tab3 = st.tabs([
        "🏥 Hospitalisation & Modalités", 
        "🔬 Biologie & Historique Médical", 
        "💊 Thérapeutique Antidiabétique"
    ])

    with tab1:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("##### 📍 Séjour Actuel")
            time_in_hospital = num("Durée d'hospitalisation (jours)", "time_in_hospital")
            medical_specialty = sel("Spécialité médicale référente", "medical_specialty")
            discharge_group = sel("Orientation / Destination de sortie", "discharge_group")

        with col2:
            st.markdown("##### 📥 Conditions d'Admission")
            admission_type = sel("Type d'admission", "admission_type")
            admission_source = sel("Origine de l'admission", "admission_source")

    with tab2:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("##### 🩺 Examens & Actes du Séjour")
            num_lab_procedures = num("Nombre d'analyses de laboratoire", "num_lab_procedures")
            num_procedures = num("Nombre d'actes / procédures", "num_procedures")
            num_medications = num("Nombre de médicaments administrés", "num_medications")
            number_diagnoses = num("Nombre de diagnostics posés", "number_diagnoses")

        with col2:
            st.markdown("##### 📊 Diagnostics & Historique (12 Mois)")
            diag_1 = sel("Diagnostic principal (CIM-9)", "diag_1")
            diag_2 = sel("Diagnostic secondaire", "diag_2")
            diag_3 = sel("Diagnostic tertiaire", "diag_3")
            
            c_a, c_b = st.columns(2)
            with c_a:
                A1Cresult = sel("HbA1c", "A1Cresult")
            with c_b:
                max_glu_serum = sel("Glycémie", "max_glu_serum")

            st.caption("Visites durant les 12 derniers mois")
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                number_outpatient = num("Consultations", "number_outpatient")
            with cv2:
                number_emergency = num("Urgences", "number_emergency")
            with cv3:
                number_inpatient = num("Hospitalisations", "number_inpatient")

    with tab3:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("##### 💉 Traitements Majeurs")
            
            c_med1, c_med2 = st.columns(2)
            with c_med1:
                insulin = st.selectbox("Insuline", MED_LEVELS)
                metformin = st.selectbox("Metformine", MED_LEVELS)
            with c_med2:
                glipizide = st.selectbox("Glipizide", MED_LEVELS)
                glyburide = st.selectbox("Glyburide", MED_LEVELS)
                
            change = sel("Ajustement global de traitement", "change")
            diabetesMed = sel("Traitement antidiabétique prescrit", "diabetesMed")

        with col2:
            st.markdown("##### 🧪 Traitements Antidiabétiques Secondaires")
            st.write("")
            c_ck1, c_ck2 = st.columns(2)
            with c_ck1:
                repaglinide = st.checkbox("Répaglinide")
                glimepiride = st.checkbox("Glimépiride")
            with c_ck2:
                pioglitazone = st.checkbox("Pioglitazone")
                rosiglitazone = st.checkbox("Rosiglitazone")

    st.write("")
    submitted = st.form_submit_button("⚡ Lancer l'Évaluation du Patient", use_container_width=True)

# -----------------------------------------------------------------------------
# Analyse des résultats & Cartes de Synthèse
# -----------------------------------------------------------------------------
if submitted:
    main = [insulin, metformin, glipizide, glyburide]
    rare = [repaglinide, glimepiride, pioglitazone, rosiglitazone]
    n_meds = sum(m != "No" for m in main) + sum(rare)
    n_med_changes = sum(m in ("Up", "Down") for m in main)
    total_visits = min(number_outpatient + number_emergency + number_inpatient, 10)

    row = dict(
        race=race, gender=gender, time_in_hospital=time_in_hospital,
        medical_specialty=medical_specialty, num_lab_procedures=num_lab_procedures,
        num_procedures=num_procedures, num_medications=num_medications,
        number_outpatient=number_outpatient, number_emergency=number_emergency,
        number_inpatient=number_inpatient, diag_1=diag_1, diag_2=diag_2, diag_3=diag_3,
        number_diagnoses=number_diagnoses, max_glu_serum=max_glu_serum, A1Cresult=A1Cresult,
        metformin=metformin, repaglinide=int(repaglinide), glimepiride=int(glimepiride),
        glipizide=glipizide, glyburide=glyburide, pioglitazone=int(pioglitazone),
        rosiglitazone=int(rosiglitazone), insulin=insulin, change=change,
        diabetesMed=diabetesMed, discharge_group=discharge_group,
        admission_type=admission_type, admission_source=admission_source,
        age_num=age_num, n_meds=n_meds, n_med_changes=n_med_changes,
        total_visits=total_visits,
    )
    d = pd.DataFrame([row])[cfg["feature_order"]]

    p = float(model.predict_proba(d)[:, 1][0])
    level = risk_level(p)
    cluster = int(kmeans.predict(scaler.transform(clustering_features(d)))[0])

    st.write("")
    st.subheader("📊 Diagnostic Prédictif & Classification")

    res_col1, res_col2 = st.columns(2, gap="large")

    # Carte 1: Risque Prédictif
    with res_col1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Niveau de Risque Prédictif")
        
        rate = cfg["risk_rates"][level]
        badge_class = {"Élevé": "badge-high", "Moyen": "badge-mid", "Faible": "badge-low"}[level]
        
        st.markdown(f'<span class="risk-badge {badge_class}">Risque {level}</span>', unsafe_allow_html=True)
        st.write("")
        
        # Jauge de probabilité visuelle
        st.write(f"**Probabilité d'appartenance au groupe à risque : {p:.1%}**")
        st.progress(p)
        
        st.metric(
            label="Taux de réadmission observé pour cette strate",
            value=f"{rate:.0%}",
            delta=f"{rate - cfg['global_rate']:+.1%} vs Taux Moyen ({cfg['global_rate']:.1%})"
        )
        st.caption("Source : Modèle Random Forest équilibré sur la cohorte hôpitaux US.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Carte 2: Profil Cluster
    with res_col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### 🧩 Typologie Patient (Clustering)")
        
        cluster_name = cfg['cluster_names'][str(cluster)]
        cluster_rate = cfg['cluster_rates'][str(cluster)]
        
        st.markdown(f"**Profil Identifié : Cluster #{cluster}**")
        st.info(f"**{cluster_name}**")
        
        st.metric(
            label="Incidence de réadmission dans ce profil typologique",
            value=f"{cluster_rate:.0%}"
        )
        st.caption("L'appartenance à un cluster décrit la complexité clinique globale et est indépendante du score de risque.")
        st.markdown('</div>', unsafe_allow_html=True)