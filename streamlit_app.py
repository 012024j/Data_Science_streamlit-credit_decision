# streamlit_app.py
import streamlit as st
import base64
import io
from PIL import Image
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize

st.set_page_config(page_title="Dashboard Crédit", layout="wide")

# === Initialisation des flags nécessaires ===
if "show_pred" not in st.session_state:
    st.session_state["show_pred"] = False
if "show_shap" not in st.session_state:
    st.session_state["show_shap"] = False
if "show_shap_global_menu0" not in st.session_state:
    st.session_state["show_shap_global_menu0"] = False

# === Style CSS global et personnalisé ===
st.markdown("""
<style>
input, select, textarea {
    font-size: 16px !important;
    font-weight: bold !important;
    text-align: center !important;
}
label {
    display: flex;
    justify-content: center;
    font-size: 16px !important;
    font-weight: 600 !important;
    margin-bottom: 5px !important;
}
div.stButton > button:first-child {
    background-color: black;
    color: white;
    font-weight: bold;
    width: 100%;
    height: 45px;
    border-radius: 10px;
}
select {
    font-weight: bold !important;
    text-align-last: center !important;
}
.pred-result-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 20px;
}
.stTabs [data-baseweb="tab-list"] {
    justify-content: center !important;
    width: 100%;
    gap: 60px;
}
.stTabs [data-baseweb="tab"] > div {
    font-size: 20px !important;
    font-weight: 700 !important;
    font-family: 'Roboto', 'Arial', sans-serif !important;
    color: #333 !important;
    padding: 10px 0;
    transition: all 0.3s ease;
    text-align: center !important;
    border: 2px solid transparent;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] > div {
    color: red !important;
    border-bottom: 4px solid red !important;
}
.stTabs [data-baseweb="tab"] > div:hover {
    color: #e60000 !important;
    text-decoration: underline;
    border-color: #e60000 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("application_test.csv")

data = load_data()

API_URL = "https://streamlit-fastapi-app.onrender.com"
endpoint = f"{API_URL}/predict"

# Fonction pour convertir une image PIL en base64
def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

image_path = "credit_score_demo.png"
image = Image.open(image_path)
image_b64 = image_to_base64(image)

try:
    image = Image.open(image_path)
    image_b64 = image_to_base64(image)
except FileNotFoundError:
    st.warning("Image de démonstration introuvable.")
    image_b64 = ""


st.markdown(f"""
<div style="display: flex; flex-direction: column; align-items: center; margin-top: -60px;">
    <h2 style="margin-bottom: 10px; font-weight: bold; color: ##7FFFD4; font-size: 35px;">
        Credit scoring
    </h2>
    <img src="data:image/png;base64,{image_b64}" width="280" style="border-radius: 10px;"/>
</div>
""", unsafe_allow_html=True)

# Onglets du menu
menu = st.tabs([
    "Critères du client et son score de solvabilté", 
    "Comparaison client / population étudiée", 
    "Influence du profil client sur la décision finale de prêt"
])

# === Dictionnaire des noms lisibles ===
NOM_FEATURES = {
    "AMT_INCOME_TOTAL": "Revenu annuel (€)",
    "AMT_CREDIT": "Montant du crédit (€)",
    "AMT_ANNUITY": "Mensualité (€)",
    "CREDIT_TERM": "Durée de remboursement (mois)",
    "EXT_SOURCE_1": "Score externe de solvabilité n°1",
    "EXT_SOURCE_2": "Score externe de solvabilité n°2",
    "EXT_SOURCE_3": "Score externe de solvabilité n°3",
    "DAYS_BIRTH": "Âge (anéée)",
    "CODE_GENDER_M": "Genre (homme:0 - femme:1)",
    "CNT_CHILDREN": "Nombre d'enfants à charge",
    "DAYS_EMPLOYED": "Année d'ancienneté (emploi)",
    "CREDIT_INCOME_PERCENT": "Ratio crédit / revenu",
    "ANNUITY_INCOME_PERCENT": "Ratio mensualité / revenu",
    "DAYS_EMPLOYED_PERCENT": "Ratio ancienneté / âge",
    "NAME_INCOME_TYPE_Working": "Type de revenu (1: Travail)",
    "REGION_RATING_CLIENT_W_CITY": "Note région (avec ville)",
    "REGION_RATING_CLIENT": "Note région (générale)",
    "REG_CITY_NOT_WORK_CITY": "Travail dans une autre ville / (1=Oui)",
    "FLAG_OWN_REALTY": "Propriétaire d'un bien immobilier / (1=Oui)",
    "OCCUPATION_TYPE_Laborers": "Travailleur de la classe ouvrière / (1=Oui)"
}

# Inversion pour selectbox/menu déroulant
NOM_FEATURES_INV = {v: k for k, v in NOM_FEATURES.items()}

# Onglet 0 : Saisie des informations client et prédiction
# Ajout d’un style global pour les champs de saisie
st.markdown("""
<style>
input[type="number"], select {
    background-color: #f8f9fa !important;
    font-size: 16px !important;
    padding: 8px 10px !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# Fonction pour afficher un label stylisé
def display_label(text):
    st.markdown(f"""
        <div style='
            background-color: #001f3f;
            color: white;
            font-weight: 700;
            font-size: 16px;
            text-align: center;
            padding: 10px 5px;
            border-radius: 10px;
            margin-bottom: 8px;
        '>{text}</div>
    """, unsafe_allow_html=True)

# ===============================
# Onglet 0 : Saisie client - partie améliorée

# Style global
st.markdown("""
<style>
input[type="number"], select {
    background-color: #f8f9fa !important;
    font-size: 16px !important;
    padding: 8px 10px !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# Fonction d'affichage des titres
def display_label(text):
    st.markdown(f"""
        <div style='
            background-color: #001f3f;
            color: white;
            font-weight: 700;
            font-size: 16px;
            text-align: center;
            padding: 10px 5px;
            border-radius: 10px;
            margin-bottom: 8px;
        '>{text}</div>
    """, unsafe_allow_html=True)

with menu[0]:
    # === Titre du formulaire ===
    st.markdown("<h4 style='text-align: center; margin-top: 2.5px;'>Saisir les caractéristiques du client</h4>", unsafe_allow_html=True)

    # === Saisie des montants principaux ===
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Revenu annuel (€)</div>", unsafe_allow_html=True)
        amt_income = st.number_input("", value=100000, key="income")
    with col2:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Montant du crédit (€)</div>", unsafe_allow_html=True)
        amt_credit = st.number_input("", value=5000, key="credit")
    with col3:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Mensualité du crédit (€)</div>", unsafe_allow_html=True)
        amt_annuity = st.number_input("", value=100, key="annuity")

    # === Saisie âge, durée et ancienneté emploi ===
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Âge (années)</div>", unsafe_allow_html=True)
        age = st.number_input("", value=45, key="age")
    with col5:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Durée de remboursement (mois)</div>", unsafe_allow_html=True)
        credit_term = st.number_input("", value=24, key="term")
    with col6:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Année d'ancièneté (emploi)</div>", unsafe_allow_html=True)
        anciennete = st.number_input("", value=20, key="anciennete")

    # === Bloc des scores de solvabilité ===
    col7, col8, col9 = st.columns(3)

    with col7:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Score de solvabilité (externe n°1)</div>", unsafe_allow_html=True)
        ext1 = st.number_input("", value=0.95, key="ext1")

    with col8:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Score de solvabilité (externe n°2)</div>", unsafe_allow_html=True)
        ext2 = st.number_input("", value=0.96, key="ext2")

    with col9:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Score de solvabilité (externe n°3)</div>", unsafe_allow_html=True)
        ext3 = st.number_input("", value=0.97, key="ext3")

    # === Bloc centré d’explication (sans décaler la mise en page) ===
    st.markdown("<br>", unsafe_allow_html=True)  # petit espacement
    # Centrage propre du toggle et du texte
    # Ajustement du centrage vers la droite
    col1, col2, col3 = st.columns([2.5, 2, 1.5])  # tu peux tester [3, 2, 1] aussi
    with col2:
        show_info = st.toggle("ℹ️ En savoir plus sur le score de solvabilité", key="toggle_info_score_all")

    if show_info:
        st.markdown("""
        <div style='
            background-color: #f8f9fa;
            border-left: 4px solid #007BFF;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 14px;
            line-height: 1.5;
            max-width: 900px;
            margin: 10px auto 30px auto;
        '>
            Le score de solvabilité est un indicateur numérique représentant la capacité du client à rembourser un crédit.<br>
            Plus le score est élevé (proche de 1), plus la probabilité de remboursement est forte.<br>
            Ce score est fourni par des organismes extérieurs spécialisés et est calculé à partir de données financières, professionnelles et personnelles.
        </div>
        """, unsafe_allow_html=True)


    # === Informations régionales et enfants ===
    col10, col11, col12 = st.columns(3)
    with col10:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Nombre d'enfants à charge</div>", unsafe_allow_html=True)
        cnt_children = st.number_input("  ", min_value=0, value=0, key="cnt_children")
    with col11:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>📍 Note de la région (ville, meilleure score 3)</div>", unsafe_allow_html=True)
        region_city_rating = st.selectbox("  ", [1, 2, 3], index=2, key="region_city")
    with col12:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>📍 Note de la région (générale, meilleure score 3)</div>", unsafe_allow_html=True)
        region_rating = st.selectbox("  ", [1, 2, 3], index=2, key="region_general")

    # === Informations démographiques et professionnelles ===
    col13, col14, col15, col16, col17 = st.columns(5)
    with col13:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Genre</div>", unsafe_allow_html=True)
        genre = st.radio(" ", ["Homme", "Femme"], index=1, key="genre")
        code_gender_m = int(genre == "Homme")

    with col14:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Travail dans une autre ville ?</div>", unsafe_allow_html=True)
        city_diff = int(st.radio(" ", ["Oui", "Non"], index=1, key="travail_ville") == "Oui")

    with col15:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Propriétaire d'un bien immobilier ?</div>", unsafe_allow_html=True)
        own_realty = int(st.radio(" ", ["Oui", "Non"], index=0, key="realty") == "Oui")

    with col16:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Travailleur de la classe ouvrière ?</div>", unsafe_allow_html=True)
        laborer = int(st.radio(" ", ["Oui", "Non"], index=1, key="laborer") == "Oui")

    with col17:
        st.markdown("<div style='text-align: center; font-weight: 600; font-size: 19px;'>Type de revenu</div>", unsafe_allow_html=True)
        income_type = st.radio(" ", ["Travail", "Autre revenu"], index=0, key="income_type")
        income_working = int(income_type == "Travail")

     # === Calcul des ratios ===
    days_birth = -age * 365
    days_employed = -anciennete * 365
    credit_income_percent = amt_credit / amt_income
    annuity_income_percent = amt_annuity / amt_income
    days_employed_percent = abs(days_employed) / abs(days_birth)
    code_gender = genre == "Genre"

    # === Affichage des ratios sous forme de cartes ===
    col_ratio1, col_ratio2, col_ratio3 = st.columns(3)
    with col_ratio1:
        st.markdown(f"""
            <div style="background-color:#d0ebff; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-weight:700; font-size:20px; color:#000;">💳 Crédit / Revenu</div>
                <div style="font-size:20px; color:#000;"><b>{credit_income_percent * 100:.2f}%</b></div>
            </div>
        """, unsafe_allow_html=True)

    with col_ratio2:
        st.markdown(f"""
            <div style="background-color:#d3f9d8; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-weight:700; font-size:20px; color:#000;">📅 Mensualité / Revenu</div>
                <div style="font-size:20px; color:#000;"><b>{annuity_income_percent * 100:.2f}%</b></div>
            </div>
        """, unsafe_allow_html=True)

    with col_ratio3:
        st.markdown(f"""
            <div style="background-color:#fff3bf; padding:15px; border-radius:10px; text-align:center;">
                <div style="font-weight:700; font-size:20px; color:#000;">⏳ Ancienneté / Âge</div>
                <div style="font-size:20px; color:#000;"><b>{days_employed_percent * 100:.2f}%</b></div>
            </div>
        """, unsafe_allow_html=True)

    # === Création du dictionnaire de caractéristiques ===
    features = {
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "CREDIT_INCOME_PERCENT": credit_income_percent,
        "ANNUITY_INCOME_PERCENT": annuity_income_percent,
        "CREDIT_TERM": credit_term,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "DAYS_BIRTH": days_birth,
        "CODE_GENDER_M": code_gender,
        "CNT_CHILDREN": cnt_children,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_EMPLOYED_PERCENT": days_employed_percent,
        "NAME_INCOME_TYPE_Working": income_working,
        "REGION_RATING_CLIENT_W_CITY": region_city_rating,
        "REGION_RATING_CLIENT": region_rating,
        "REG_CITY_NOT_WORK_CITY": city_diff,
        "FLAG_OWN_REALTY": own_realty,
        "OCCUPATION_TYPE_Laborers": laborer,
    }

def format_value(feature_name, value):
    if feature_name == "CODE_GENDER_M":
        return "Genre" if value else "Femme"
    elif feature_name == "NAME_INCOME_TYPE_Working":
        return "Travail" if value else "Autre revenu"
    elif feature_name == "REG_CITY_NOT_WORK_CITY":
        return "Oui" if value else "Non"
    elif feature_name == "FLAG_OWN_REALTY":
        return "Oui" if value else "Non"
    elif feature_name == "OCCUPATION_TYPE_Laborers":
        return "Oui" if value else "Non"
    else:
        return value
    
    # === Boutons prédiction (à gauche) et SHAP global (à droite) ===
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
col_pred, col_shap = st.columns([1, 1])

with col_pred:
    if st.button("Évaluer la solvabilité du client", use_container_width=True, key="btn_predire"):
        st.session_state["show_pred"] = True

with col_shap:
    if st.button("💡 Importance globale : critères d'évaluation de risque crédit", use_container_width=True, key="btn_shap_global_menu0"):
        st.session_state["show_shap_global_menu0"] = True

# === Résultat prédiction et SHAP global côte à côte ===
graph_col1, graph_col2 = st.columns(2)

# Bloc prédiction
with graph_col1:
    if st.session_state.get("show_pred"):
        response = requests.post(endpoint, json={"features": features})
        if response.status_code == 200:
            result = response.json()
            bg_color = "#d3f9d8" if result['classe'] == "accepté" else "#ffe3e3"
            text_color = "#2f9e44" if result['classe'] == "accepté" else "#c92a2a"
            st.markdown(f"""
            <div class='pred-result-box'>
                <div style='background-color:{bg_color}; padding:10px; border-radius:8px; text-align:left; margin-bottom:10px;'>
                    <span style='color:{text_color}; font-size:18px; font-weight:bold;'>Décision d'octroi de crédit : {result['classe']}</span>
                </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["proba_defaut"] * 100,
                title={"text": "Probabilité défaut"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "crimson"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4f4dd"},
                        {"range": [30, 70], "color": "#fff4ce"},
                        {"range": [70, 100], "color": "#f9dcdc"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": result["proba_defaut"] * 100}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Bloc SHAP global (population)
with graph_col2:
    if st.session_state.get("show_shap_global_menu0"):
        sample = data.sample(n=min(1000, len(data)), random_state=42)
        X_sample = sample[features.keys()]
        model = joblib.load("Best_XGBoost_Business_Model.pkl")
        booster_model = model.named_steps["clf"]

        shap_vals = shap.Explainer(booster_model)(X_sample)
        abs_vals = np.abs(shap_vals.values).mean(axis=0)

        NOM_FEATURES = {
            "EXT_SOURCE_3": "Score de solvabilité externe n°3",
            "EXT_SOURCE_2": "Score de solvabilité externe n°2",
            "EXT_SOURCE_1": "Score de solvabilité externe n°1",
            "CODE_GENDER_M": "Genre",
            "AMT_INCOME_TOTAL": "Revenu annuel (€)",
            "AMT_CREDIT": "Montant crédit (€)",
            "AMT_ANNUITY": "Mensualité (€)",
            "CREDIT_TERM": "Durée crédit (mois)",
            "CNT_CHILDREN": "Nombre enfants",
            "DAYS_EMPLOYED": "Ancienneté emploi",
            "DAYS_EMPLOYED_PERCENT": "Ancienneté / âge",
            "REGION_RATING_CLIENT": "Note de la région (ville)",
            "NAME_INCOME_TYPE_Working": "Typologie de revenu",
            "REG_CITY_NOT_WORK_CITY": "Travail hors de sa ville",
            "FLAG_OWN_REALTY": "Propriétaire d'un bien immobilier",
            "OCCUPATION_TYPE_Laborers": "Ouvrier"
        }

        shap_df = pd.DataFrame({
            "Critère": [NOM_FEATURES.get(k, k) for k in features.keys()],
            "Importance": abs_vals
        }).sort_values(by="Importance", ascending=False).head(10)

        norm = Normalize(vmin=shap_df["Importance"].min(), vmax=shap_df["Importance"].max())
        colors = cm.Blues(norm(shap_df["Importance"]))
        
        fig, ax = plt.subplots(figsize=(10, 0.7 * len(shap_df) + 1))
        bars = ax.barh(shap_df["Critère"], shap_df["Importance"], color=colors, height=0.9)

        ax.set_xlabel("Importance moyenne (population étudiée)")
        ax.set_title("Top 10 des critères qui influencent le plus l'accord de crédit (selon l'ensemble des clients étudiés)", fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.invert_yaxis()

        # ✅ Supprimer les bordures du cadre
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)

        # ✅ Affichage des valeurs numériques proprement
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va='center',
                ha='left',
                fontsize=10
            )
        # ✅ Afficher dans Streamlit
        st.pyplot(fig)

        # Une fois affiché, désactiver la clé
        st.session_state["show_shap_global_menu0"] = False

# === Fonction pour afficher une valeur client formatée correctement ===
def formatter_valeur(var_label, valeur):
    label = var_label.lower()

    if "âge" in label:
        return f"{int(valeur)} ans"
    elif "ancienneté emploi" in label or "days_employed" in label:
        return f"{abs(valeur) // 365} ans"
    elif "mois" in label or "durée du crédit" in label:
        return f"{int(valeur)} mois"
    elif "mensualité" in label or ("revenu" in label and "ratio" not in label) or "montant" in label:
        return f"{int(valeur)} €"
    elif "score" in label:
        return f"{round(valeur, 2)}"
    elif "ratio" in label:
        return f"{round(valeur * 100, 2)} %"
    elif "note région" in label:
        return f"{int(valeur)}"
    elif "travail" in label or "propriétaire" in label or "classe ouvrière" in label or "type de revenu" in label:
        return str(int(valeur))
    elif "enfants" in label:
        return f"{int(valeur)}"
    else:
        return str(valeur)


# === Bloc comparaison avec la population ===
with menu[1]:
    st.markdown("<div style='text-align: center; font-size: 25px; font-weight: 600;'>Sélectionnez le critère client à comparer avec la population étudiée :</div>", unsafe_allow_html=True)

    data_display = data.copy()
    NOM_FEATURES_INV = {v: k for k, v in NOM_FEATURES.items()}

    var_label = st.selectbox(" ", list(NOM_FEATURES_INV.keys()))
    var = NOM_FEATURES_INV[var_label]

    if var == "DAYS_BIRTH":
        data_display["DAYS_BIRTH"] = abs(data_display["DAYS_BIRTH"]) // 365
        features["DAYS_BIRTH"] = abs(features["DAYS_BIRTH"]) // 365

    if var == "DAYS_EMPLOYED":
        data_display["DAYS_EMPLOYED"] = abs(data_display["DAYS_EMPLOYED"]) // 365
        features["DAYS_EMPLOYED"] = abs(features["DAYS_EMPLOYED"]) // 365

    if var in data.columns:
        fig = px.box(data_display, y=var, points="all", labels={var: var_label})
        valeur_client = features[var]
        texte_client = formatter_valeur(var_label, valeur_client)

        fig.add_scatter(
            x=[0],
            y=[valeur_client],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Client",
            text=[texte_client],
            hoverinfo="text"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Variable non disponible dans le dataset.")

    # === Graphe croisé ===
    st.markdown("<div style='text-align: center; font-size: 25px; font-weight: 600;'>Visualisez deux critères du dossier client à comparer :</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size: 24px;'>Critère n°1</div>", unsafe_allow_html=True)
    x_label = st.selectbox("", list(NOM_FEATURES_INV.keys()), key="x")
    x = NOM_FEATURES_INV[x_label]

    st.markdown("<div style='font-size: 24px;'>Critère n°2</div>", unsafe_allow_html=True)
    y_label = st.selectbox("", list(NOM_FEATURES_INV.keys()), key="y")
    y = NOM_FEATURES_INV[y_label]

    for var_check in [x, y]:
        if var_check == "DAYS_BIRTH":
            data_display["DAYS_BIRTH"] = abs(data["DAYS_BIRTH"]) // 365
            features["DAYS_BIRTH"] = abs(features["DAYS_BIRTH"]) // 365
        if var_check == "DAYS_EMPLOYED":
            data_display["DAYS_EMPLOYED"] = abs(data["DAYS_EMPLOYED"])
            features["DAYS_EMPLOYED"] = abs(features["DAYS_EMPLOYED"])

    if x in data.columns and y in data.columns:
        fig2 = px.scatter(data_display, x=x, y=y, opacity=0.4, labels={x: x_label, y: y_label})
        fig2.add_scatter(
            x=[features[x]],
            y=[features[y]],
            mode="markers",
            marker=dict(color="red", size=12),
            name="Client"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Variables non disponibles dans le dataset.")

# === SHAP LOCALE : Explication personnalisée ===
with menu[2]:
    # Titre du tableau avec texte agrandi
    st.markdown("<div style='text-align: center; font-size: 25px; font-weight: 600;'> Les critères les plus influents sur le résultat de la prédiction du risque crédit </div>", unsafe_allow_html=True)


    # Chargement du modèle
    model = joblib.load("Best_XGBoost_Business_Model.pkl")
    booster_model = model.named_steps["clf"]

    explainer = shap.Explainer(booster_model)
    X_client = np.array([list(features.values())])
    shap_values = explainer(X_client)

    # Construction du DataFrame SHAP
    feature_names_readable = [NOM_FEATURES.get(f, f) for f in features.keys()]
    shap_df = pd.DataFrame({
        "Critère d'influence": feature_names_readable,
        "Valeur": [v if isinstance(v, (int, float)) else np.nan for v in features.values()],
        "Facteurs déterminants dans la décision de crédit du client": shap_values.values[0]
    }).sort_values(by="Facteurs déterminants dans la décision de crédit du client", key=np.abs, ascending=False).head(10)



# === SHAP LOCALE : Explication personnalisée ===
with menu[2]:
    # Chargement du modèle et préparation
    model = joblib.load("Best_XGBoost_Business_Model.pkl")
    booster_model = model.named_steps["clf"]
    explainer = shap.Explainer(booster_model)

    X_client = np.array([list(features.values())])
    shap_values = explainer(X_client)

    # Construction du DataFrame SHAP
    feature_names_readable = [NOM_FEATURES.get(f, f) for f in features.keys()]
    shap_impact = shap_values.values[0].tolist()

    shap_df = pd.DataFrame({
        "Critère d'influence": feature_names_readable,
        "Valeur": [format_value(f, features[f]) for f in features.keys()],
        "Facteurs déterminants dans la décision de crédit du client": shap_impact
    }).sort_values(by="Facteurs déterminants dans la décision de crédit du client", key=np.abs, ascending=False).head(10)

    # Mise en forme (nom en gras)
    shap_df["Critère d'influence"] = shap_df["Critère d'influence"].apply(lambda x: f"<b>{x}</b>")

    # Conversion HTML du tableau
    table_html = shap_df.to_html(
        index=False,
        escape=False,
        border=0,
        classes='shap-table',
        justify='center',
        float_format=lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else ""
    )

    # CSS styling
    st.markdown("""
        <style>
        .shap-table {
            font-size: 18px;
            font-family: Arial, sans-serif;
            width: 80%;
            border-collapse: collapse;
            margin: 0 auto;
        }
        .shap-table th {
            text-align: center;
            padding: 12px;
            background-color: #000000;
            color: white;
            font-size: 20px;
        }
        .shap-table td {
            text-align: center;
            padding: 10px;
        }
        .shap-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .shap-table tr:hover {
            background-color: #eef;
        }
        .download-container {
            display: flex;
            justify-content: center;
            margin-top: 25px;
        }
        .custom-download-button button {
            background-color: #444;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        .custom-download-button button:hover {
            background-color: #000;
        }
        </style>
    """, unsafe_allow_html=True)

    # Affichage du tableau
    st.markdown("<div style='display: flex; justify-content: center;'>"
                f"{table_html}"
                "</div>", unsafe_allow_html=True)

    # Export CSV
    csv_data = shap_df.copy()
    csv_data["Critère d'influence"] = csv_data["Critère d'influence"].str.replace("<b>", "").str.replace("</b>", "")
    csv_bytes = csv_data.to_csv(index=False).encode("utf-8")

    # Bouton de téléchargement
    col1, col2, col3 = st.columns([2.5, 2.5, 1])
    with col2:
        st.download_button(
            label="📥 Télécharger les données au format CSV",
            data=csv_bytes,
            file_name="explication_shap_locale.csv",
            mime="text/csv",
            key="shap_download",
            help="Cliquez pour enregistrer le tableau",
        )

    # === GRAPH SHAP ===
    top_n = 10
    shap_df_clean = shap_df.copy()
    shap_df_clean["Critère d'influence"] = shap_df_clean["Critère d'influence"].str.replace("<b>", "", regex=False).str.replace("</b>", "", regex=False)

    top_shap_df = shap_df_clean.head(top_n).sort_values("Facteurs déterminants dans la décision de crédit du client", ascending=True)

    # Couleur normalisée
    abs_norm = Normalize(
        vmin=top_shap_df["Facteurs déterminants dans la décision de crédit du client"].abs().min(),
        vmax=top_shap_df["Facteurs déterminants dans la décision de crédit du client"].abs().max()
    )
    colors = cm.Blues(abs_norm(top_shap_df["Facteurs déterminants dans la décision de crédit du client"].abs()))

    # Titre
    st.markdown("<h2 style='text-align: center; color: black;'>📈 Visualisation des facteurs d'influence dans la décision de crédit du client</h2>", unsafe_allow_html=True)

    # Graphique
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(
        top_shap_df["Critère d'influence"],
        top_shap_df["Facteurs déterminants dans la décision de crédit du client"],
        color=colors
    )

    # Suppression des bordures
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Axe X et Y
    ax.set_xlabel("Impact sur la décision", fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9, pad=8)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Affichage des valeurs numériques
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va='center',
                ha='left' if width > 0 else 'right',
                fontsize=8, color='black')

    plt.subplots_adjust(left=0.15)
    st.pyplot(fig)

