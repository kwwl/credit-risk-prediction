import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import matplotlib.pyplot as plt


# chargement du model via google drive
@st.cache_data(show_spinner=True)
def download_model(file_id, destination="models/pipeline_model.pkl"):
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(destination):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(destination, "wb") as f:
            f.write(response.content)
    return joblib.load(destination)


pipeline = download_model("1diBY4e-AVEwVj1mpt2bUabju1_dS-wkA")


st.title("Prédiction de risque de prêt")

st.subheader("Comment fonctionne cette IA ?")
st.write(
    """
Cette application utilise un pipeline Machine Learning complet :
- Prétraitement des données (StandardScaler + OneHotEncoder)
- Gestion du déséquilibre avec SMOTE
- Classificateur XGBoost entraîné sur des données historiques
"""
)

st.header("Informations du dossier")


age = st.number_input("Âge", min_value=18, max_value=100, value=30)
income = st.number_input("Revenu annuel (€)", min_value=0, value=30000)
loan_amount = st.number_input("Montant du prêt (€)", min_value=0, value=10000)
months_employed = st.number_input("Mois d'emploi", min_value=0, value=12)
interest_rate = st.number_input(
    "Taux d'intérêt (%)", min_value=0.0, max_value=100.0, value=5.0
)
loan_term = st.number_input("Durée du prêt (mois)", min_value=1, value=12)

education = st.selectbox("Éducation", ["High School", "Bachelor's", "Master's", "PhD"])
employment_type = st.selectbox(
    "Type d'emploi", ["Full-time", "Part-time", "Unemployed"]
)
marital_status = st.selectbox(
    "État civil", ["Married", "Single", "Divorced", "Widowed"]
)
has_mortgage = st.selectbox("Possède un prêt immobilier ?", ["Yes", "No"])
has_dependents = st.selectbox("A des personnes à charge ?", ["Yes", "No"])
loan_purpose = st.selectbox("Objet du prêt", ["Auto", "Business", "Education", "Other"])
has_cosigner = st.selectbox("Co-signataire ?", ["Yes", "No"])


data = pd.DataFrame(
    {
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [700],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [3],
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [loan_amount / (income + 1e-5)],
        "Education": [education],
        "EmploymentType": [employment_type],
        "MaritalStatus": [marital_status],
        "HasMortgage": [has_mortgage],
        "HasDependents": [has_dependents],
        "LoanPurpose": [loan_purpose],
        "HasCoSigner": [has_cosigner],
        "Loan_to_Income": [loan_amount / (income + 1e-5)],
        "Employment_to_LoanTerm": [months_employed / (loan_term + 1e-5)],
    }
)

if st.button("Évaluer le dossier"):
    proba_default = pipeline.predict_proba(data)[:, 1][0]
    decision = "À vérifier" if proba_default > 0.1 else "Prêt à approuver"
    st.write(f"**Décision :** {decision}")
    st.write(f"**Probabilité de défaut estimée :** {proba_default:.2%}")

show_details = st.checkbox("Afficher les détails avancés du modèle ?")

if show_details:
    st.subheader("Données utilisées pour la prédiction")
    st.dataframe(data.T.rename(columns={0: "Valeur"}))

    st.subheader("Pipeline utilisé")
    st.text(str(pipeline))

    try:
        importances = pipeline.named_steps["classifier"].feature_importances_
        features = pipeline.named_steps["preprocessor"].transformers_[0][2] + list(
            pipeline.named_steps["preprocessor"]
            .transformers_[1][1]
            .get_feature_names_out()
        )
        st.subheader("Importance des features")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(features, importances)
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    except Exception as e:
        st.write(
            "Visualisation des importances des features non disponible pour ce pipeline."
        )
