# Credit Risk Prediction

[**Voir l'application en ligne**](https://credit-risk-prediction-lk.streamlit.app/)

## 🚀 Présentation

Ce projet est une application Streamlit qui permet d’évaluer le **risque de défaut de paiement d’un prêt** à partir des informations d’un dossier client.  
L’application utilise un **pipeline Machine Learning complet** pour fournir une prédiction et montrer ce qui se cache derrière la décision.

Le modèle est basé sur un **XGBoost Classifier**, avec :

- Prétraitement des données (StandardScaler + OneHotEncoder)
- Gestion du déséquilibre des classes avec **SMOTE**
- Création de features dérivées comme `DTIRatio` et `Employment_to_LoanTerm`

---

## 🎯 Fonctionnalités principales

- **Prédiction du risque de défaut** pour un dossier client
- **Interface simple et intuitive** pour saisir les informations du dossier
- **Affichage transparent du modèle** (optionnel) :
  - Pipeline complet
  - Features utilisées et valeurs correspondantes
  - Importances des features (XGBoost)

### Features calculées

- **DTIRatio (Debt-to-Income Ratio)** : ratio du montant du prêt sur le revenu annuel
- **Employment_to_LoanTerm** : ratio de la durée d’emploi sur la durée du prêt, indicateur de stabilité financière

---

## 🛠️ Installation et usage local

1. Cloner le dépôt :

```bash
git clone https://github.com/kwwl/credit-risk-prediction.git
cd credit-risk-prediction
Créer et activer l'environnement virtuel :

bash
Copier le code
python -m venv crp_env
source crp_env/bin/activate  # macOS/Linux
# crp_env\Scripts\activate   # Windows
Installer les dépendances :

bash
Copier le code
pip install -r requirements.txt
Lancer l’application Streamlit :

bash
Copier le code
streamlit run app.py
📊 Structure du projet
powershell
Copier le code
credit-risk-prediction/
├─ app.py                  # Interface Streamlit
├─ models/                 # Modèles sauvegardés (.pkl)
├─ data/                   # Datasets CSV
├─ src/                    # Scripts de préprocessing et d'entraînement
├─ requirements.txt        # Dépendances Python
└─ README.md
🌐 Déploiement
L’application est déployée sur Streamlit Cloud :
https://credit-risk-prediction-lk.streamlit.app/

Le modèle est téléchargé automatiquement depuis Google Drive pour garantir que l’application fonctionne sans fichier local.

💡 A propos
Ce projet a été développé pour démontrer :

L’utilisation de pipelines ML avec preprocessing et oversampling

La création de features financières dérivées

La transparence dans les décisions d’une IA pour le recruteur ou le client

yaml
Copier le code

---

Si tu veux, je peux te faire une **version encore plus “recruteur-friendly”** avec des **captures d’écran Streamlit**, un **schéma du pipeline**, et une section “Pourquoi ce projet montre que je sais faire du ML”.

Veux‑tu que je fasse ça ?
```
