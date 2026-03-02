# src/train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


# Charger le dataset
data_path = "data/Loan_default.csv"
df = pd.read_csv(data_path)


# Prétraitement
numeric_features = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
]
categorical_features = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

# Nouvelles features
df["Loan_to_Income"] = df["LoanAmount"] / (df["Income"] + 1e-5)
df["Employment_to_LoanTerm"] = df["MonthsEmployed"] / (df["LoanTerm"] + 1e-5)
numeric_features += ["Loan_to_Income", "Employment_to_LoanTerm"]

X = df[numeric_features + categorical_features]
y = df["Default"]


# Split train/test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Pipeline imblearn

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
        (
            "classifier",
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            ),
        ),
    ]
)


# Entraînement

pipeline.fit(X_train, y_train)
print("Modèle entraîné avec pipeline complet")

# Évaluation

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC-AUC :", roc_auc_score(y_test, y_prob))
print("Classification Report :\n", classification_report(y_test, y_pred))


# Sauvegarde

joblib.dump(pipeline, "models/pipeline_model.pkl")
print("Pipeline complet sauvegardé : models/pipeline_model.pkl")
