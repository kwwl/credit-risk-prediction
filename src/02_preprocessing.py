import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1️⃣ Chemin direct vers le dataset Kaggle
csv_file = "/Users/kemillamouri/Desktop/Data IA HETIC 2/Cours Python/credit-risk-prediction/data/Loan_default.csv"
df = pd.read_csv(csv_file)
print("✔ Dataset original chargé :", df.shape)

# 2️⃣ Colonnes catégorielles
categorical_cols = df.select_dtypes(include=["object"]).columns

# LabelEncoding safe pour RandomForest
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
print("✅ Encodage LabelEncoding terminé")

# 3️⃣ Supprimer doublons
df = df.drop_duplicates()
print("✅ Doublons supprimés")

# 4️⃣ Colonnes numériques (exclure la cible 'Default')
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("Default")
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("✅ Standardisation terminée")

# 5️⃣ Sauvegarder le dataset préparé
output_path = "/Users/kemillamouri/Desktop/Data IA HETIC 2/Cours Python/credit-risk-prediction/data/df_encoded.csv"
df.to_csv(output_path, index=False)
print(f"✅ Dataset encodé sauvegardé dans {output_path}")
