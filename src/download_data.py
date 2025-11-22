import os
import zipfile
import shutil
import subprocess


def install_kaggle_credentials(config_path="config/kaggle.json"):
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier {config_path} introuvable.")

    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(config_path, kaggle_json_path)
    os.chmod(kaggle_json_path, 0o600)

    print("Identifiants Kaggle installés !")


def download_dataset(
    dataset_name="nikhil1e9/loan-default", output_zip="loan-default.zip"
):
    print(f"Téléchargement du dataset : {dataset_name}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_name, "-p", "."],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Erreur lors du téléchargement :")
        print(result.stderr)
        raise RuntimeError("Échec du téléchargement Kaggle")

    print("Dataset téléchargé !")


def extract_dataset(zip_path="loan-default.zip", extract_to="data"):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Le fichier {zip_path} n'existe pas.")

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    print(f"Extraction terminée dans : {extract_to}")

    return os.listdir(extract_to)


def main():
    print("Démarrage du script de téléchargement\n")
    install_kaggle_credentials()
    download_dataset()
    extracted_files = extract_dataset()

    print("\n Fichiers extraits :")
    for f in extracted_files:
        print(" -", f)

    print("\nTéléchargement terminé !")


if __name__ == "__main__":
    main()
