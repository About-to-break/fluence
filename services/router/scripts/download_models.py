import os
import sys
import urllib.request
import zipfile
import tempfile
import shutil
import time
from huggingface_hub import hf_hub_download

# ===== ПРЯМАЯ ССЫЛКА НА АРХИВ =====
RELEASE_URL = "https://github.com/MeLver0/sum/releases/download/rel4/adaptive_router_models_v1.1.0.zip"

# ===== HUGGING FACE KENLM =====
HF_REPO_ID = "BramVanroy/kenlm_wikipedia_en"
HF_FILENAME = "wiki_en_dep.arpa.bin"

# ===== ПУТИ =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTING_CORE = os.path.join(PROJECT_ROOT, 'internal', 'routing_core')
MODELS_DIR = os.path.join(ROUTING_CORE, 'models')
DATA_DIR = os.path.join(ROUTING_CORE, 'data')


def download_with_retry(url, dest_path, description, retries=3):
    """Скачивает файл с повторными попытками."""
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {description} (attempt {attempt})...")
            urllib.request.urlretrieve(url, dest_path)
            print(" Done!")
            return
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(5)
            else:
                raise


def download_kenlm():
    dest_path = os.path.join(DATA_DIR, 'kenlm_wiki_en.bin')

    if os.path.exists(dest_path):
        print(f"KenLM already exists: {dest_path}")
        return

    print("Downloading KenLM model from Hugging Face...")
    print(f"  Repo: {HF_REPO_ID}")
    print(f"  File: {HF_FILENAME}")

    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=tempfile.gettempdir(),
        )
        shutil.copy(model_path, dest_path)
        print(f"KenLM saved to: {dest_path}")
    except Exception as e:
        print(f"Failed to download KenLM: {e}")
        raise


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. KenLM с Hugging Face
    download_kenlm()

    # 2. Модель, конфиг и центроиды из Release
    xgb_path = os.path.join(MODELS_DIR, 'router_regressor_xgb.pkl')
    config_path = os.path.join(MODELS_DIR, 'router_config_xgb_regressor.json')
    centroids_path = os.path.join(DATA_DIR, 'labse_centroids.pkl')

    # Загружаем архив, если хотя бы одного файла нет
    if not (os.path.exists(xgb_path) and os.path.exists(config_path) and os.path.exists(centroids_path)):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "models.zip")

            try:
                download_with_retry(RELEASE_URL, archive_path, "model archive")
            except Exception as e:
                print(f"\nFailed to download: {e}")
                print(f"\nPlease download manually from:")
                print(f"  {RELEASE_URL}")
                sys.exit(1)

            print("\nExtracting...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                extract_dir = os.path.join(tmpdir, 'extracted')
                zip_ref.extractall(extract_dir)

                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        src = os.path.join(root, file)
                        if file == 'router_regressor_xgb.pkl':
                            shutil.copy2(src, xgb_path)
                            print(f"Regressor model: {xgb_path}")
                        elif file == 'router_config_xgb_regressor.json':
                            shutil.copy2(src, config_path)
                            print(f"Config: {config_path}")
                        elif file == 'labse_centroids.pkl':
                            shutil.copy2(src, centroids_path)
                            print(f"Centroids: {centroids_path}")

        # Дополнительная проверка, что центроиды действительно появились
        if not os.path.exists(centroids_path):
            print("ERROR: labse_centroids.pkl was not extracted from the archive!")
            sys.exit(1)
    else:
        print("All model files already exist.")


if __name__ == "__main__":
    main()