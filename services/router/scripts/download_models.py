import os
import sys
import urllib.request
import zipfile
import tempfile
import shutil
from huggingface_hub import hf_hub_download

# ===== ПРЯМАЯ ССЫЛКА НА АРХИВ =====
RELEASE_URL = "https://github.com/MeLver0/sum/releases/download/Release/adaptive_router_models_v1.0.0.zip"

# ===== HUGGING FACE KENLM =====
HF_REPO_ID = "BramVanroy/kenlm_wikipedia_en"
HF_FILENAME = "wiki_en_dep.arpa.bin"

# ===== ПУТИ =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTING_CORE = os.path.join(PROJECT_ROOT, 'internal', 'routing_core')
MODELS_DIR = os.path.join(ROUTING_CORE, 'models')
DATA_DIR = os.path.join(ROUTING_CORE, 'data')


def download_with_progress(url, dest_path, description):
    print(f"Downloading {description}...")
    print(f"  URL: {url}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print("\n Done!")


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
            cache_dir=os.path.join(DATA_DIR, 'hf_cache')
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

    # 2. XGBoost и конфиг из твоего Release
    xgb_path = os.path.join(MODELS_DIR, 'router_classifier_xgb.joblib')
    config_path = os.path.join(MODELS_DIR, 'router_config_xgb.json')

    if os.path.exists(xgb_path) and os.path.exists(config_path):
        print("XGBoost model and config already exist")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "models.zip")

            try:
                download_with_progress(RELEASE_URL, archive_path, "model archive")
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
                        if file == 'router_classifier_xgb.joblib':
                            src = os.path.join(root, file)
                            shutil.copy2(src, xgb_path)
                            print(f"XGBoost model: {xgb_path}")
                        elif file == 'router_config_xgb.json':
                            src = os.path.join(root, file)
                            shutil.copy2(src, config_path)
                            print(f"Config: {config_path}")




if __name__ == "__main__":
    main()