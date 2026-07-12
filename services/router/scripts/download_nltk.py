import requests
import zipfile
import os
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

NLTK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nltk_data')

resources = [
    ('tokenizers/punkt.zip', 'tokenizers/punkt'),
    ('tokenizers/punkt_tab.zip', 'tokenizers/punkt_tab'),
    ('corpora/stopwords.zip', 'corpora/stopwords'),
    ('corpora/words.zip', 'corpora/words'),
    ('taggers/averaged_perceptron_tagger_eng.zip', 'taggers/averaged_perceptron_tagger_eng'),
    ('taggers/universal_tagset.zip', 'taggers/universal_tagset'),
]

# Два источника: оригинальный и зеркало через Cloudflare
BASE_URLS = [
    'https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/'
]

for url_path, extract_subdir in resources:
    dest_dir = os.path.join(NLTK_DIR, extract_subdir)
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = '/tmp/nltk_temp.zip'

    downloaded = False

    # Пробуем все зеркала
    for base_url in BASE_URLS:
        url = base_url + url_path
        for attempt in range(3):  # по 3 попытки на каждое зеркало
            try:
                print(f'Downloading {url} (attempt {attempt + 1})...')
                r = requests.get(url, verify=False, timeout=30)
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    f.write(r.content)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(dest_dir)
                os.remove(zip_path)
                print('  OK')
                downloaded = True
                break  # Если скачалось, прерываем цикл попыток
            except Exception as e:
                print(f'  Attempt {attempt + 1} failed: {e}')
                if attempt < 2:
                    time.sleep(3)

        if downloaded:
            break  # Если скачалось, прерываем цикл зеркал

    if not downloaded:
        print(f'\nFATAL ERROR: Failed to download {url_path} from all sources!')
        raise SystemExit(1)

print('NLTK data downloaded successfully')