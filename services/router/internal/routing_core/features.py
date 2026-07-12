"""
Feature extraction for adaptive router (updated with LaBSE centroids).
"""

import re
import logging
import os
import pickle
import torch
import kenlm
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# Глобальные объекты для переиспользования
_labse_model = None
_centroids = None
_kenlm_model = None

SLANG_DICT = {
    'gonna','wanna','gotta','kinda','sorta','dunno','ya','nah','yep','nope',
    'bro','dude','mate','fella','chick','buddy','pal','homie','guv','lad',
    'cool','awesome','dope','lit','sick','rad','groovy','lame','bogus','meh',
    'hangover','booze','crash','bail','cram','cringe','ghost','salty','shook',
    'flex','clapback','vibe','chill','extra','slay','stan','simp','yeet',
    'sus','bruh','lowkey','highkey','noob','pwn','af','asf','tbh','imo',
    'idk','omg','lol','lmao','rofl','brb','btw','fyi','gg','ez'
}

AWL = {
    'analysis','approach','assessment','assume','authority','available',
    'benefit','concept','consistent','constitutional','context','contract',
    'create','data','definition','derived','distribution','economic',
    'environment','established','estimate','evidence','export','factors',
    'financial','formula','function','identified','income','indicate',
    'individual','interpretation','involved','issues','labour','legal',
    'legislation','major','method','occur','percent','period','policy',
    'principle','procedure','process','required','research','response',
    'role','section','sector','significant','similar','source','specific',
    'structure','theory','variable'
}

INTERJ = {
    'ah','alas','aha','bah','blah','boo','bravo','cheers','darn','dear',
    'duh','eh','ew','gosh','ha','hah','hallelujah','hey','hmm','huh',
    'hurrah','jeez','mhm','nah','oh','ooh','oops','ouch','phew','psst',
    'shh','ugh','uh','wahoo','well','whoa','wow','yahoo','yay','yikes',
    'yippee','yuck'
}


def _init_models(data_dir: str):
    """Инициализация моделей (вызывается один раз при старте)."""
    global _labse_model, _centroids, _kenlm_model

    if _labse_model is None:
        logger.info("Loading LaBSE model...")
        _labse_model = SentenceTransformer('sentence-transformers/LaBSE')

    if _centroids is None:
        centroids_path = os.path.join(data_dir, 'labse_centroids.pkl')
        if not os.path.exists(centroids_path):
            raise FileNotFoundError(f"Centroids file not found: {centroids_path}")
        logger.info("Loading centroids...")
        with open(centroids_path, 'rb') as f:
            _centroids = pickle.load(f)

    if _kenlm_model is None:
        kenlm_path = os.path.join(data_dir, 'kenlm_wiki_en.bin')
        if not os.path.exists(kenlm_path):
            raise FileNotFoundError(f"KenLM model not found: {kenlm_path}")
        logger.info("Loading KenLM model...")
        _kenlm_model = kenlm.Model(kenlm_path)

    logger.info("All models initialized")


def extract_features(text: str) -> dict:
    """
    Извлечь все 36 признаков из исходного текста.
    Возвращает словарь {имя_признака: значение}.
    """
    feat = {}

    # === Базовые лингвистические ===
    feat['src_len_chars'] = len(text)
    tokens = word_tokenize(text)
    feat['src_len_words'] = len(tokens)
    if tokens:
        feat['src_avg_word_len'] = sum(len(t) for t in tokens) / len(tokens)
    else:
        feat['src_avg_word_len'] = 0

    feat['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    feat['punct_noise'] = sum(1 for c in text if c in '!?') / max(len(text), 1)

    # TTR
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]
    feat['TTR'] = len(set(alpha_tokens)) / max(len(alpha_tokens), 1)

    # === KenLM перплексия ===
    kenlm_text = text.lower()
    kenlm_text = re.sub(r'([.,!?;:()\[\]{}"\'`])', r' \1 ', kenlm_text)
    kenlm_text = re.sub(r'\s+', ' ', kenlm_text).strip()
    logp = _kenlm_model.score(kenlm_text, bos=True, eos=True)
    wc = len(kenlm_text.split()) + 1
    if wc and logp != -float('inf'):
        feat['kenlm_perplexity'] = 10 ** (-logp / wc)
    else:
        feat['kenlm_perplexity'] = 1000.0

    # === LaBSE сходства с центрами ===
    src_emb = _labse_model.encode([text], convert_to_tensor=True, show_progress_bar=False)
    for domain, centroid in _centroids.items():
        c_tensor = torch.from_numpy(centroid).unsqueeze(0).to(src_emb.device)
        c_tensor = torch.nn.functional.normalize(c_tensor, p=2, dim=1)
        sim = util.cos_sim(src_emb, c_tensor).squeeze().cpu().numpy().item()
        feat[f'sim_{domain}'] = sim

    # === Кастомные индикаторы ===
    words_lower = [t.lower() for t in tokens if t.isalpha()]
    n_words = max(len(words_lower), 1)

    feat['slang_ratio'] = sum(1 for w in words_lower if w in SLANG_DICT) / n_words
    feat['formality_ratio'] = sum(1 for w in words_lower if w in AWL) / n_words
    feat['interjection_ratio'] = sum(1 for w in words_lower if w in INTERJ) / n_words

    # Рифма
    sentences = sent_tokenize(text)
    if len(sentences) >= 2:
        endings = []
        for s in sentences:
            s_tokens = word_tokenize(s)
            if len(s_tokens) >= 2 and len(s_tokens[-1]) >= 3:
                endings.append(s_tokens[-1][-3:].lower())
        if len(endings) >= 2:
            matches = sum(1 for i in range(len(endings)-1) if endings[i] == endings[i+1])
            feat['rhyme_score'] = matches / (len(endings) - 1)
        else:
            feat['rhyme_score'] = 0.0
    else:
        feat['rhyme_score'] = 0.0

    # === Читабельность (textstat) ===
    try:
        import textstat
        feat['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        feat['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        feat['smog_index'] = textstat.smog_index(text)
        feat['gunning_fog'] = textstat.gunning_fog(text)
        feat['automated_readability_index'] = textstat.automated_readability_index(text)
        feat['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
    except ImportError:
        for metric in ['flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index',
                       'gunning_fog', 'automated_readability_index', 'dale_chall_readability_score']:
            feat[metric] = 0.0

    # === POS-признаки (NLTK) ===
    try:
        pos_tags = nltk.pos_tag(tokens, tagset='universal')
        counts = {'nouns': 0, 'verbs': 0, 'adjs': 0, 'advs': 0, 'pronouns': 0, 'preps': 0}
        for _, tag in pos_tags:
            if tag == 'NOUN': counts['nouns'] += 1
            elif tag == 'VERB': counts['verbs'] += 1
            elif tag == 'ADJ': counts['adjs'] += 1
            elif tag == 'ADV': counts['advs'] += 1
            elif tag == 'PRON': counts['pronouns'] += 1
            elif tag == 'ADP': counts['preps'] += 1
        total = len(tokens)
        for key in counts:
            feat[key] = counts[key] / total if total > 0 else 0.0
    except Exception:
        for key in ['nouns', 'verbs', 'adjs', 'advs', 'pronouns', 'preps']:
            feat[key] = 0.0

    return feat