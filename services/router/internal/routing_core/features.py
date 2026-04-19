"""
Feature extraction for adaptive router.
"""

import re
import math
import os
import logging
import kenlm
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract linguistic features from English text."""

    def __init__(self, kenlm_path: str):
        if not os.path.exists(kenlm_path):
            raise FileNotFoundError(f"KenLM model not found: {kenlm_path}")

        logger.info(f"Loading KenLM model...")
        self.kenlm = kenlm.Model(kenlm_path)

        # Download NLTK data if needed
        try:
            self.stop_words = set(stopwords.words('english'))
            self.english_vocab = set(words.words())
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.english_vocab = set(words.words())

        logger.info("FeatureExtractor initialized")

    def extract(self, text: str) -> dict:
        """Extract all 7 features from text."""
        features = {}

        # Length features
        features['source_length_chars'] = len(text)

        tokens = word_tokenize(text)
        features['source_length_words'] = len(tokens)

        alpha_tokens = [t for t in tokens if t.isalpha()]
        if alpha_tokens:
            features['avg_word_length'] = sum(len(t) for t in alpha_tokens) / len(alpha_tokens)
        else:
            features['avg_word_length'] = 0

        # Type-token ratio
        unique_tokens = set(t.lower() for t in tokens if t.isalpha())
        features['type_token_ratio'] = len(unique_tokens) / len(tokens) if tokens else 0

        # Statistical features (KenLM)
        kenlm_text = self._tokenize_for_kenlm(text)
        perplexity = self.kenlm.perplexity(kenlm_text)
        features['perplexity'] = perplexity
        features['log_perplexity'] = math.log(perplexity + 1)

        word_count = len(kenlm_text.split())
        features['perplexity_per_word'] = perplexity / word_count if word_count else 0

        return features

    def _tokenize_for_kenlm(self, text: str) -> str:
        """Tokenize text for KenLM model."""
        text = text.lower()
        text = re.sub(r'([.,!?;:()\[\]{}"\'`])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text