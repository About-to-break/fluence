from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import logging
import torch

class TranslationEmptyError(Exception):
    pass


class MachineTranslator:
    def __init__(self,
                 model_name: str,
                 max_tokens: int,
                 target_language: str,
                 source_language: str,
                 ):
        try:
            logging.info(f"Loading model {model_name}")
            self.model_name = str(model_name)
            self.max_tokens = int(max_tokens)
            self.target_language = str(target_language)
            self.source_language = str(source_language)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float16)
            self.model.config.tie_word_embeddings = False
            self.tokenizer.src_lang = self.source_language
            logging.info(f"Loaded model {model_name}")
        except Exception as e:
            logging.error(e)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target_language),
            max_new_tokens=self.max_tokens
        )

        result = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        return result[0]

    def translate_batch(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target_language),
            max_new_tokens=self.max_tokens
        )

        results = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        return results
