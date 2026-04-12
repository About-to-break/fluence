from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import logging
import torch


class TranslationEmptyError(Exception):
    pass


class MachineTranslator:
    def __init__(
            self,
            model_name: str,
            max_length: int,
            max_new_tokens: int,
            target_language: str,
            source_language: str,
    ):
        self.model = None
        self.tokenizer = None

        self.max_new_tokens = int(max_new_tokens)
        self.max_length = int(max_length)
        self.target_language = str(target_language)
        self.source_language = str(source_language)

        try:
            logging.info(f"Loading model {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                src_lang=self.source_language
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            self.model.config.tie_word_embeddings = False

            token = f"__{self.target_language}__"
            self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(token)

            logging.info(f"Loaded model {model_name}")
            logging.info(f"Target language token: {token} → {self.forced_bos_token_id}")

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            raise TranslationEmptyError("Empty text for translation")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.forced_bos_token_id,
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            num_beams=5,
            early_stopping=True,
        )

        result = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )

        return result[0]

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.forced_bos_token_id,
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            num_beams=5,
            early_stopping=True,
        )

        results = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )

        return results
