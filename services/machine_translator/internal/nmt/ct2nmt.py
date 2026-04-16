import ctranslate2
import transformers
import asyncio
import time
import os
import logging
from typing import List, Tuple, Optional
from huggingface_hub import snapshot_download

# Настройка логгера для этого модуля
logger = logging.getLogger(__name__)


class TranslationEmptyError(Exception):
    pass


def download_model(hf_model_repo: str, target_dir: str, token: str):
    logger.info(f"Downloading model from {hf_model_repo} to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(repo_id=hf_model_repo, local_dir=target_dir, token=token)
    logger.info(f"Model downloaded successfully")


class CT2Translator:
    _instance = None

    def __init__(
            self,
            model_path: str,
            tgt_lang: str,
            src_lang: str,
            device: str = "cpu",
            compute_type: str = "int8",
            intra_threads: int = 8,
            inter_threads: int = 1,
    ):
        self.model_path = model_path
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang

        logger.info(f"Initializing CT2Translator with model={model_path}")
        logger.info(f"  device={device}, compute_type={compute_type}")
        logger.info(f"  intra_threads={intra_threads}, inter_threads={inter_threads}")
        logger.info(f"  src_lang={src_lang}, tgt_lang={tgt_lang}")

        load_start = time.time()
        self.translator = ctranslate2.Translator(
            model_path,
            device=device,
            compute_type=compute_type,
            intra_threads=int(intra_threads),
            inter_threads=int(inter_threads),
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, fix_mistral_regex=True
        )
        logger.info(f"Model loaded in {time.time() - load_start:.2f}s")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def translate(self, text_to_translate: str, verbose: bool = False) -> Tuple[str, float]:
        """For testing purposes only."""
        start_total = time.time()

        source_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(f"{self.src_lang} {text_to_translate}")
        )

        results = self.translator.translate_batch(
            [source_tokens], target_prefix=[[self.tgt_lang]]
        )

        target_tokens = results[0].hypotheses[0]
        if target_tokens[0] == self.tgt_lang:
            target_tokens = target_tokens[1:]

        translated = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(target_tokens)
        )

        elapsed = time.time() - start_total

        if verbose:
            logger.debug(
                f"Single translation: '{text_to_translate[:30]}...' -> '{translated[:30]}...' ({elapsed:.3f}s)")

        return translated, elapsed

    def translate_batch(self, texts: List[str]) -> Tuple[List[str], float]:
        """Use this for batch translation."""
        if not texts:
            return [], 0.0

        logger.info(f"🔄 Translating batch of {len(texts)} texts...")
        start_total = time.time()

        source_batch = [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(f"{self.src_lang} {text}")
            )
            for text in texts
        ]

        target_prefixes = [[self.tgt_lang] for _ in texts]

        results = self.translator.translate_batch(
            source_batch, target_prefix=target_prefixes
        )

        translations = []
        for result in results:
            tokens = result.hypotheses[0]
            if tokens[0] == self.tgt_lang:
                tokens = tokens[1:]
            translations.append(
                self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens))
            )

        elapsed = time.time() - start_total
        throughput = len(texts) / elapsed if elapsed > 0 else 0

        logger.info(
            f"✅ Batch completed: {len(texts)} texts in {elapsed:.3f}s "
            f"({throughput:.1f} texts/sec, {elapsed / len(texts) * 1000:.1f} ms/text)"
        )

        return translations, elapsed


class Batcher:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, translator: CT2Translator, max_wait: float = 0.05):
        # Предотвращаем повторную инициализацию синглтона
        if hasattr(self, '_initialized'):
            return

        self.translator = translator
        self.max_wait = max_wait
        self.queue: List[str] = []
        self.futures: List[asyncio.Future] = []
        self.lock = asyncio.Lock()
        self.task: Optional[asyncio.Task] = None

        self.total_batches = 0
        self.total_texts = 0
        self.total_wait_time = 0.0

        self.instance_id = id(self)
        self._initialized = True

        logger.info(f"Batcher initialized: max_wait={max_wait * 1000:.0f}ms, id={self.instance_id}")

    async def translate(self, text: str) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async with self.lock:
            queue_was_empty = len(self.queue) == 0
            self.queue.append(text)
            self.futures.append(future)

            logger.info(
                f"📥 Text added to queue. Queue size: {len(self.queue)}, task exists: {self.task is not None}, batcher id: {self.instance_id}")

            if queue_was_empty:
                logger.info(f"📥 First text in batch, starting timer... (batcher id: {self.instance_id})")

            if self.task is None:
                self.task = asyncio.create_task(self._process())
                logger.info(f"🚌 Batch processor started (batcher id: {self.instance_id})")
            else:
                logger.info(f"⏳ Batch processor already running, waiting... (batcher id: {self.instance_id})")

        return await future

    async def _process(self):
        wait_start = time.time()
        await asyncio.sleep(self.max_wait)
        wait_time = time.time() - wait_start

        # Собираем накопленные тексты
        async with self.lock:
            texts = self.queue.copy()
            futures = self.futures.copy()
            batch_size = len(texts)

            self.queue.clear()
            self.futures.clear()
            # НЕ сбрасываем self.task здесь!

            logger.info(f"🔍 _process: collected {batch_size} texts from queue (batcher id: {self.instance_id})")

        if texts:
            self.total_batches += 1
            self.total_texts += batch_size
            self.total_wait_time += wait_time

            logger.info(
                f"🚌 Batch #{self.total_batches}: {batch_size} texts collected "
                f"(waited {wait_time * 1000:.1f}ms, batcher id: {self.instance_id})"
            )

            loop = asyncio.get_running_loop()
            results, _ = await loop.run_in_executor(
                None, self.translator.translate_batch, texts
            )

            for future, result in zip(futures, results):
                future.set_result(result)
        else:
            logger.info(f"⏳ _process timed out with empty queue (batcher id: {self.instance_id})")

        # ТОЛЬКО ПОСЛЕ ЗАВЕРШЕНИЯ ПЕРЕВОДА проверяем очередь и сбрасываем task
        async with self.lock:
            if self.queue:
                # За время перевода накопились новые тексты — запускаем новый батч
                logger.info(
                    f"🔄 {len(self.queue)} texts arrived during translation, starting new batch (batcher id: {self.instance_id})")
                self.task = asyncio.create_task(self._process())
            else:
                # Очередь пуста — сбрасываем task
                self.task = None
                logger.info(f"✅ Batch processor finished, queue empty (batcher id: {self.instance_id})")



# Глобальный синглтон батчера
_batcher: Optional[Batcher] = None


def get_batcher(translator: CT2Translator) -> Batcher:
    global _batcher
    if _batcher is None:
        logger.info("Creating global batcher instance")
        _batcher = Batcher(translator)
    return _batcher