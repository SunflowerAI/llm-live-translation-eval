import hashlib
from random import *
import time
import threading
from contextlib import contextmanager
from typedefinitions import *
from latency_logger import log_latency

# Locale the production interpreter prompt is written for (e.g. "Australian").
SERMON_LOCALE = "Australian"

# Single knob for total concurrent inference (network) calls across the whole
# run. The live pipeline fans out languages × sermons × judges freely; every
# actual API call passes through `inference_slot()`, so this bounds provider
# load regardless of how deep the fan-out goes. Tune via set_inference_concurrency.
_inference_semaphore = threading.BoundedSemaphore(16)


def set_inference_concurrency(limit):
    """Set the global cap on concurrent inference calls."""
    global _inference_semaphore
    _inference_semaphore = threading.BoundedSemaphore(limit)


@contextmanager
def inference_slot():
    """Acquire one slot of the global inference-concurrency budget.

    Reads the semaphore at call time (not import time) so set_inference_concurrency
    takes effect even after callers have imported this module.
    """
    semaphore = _inference_semaphore
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def generate_translation_prompt(
    from_lang,
    to_lang,
    text: str,
    context: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Build a chat-style message list for live, segment-by-segment translation.

    This adapts the harness from document translation to live translation: rather
    than a single self-contained prompt, the model is handed recent conversational
    context so it can stay consistent across an ongoing stream of speech.

    Returns a list of OpenAI-style ``{"role", "content"}`` messages:

    (a) a **system prompt** carrying the translation instructions;
    (b) up to **five user/assistant pairs** of the immediately preceding source
        segments and their translations (``context``, oldest first); and
    (c) the **current source segment as the final user message**, which the model
        answers with its translation.

    ``context`` is a list of ``(source_segment, translation)`` tuples for the
    preceding segments, oldest first; only the last five are used. Pass ``None``
    (the default) for document mode or the first segment of a stream, yielding just
    the system prompt and the current segment.
    """
    # Callers pass TranslatableLanguage enums; use the human language name rather
    # than the enum repr ("German", not "TranslatableLanguage.German").
    from_lang = getattr(from_lang, "value", from_lang)
    to_lang = getattr(to_lang, "value", to_lang)

    # The production interpreter system prompt, verbatim, so the benchmark
    # measures the prompt actually shipped. The streaming context is supplied by
    # the user/assistant pairs below rather than spelled out here.
    system_prompt = (
        f"You are a simultaneous interpreter for {SERMON_LOCALE} church sermons. "
        f"Your task is to translate the input from {from_lang} to {to_lang}, "
        f"returning only the translated text. Ensure that the output is exclusively "
        f"in {to_lang}. Be concise, while ensuring that the translation flows smoothly."
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for source_segment, translation in (context or [])[-5:]:
        messages.append({"role": "user", "content": source_segment})
        messages.append({"role": "assistant", "content": translation})

    messages.append({"role": "user", "content": text})
    return messages


def split_system_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split a canonical message list into ``(system_prompt, conversation)``.

    For providers (e.g. Anthropic) that take the system prompt as a separate
    argument rather than as a message with ``role == "system"``.
    """
    system_prompt = ""
    conversation = []
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
        else:
            conversation.append(message)
    return system_prompt, conversation


def messages_to_genai_contents(messages: list[dict]) -> list[dict]:
    """Convert a canonical message list into google-genai ``contents``.

    google-genai uses ``"model"`` for the assistant role and carries the system
    prompt out of band (via ``system_instruction``), so system messages are
    dropped here.
    """
    contents = []
    for message in messages:
        if message["role"] == "system":
            continue
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    return contents


def messages_to_flat_prompt(messages: list[dict]) -> str:
    """Flatten a canonical message list into a single prompt string.

    For completion-style APIs (e.g. Cohere ``generate``) that take one prompt
    rather than a list of chat turns. Preserves the preceding pairs as a labelled
    transcript and cues the model to emit the final translation.
    """
    parts = []
    for message in messages:
        role = message["role"]
        if role == "system":
            parts.append(message["content"])
        elif role == "user":
            parts.append(f"Source: {message['content']}")
        elif role == "assistant":
            parts.append(f"Translation: {message['content']}")
    parts.append("Translation:")
    return "\n".join(parts)


def md5hash(text):
    return hashlib.md5(bytes(text, "utf-8")).hexdigest()


def get_translation_with_cache_check(
    model, source, target, category, sentence, cache, context=None
):
    # Live-translation context makes the output depend on the preceding segments
    # and their translations, so it must enter the cache key. Only append it when
    # present, so document-mode (context=None) keys stay identical to before and
    # the committed cache.db remains valid.
    context_suffix = f"|Context md5:{md5hash(repr(context))}" if context else ""
    cache_key = f"TRANSLATION From Language:{source.value} To Language: {target.value}|Model: {model.unique_id()}|Sentence md5:{md5hash(sentence)}{context_suffix}"
    check = cache.get(cache_key)
    if check:
        if len(check) < 3 or "483" in check:
            return None
        else:
            return check

    start_t = None
    end_t = None
    translation = None

    # Bounded retries: a persistent failure on one segment must not hang the
    # whole run (chains are awaited as a barrier before judging).
    max_attempts = 5
    i = 0
    while i < max_attempts:
        print("Iteration of translate loop")
        i += 1
        try:
            start_t = time.time()
            with inference_slot():
                translation = model.inference_source.translate(
                    source, target, sentence, model.temp, context
                )
            # Treat a missing/empty/refused translation as a failure for this
            # segment rather than retrying forever (e.g. providers occasionally
            # return 200 with null content).
            if translation is None or len(translation) < 3 or "483" in translation:
                print("INCORRECT TRANSLATION LENGTH", translation, "ON", cache_key)
                cache.set(cache_key, translation or "")
                return None

            end_t = time.time()
            break
        except Exception as e:
            print("Exception during inference", e, model)
            time.sleep(i * choice(range(3, 12)))
            continue
    else:
        print("Giving up after", max_attempts, "attempts on", cache_key)
        return None

    cache.set(cache_key, translation)
    log_latency(model.unique_id(), category, sentence, end_t - start_t)
    return translation
