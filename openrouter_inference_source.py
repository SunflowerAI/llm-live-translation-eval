from typedefinitions import *
import requests
from utils import generate_translation_prompt

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenrouterExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str, model_slug: str, also_add=""):
        self.model_slug = model_slug
        self.api_key = api_key
        self.also_add = also_add

    def is_tl_valid(self, target_lang: TranslatableLanguage):
        return True

    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
        context: list[tuple[str, str]] | None = None,
    ) -> str:
        messages = generate_translation_prompt(
            source_lang, target_lang, text, context
        )
        if self.also_add:
            # Prepend control tokens (e.g. qwen "/no_think") to the current segment.
            messages[-1]["content"] = self.also_add + messages[-1]["content"]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_slug,
            "temperature": temperature,
            "max_tokens": (len(text) * 6) + 4000,
            "messages": messages,
        }

        # (connect, read) timeout so a hung/silent socket raises instead of
        # blocking a worker (and its concurrency slot) forever.
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=(15, 150)
        )
        if response.status_code != 200:
            raise Exception(f"OpenRouter error {response.status_code}: {response.text}")

        result = response.json()
        print(result)
        # Providers occasionally return 200 with null content; treat as empty so
        # the caller's length check handles it as a failure (not a None crash).
        content = result["choices"][0]["message"]["content"]
        return content if content is not None else ""


class OpenrouterGenericInference(AbstractGenericInference):
    def __init__(self, api_key: str, model_slug: str, also_add=None):
        self.api_key = api_key
        self.model_slug = model_slug
        self.append = also_add

    def infer(
        self,
        user_prompt: str,
        temperature: float,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        append_text = ""
        if self.append is not None:
            append_text = self.append + " "

        payload = {
            "model": self.model_slug,
            "temperature": temperature,
            "max_tokens": 9000,
            "messages": [
                {"role": "user", "content": append_text + user_prompt},
            ],
        }

        # (connect, read) timeout so a hung/silent socket raises instead of
        # blocking a worker (and its concurrency slot) forever.
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=(15, 150)
        )
        if response.status_code != 200:
            raise Exception(f"OpenRouter error {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]
