from typedefinitions import *
from mistralai import Mistral
from utils import generate_translation_prompt


class MistralExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str, model_slug: str):
        self.api_key = api_key
        self.model_slug = model_slug
        self.client = Mistral(api_key=self.api_key)

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

        response = self.client.chat.complete(
            model=self.model_slug,
            temperature=temperature,
            messages=messages,
        )

        return response.choices[0].message.content
