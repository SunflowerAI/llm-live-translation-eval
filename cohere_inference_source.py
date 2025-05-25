from typedefinitions import *
from cohere import Client
from utils import generate_translation_prompt


class CohereExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str, model_slug: str):
        self.api_key = api_key
        self.model_slug = model_slug
        self.client = Client(api_key=self.api_key)

    def is_tl_valid(self, target_lang: TranslatableLanguage):
        return True

    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
    ) -> str:
        prompt = generate_translation_prompt(source_lang, target_lang, text)

        response = self.client.generate(
            model=self.model_slug,
            prompt=prompt,
            max_tokens=(len(text) * 6) + 4000,
            temperature=temperature,
        )

        return response.generations[0].text.strip()
