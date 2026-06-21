from typedefinitions import *
from cohere import Client
from utils import generate_translation_prompt, messages_to_flat_prompt


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
        context: list[tuple[str, str]] | None = None,
    ) -> str:
        # Cohere's generate endpoint takes a single prompt, so flatten the chat
        # messages (system + preceding pairs + current segment) into a transcript.
        messages = generate_translation_prompt(
            source_lang, target_lang, text, context
        )
        prompt = messages_to_flat_prompt(messages)

        response = self.client.generate(
            model=self.model_slug,
            prompt=prompt,
            max_tokens=(len(text) * 6) + 4000,
            temperature=temperature,
        )

        return response.generations[0].text.strip()
