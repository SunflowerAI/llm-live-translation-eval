from typedefinitions import *
import anthropic
from utils import generate_translation_prompt, split_system_messages


class AnthropicExecutableTranslator(AbstractExecutableTranslator):
    def __init__(self, api_key: str, model_slug: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_slug = model_slug

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
        system_prompt, conversation = split_system_messages(messages)

        response = self.client.messages.create(
            model=self.model_slug,
            temperature=temperature,
            max_tokens=1024,
            system=system_prompt,
            messages=conversation,
        )

        return response.content[0].text
