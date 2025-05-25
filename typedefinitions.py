from dataclasses import dataclass, asdict
import json
from enum import Enum
from abc import ABC, abstractmethod


class TranslatableLanguage(Enum):
    English = "English"
    German = "German"
    Chinese = "Chinese"
    Hungarian = "Hungarian"
    French = "French"
    Japanese = "Japanese"
    Italian = "Italian"
    EuropeanSpanish = "EuropeanSpanish"
    Ukrainian = "Ukrainian"
    Swedish = "Swedish"
    Korean = "Korean"
    Thai = "Thai"
    Welsh = "Welsh"
    Swahili = "Swahili"
    Esperanto = "Esperanto"

    def nuenki_code(self) -> str:
        if self == TranslatableLanguage.EuropeanSpanish:
            return "Spanish"

        return self.value

    def deepl_code(self) -> str:
        _code_map = {
            TranslatableLanguage.English: "EN",
            TranslatableLanguage.German: "DE",
            TranslatableLanguage.Chinese: "ZH",
            TranslatableLanguage.Hungarian: "HU",
            TranslatableLanguage.French: "FR",
            TranslatableLanguage.Japanese: "JA",
            TranslatableLanguage.Italian: "IT",
            TranslatableLanguage.EuropeanSpanish: "ES",
            TranslatableLanguage.Ukrainian: "UK",
            TranslatableLanguage.Swedish: "SV",
            TranslatableLanguage.Korean: "KO",
        }
        return _code_map[self]


class EnumSupportedEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


class ModelName(Enum):
    GPT_41 = "GPT 4.1"
    DeepL = "DeepL"
    GPT_4o = "GPT 4o"
    Grok_3_Beta = "Grok 3 Beta"
    Claude_Sonnet_35_2024_10_22 = "Claude Sonnet 3.5 2024-10-22"
    Claude_Sonnet_37_2025_02_19 = "Claude Sonnet 3.7 2025-02-19"
    GPT_41_Mini = "GPT 4.1 Mini"
    Gemini_25_Flash_Preview_04_17 = "Gemini 2.5 Flash Preview 04-17"
    Gemma3_27B = "Gemma 3 27b"
    Lingvanex = "Lingvanex"
    Llama_4_Scout = "LLama 4 Scout"
    Llama_4_Maverick = "LLama 4 Maverick"
    Llama33_70b = "Llama 3.3 70b"
    GPT_41_Nano = "GPT 4.1 Nano"
    Qwen_25_32B = "Qwen 2.5 32B"
    Llama_31_8b = "LLama 3.1 8b instant"
    Mistral_Small_31_24B = "Mistral Small 3.1 24B"
    Mistral_Saba_24B = "Mistral Saba 24B"
    Qwen3_30_a3b = "Qwen 3 30B A3B"
    Qwen3_14b = "Qwen 3 14B"
    Qwen3_32b = "Qwen 3 32B"
    Qwen3_235b_a22b = "Qwen 3 235B A22B"
    Nuenki_Hybrid = "Nuenki Hybrid"
    Deepseek_R1 = "Deepseek R1"
    Deepseek_V3 = "Deepseek V3"
    Deepseek_V3_NEW = "Deepseek V3 03-24"
    Mistral_Medium_3 = "Mistral Medium 3"
    GPT_4_Turbo = "GPT 4 Turbo"
    Claude_4_Sonnet = "Claude 4 Sonnet"
    Claude_4_Opus = "Claude 4 Opus"
    GPT_3_5_Turbo = "GPT 3.5 Turbo"
    Aya_Expanse_32B = "Aya Expanse 32B"


class ModelCompany(Enum):
    OpenAI = "OpenAI"
    DeepL = "DeepL"
    X_AI = "X AI"
    Anthropic = "Anthropic"
    Google = "Google"
    Lingvanex = "Lingvanex"
    Meta = "Meta"
    Alibaba = "Alibaba"
    Mistral = "Mistral"
    Nuenki = "Nuenki"
    Deepseek = "Deepseek"
    Cohere = "Cohere"

    def colour(self):
        match self:
            case ModelCompany.Google:
                return "#4285F4"  # blue
            case ModelCompany.Meta:
                return "#9b59b6"  # purple hue
            case ModelCompany.Mistral:
                return "#FF9500"  # vibrant orange
            case ModelCompany.OpenAI:
                return "#00C853"  # green
            case ModelCompany.Anthropic:
                return "#FFB300"  # amber
            case ModelCompany.X_AI:
                return "#808080"  # grey (Grok)
            case ModelCompany.Alibaba:
                return "#FF5722"  # deep orange
            case ModelCompany.Lingvanex:
                return "#607D8B"  # blue-grey
            case ModelCompany.DeepL:
                return "#2c3e50"  # deep dark blue
            case ModelCompany.Nuenki:
                return "#6bab90"  # brand colour
            case ModelCompany.Cohere:
                return "#f0dff3"
            case _:
                return "black"  # default color for unknown company


class InferenceCompany(Enum):
    Groq = "Groq"
    Openrouter = "Openrouter"
    Mistral = "Mistral"
    Anthropic = "Anthropic"
    DeepL = "DeepL"
    Lingvanex = "Lingvanex"
    Nuenki = "Nuenki"
    Cohere = "Cohere"


# I'm not usually one for inheritance, but it works so nicely here!
class AbstractExecutableTranslator(ABC):
    @abstractmethod
    def translate(
        self,
        source_lang: TranslatableLanguage,
        target_lang: TranslatableLanguage,
        text: str,
        temperature: float,
    ):
        """Translate"""
        raise NotImplementedError

    @abstractmethod
    def is_tl_valid(self, target_lang: TranslatableLanguage):
        pass


class AbstractGenericInference(ABC):
    @abstractmethod
    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        """Run a chat completion with a system and user prompt at the given temperature."""
        raise NotImplementedError


@dataclass
class TestedEntry:
    model_name: ModelName
    model_company: ModelCompany
    inference_service_name: InferenceCompany
    inference_source: AbstractExecutableTranslator
    temp: float | None
    thinking: str | None

    def unique_id(self):
        return (
            self.model_name.value
            + self.model_company.value
            + self.inference_service_name.value
            + str(self.temp)
            + str(self.thinking)
        )

    def dump_data(self):
        return {
            "model_name": self.model_name.value,
            "model_company": self.model_company.value,
            "inference_service_name": self.inference_service_name.value,
            "temp": self.temp,
        }


@dataclass
class ComparisonItem:
    # keys (ish!)
    language: TranslatableLanguage
    tested_entry_a: TestedEntry
    tested_entry_b: TestedEntry

    sentence: str
    sentence_category: str
    evaluating_model: str

    # items
    entry_a_translation: str
    entry_b_translation: str

    a_success: bool
    b_success: bool
    identical: bool
    evaluating_response: str


@dataclass
class RankItem:
    language: TranslatableLanguage
    tested_entry: TestedEntry

    sentence: str
    sentence_category: str
    evaluating_model: str

    translation: str

    score: int


@dataclass
class CoherenceIteration:
    sentence_cat: str
    sentence: str
    depth: int
    staged_text: str

    evaluations: list

    tested_entry: TestedEntry
