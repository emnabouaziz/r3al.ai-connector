"""System prompt of every training example.

DECISION: the system prompt is written in the SAME LANGUAGE as the example
(mode "example_language", the default). A Dutch question gets a Dutch
system prompt that also says "answer in Dutch", so the model learns to
answer in the customer's language instead of drifting to English. Mode
"english" (one English prompt for all examples) is available for clients
whose serving stack injects a single fixed system prompt.

The insurer's name always comes from config (build.insurer_name); it is
never hardcoded. Templates can be overridden per language in config.
"""

from dataclasses import dataclass, field
from typing import Any

DEFAULT_TEMPLATES: dict[str, str] = {
    "nl": ("Je bent de klantenservice-assistent van {insurer}. Antwoord in het Nederlands, beleefd en precies. "
           "Verzin geen informatie die je niet hebt."),
    "fr": ("Vous êtes l'assistant du service client de {insurer}. Répondez en français, poliment et avec "
           "précision. N'inventez aucune information."),
    "en": ("You are the support assistant of {insurer}. Answer in English, politely and precisely. "
           "Do not make up information you do not have."),
}
MODES = ("example_language", "english")


class SystemPromptError(ValueError):
    """Invalid system prompt configuration."""


@dataclass(frozen=True)
class SystemPrompts:
    insurer: str
    mode: str = "example_language"
    templates: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TEMPLATES))

    def for_language(self, language: str) -> str:
        lang = language if self.mode == "example_language" and language in self.templates else "en"
        return self.templates[lang].format(insurer=self.insurer)


def from_config(build: dict[str, Any]) -> SystemPrompts:
    insurer = build.get("insurer_name")
    if not isinstance(insurer, str) or not insurer.strip():
        raise SystemPromptError("build.insurer_name is required")
    mode = str(build.get("system_prompt_language", "example_language"))
    if mode not in MODES:
        raise SystemPromptError(f"build.system_prompt_language must be one of {MODES}")
    templates = {**DEFAULT_TEMPLATES, **dict(build.get("system_prompt_templates") or {})}
    for lang, template in templates.items():
        if "{insurer}" not in template:
            raise SystemPromptError(f"system prompt template '{lang}' must contain {{insurer}}")
    return SystemPrompts(insurer=insurer.strip(), mode=mode, templates=templates)
