"""The Claude study helper: explain, answer questions, summarize, quiz, narrate, describe.

Uses the official ``anthropic`` SDK (optional dependency: ``pip install "athena-reader[ai]"``).
Nothing is sent anywhere unless the user has turned the helper on in Settings and
added an API key. Every answer is plain text meant to be read aloud.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, model_validator

from ..config import DEFAULT_AI_MODEL, Settings
from ..models import Block
from . import prompts
from .prompts import clean_inline, to_speakable

logger = logging.getLogger(__name__)

MAX_TOKENS = 16_000
CLIENT_TIMEOUT_S = 120.0
CLIENT_MAX_RETRIES = 2
# Server-side refusal fallback: if a safety classifier declines, the API re-runs
# the request on the model Anthropic recommends for that refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_HISTORY_TURNS = 10  # question/answer pairs of chat history kept for ask()
MAX_QUIZ_QUESTIONS = 20

Effort = Literal["low", "medium", "high"]

OFF_MESSAGE = (
    "The study helper is turned off. You can turn it on in Settings. "
    "It sends the text you ask about to Anthropic (Claude) to answer."
)
NO_KEY_MESSAGE = "Add your Anthropic API key in Settings to use the study helper."
NOT_INSTALLED_MESSAGE = (
    "The study helper needs the Anthropic package, which isn't installed. "
    'Install it with: pip install "athena-reader[ai]"'
)
REFUSAL_MESSAGE = "Claude declined to answer that one. Try rephrasing."
MAX_TOKENS_NOTE = (
    "I had to stop there because the answer ran long. Ask for a shorter version if you need the rest."
)
OUT_OF_ROOM_MESSAGE = "Claude ran out of room before answering. Try a shorter passage or question."
EMPTY_ANSWER_MESSAGE = "Claude didn't return an answer. Try again."
QUIZ_FORMAT_MESSAGE = "Claude's quiz came back in an unexpected format. Try again."

AUTH_MESSAGE = "Your Anthropic API key was rejected. Check it in Settings."
PERMISSION_MESSAGE = (
    "Your Anthropic API key doesn't have access to this. "
    "Check your Anthropic account, or use a different key in Settings."
)
RATE_LIMIT_MESSAGE = "Claude is busy right now. Try again in a minute."
TOO_LONG_MESSAGE = "This section is too long for Claude to read in one go. Try a shorter section or passage."
NO_CREDIT_MESSAGE = (
    "Your Anthropic account is out of credit. Add credit in the Anthropic Console, then try again."
)
SERVER_MESSAGE = "Anthropic's servers had a problem. Try again in a few minutes."
CONNECTION_MESSAGE = "Couldn't reach Anthropic. Check your internet connection."
GENERIC_MESSAGE = "Something went wrong while talking to Claude. Try again."

_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]{4,}")
_LETTER_PREFIX = re.compile(r"^\(?([A-Ha-h])[.):]\s+")
_TOO_LONG_HINTS = ("prompt is too long", "too many tokens", "context window", "context length")


class AIUnavailable(RuntimeError):
    """The study helper can't be used: turned off, no API key, or SDK not installed.

    The message is safe to show to the user as is.
    """


class AIError(RuntimeError):
    """A request to Claude failed. The message is safe to show to the user as is."""


class QuizQuestion(BaseModel):
    question: str
    choices: list[str]
    answer_index: int
    explanation: str

    @model_validator(mode="after")
    def _check(self) -> QuizQuestion:
        if not self.question.strip():
            raise ValueError("question is empty")
        if len(self.choices) < 2 or any(not choice.strip() for choice in self.choices):
            raise ValueError("a question needs at least two non-empty choices")
        if not 0 <= self.answer_index < len(self.choices):
            raise ValueError("answer_index is out of range")
        return self


class Assistant:
    """Claude-powered study helpers. Each method returns text ready to be spoken.

    The client is created lazily on first use (once per Assistant) from the
    settings given here; build a new Assistant after the key or model changes.
    Methods raise :class:`AIUnavailable` when the helper can't be used and
    :class:`AIError` when a request fails. Neither message ever contains the key.
    """

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self._client_lock = threading.Lock()

    @property
    def model(self) -> str:
        return (self._settings.ai_model or "").strip() or DEFAULT_AI_MODEL

    def availability(self) -> tuple[bool, str]:
        """(usable, reason) without any network call. ``reason`` says how to fix it."""
        try:
            self._check_settings()
            if self._client is None:
                _import_anthropic()
        except AIUnavailable as exc:
            return False, str(exc)
        return True, ""

    # -- public helpers --------------------------------------------------------

    def explain(self, passage: str, *, context: str, book_title: str, section_title: str = "") -> str:
        """Plain-language explanation of ``passage``, using the section ``context``."""
        self._get_client()
        passage, context = passage or "", context or ""
        if not passage.strip():
            raise AIError("Select some text to explain first.")
        has_material = bool(context.strip())
        system = prompts.system_blocks(context, title=prompts.material_title(book_title, section_title))
        content = prompts.explain_prompt(passage, has_material=has_material)
        return self._answer(system, [{"role": "user", "content": content}], effort="medium")

    def ask(
        self,
        question: str,
        *,
        context: str,
        book_title: str,
        history: list[dict[str, str]] | None = None,
        section_title: str = "",
    ) -> str:
        """Answer a question about the section. ``history`` holds earlier turns as
        ``{"role": "user" | "assistant", "content": str}``; only the last
        :data:`MAX_HISTORY_TURNS` question/answer pairs are sent."""
        self._get_client()
        question, context = question or "", context or ""
        if not question.strip():
            raise AIError("Type a question first.")
        has_material = bool(context.strip())
        system = prompts.system_blocks(context, title=prompts.material_title(book_title, section_title))
        messages = _history_messages(history)
        messages.append({"role": "user", "content": prompts.ask_prompt(question, has_material=has_material)})
        return self._answer(system, messages, effort="medium")

    def summarize(
        self,
        section_text: str,
        *,
        section_title: str,
        book_title: str,
        length: Literal["short", "detailed"] = "short",
    ) -> str:
        """A short (5-8 sentences) or detailed (~400-700 words) spoken summary."""
        self._get_client()
        section_text = section_text or ""
        if not section_text.strip():
            raise AIError("This section has no text to summarize.")
        if length not in ("short", "detailed"):
            raise ValueError(f"length must be 'short' or 'detailed', not {length!r}")
        system = prompts.system_blocks(section_text, title=prompts.material_title(book_title, section_title))
        content = prompts.summarize_prompt(length)
        return self._answer(system, [{"role": "user", "content": content}], effort="medium")

    def quiz(
        self, section_text: str, *, section_title: str, count: int = 5, book_title: str = ""
    ) -> list[QuizQuestion]:
        """``count`` (1-20) four-choice questions answerable from the section."""
        self._get_client()
        section_text = section_text or ""
        if not section_text.strip():
            raise AIError("This section has no text to make a quiz from.")
        count = max(1, min(int(count), MAX_QUIZ_QUESTIONS))
        system = prompts.system_blocks(section_text, title=prompts.material_title(book_title, section_title))
        messages = [{"role": "user", "content": prompts.quiz_prompt(count)}]
        output_format = {"type": "json_schema", "schema": prompts.QUIZ_SCHEMA}
        for attempt in (1, 2):
            response = self._send(system, messages, effort="medium", output_format=output_format)
            raw, _truncated = _response_text(response)
            questions = parse_quiz(raw, count)
            if questions:
                return questions
            logger.info("Quiz response was not valid JSON for the schema (attempt %d of 2)", attempt)
        raise AIError(QUIZ_FORMAT_MESSAGE)

    def narrate_block(
        self, block: Block, *, context: str = "", book_title: str = "", section_title: str = ""
    ) -> str:
        """Turn a table or equation (or any block) into natural speech."""
        self._get_client()
        if not prompts.block_content(block).strip():
            raise AIError("That part of the page has no text for Claude to read.")
        system = prompts.system_blocks(context, title=prompts.material_title(book_title, section_title))
        content = prompts.narrate_prompt(block)
        return self._answer(system, [{"role": "user", "content": content}], effort="low")

    def describe_image(
        self, png: bytes, *, context: str = "", book_title: str = "", section_title: str = ""
    ) -> str:
        """Describe the figures, diagrams, charts and photos on a rendered page image."""
        self._get_client()
        context = context or ""
        media_type = _image_media_type(png or b"")
        if media_type is None:
            raise AIError("That page image couldn't be read, so it can't be described.")
        system = prompts.system_blocks(context, title=prompts.material_title(book_title, section_title))
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(png).decode("ascii"),
                },
            },
            {"type": "text", "text": prompts.describe_image_prompt(has_material=bool(context.strip()))},
        ]
        return self._answer(system, [{"role": "user", "content": content}], effort="medium")

    # -- internals -------------------------------------------------------------

    def _check_settings(self) -> None:
        if not self._settings.ai_enabled:
            raise AIUnavailable(OFF_MESSAGE)
        if not self._settings.resolved_api_key():
            raise AIUnavailable(NO_KEY_MESSAGE)

    def _get_client(self) -> Any:
        self._check_settings()
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    anthropic = _import_anthropic()
                    self._client = anthropic.Anthropic(
                        api_key=self._settings.resolved_api_key(),
                        max_retries=CLIENT_MAX_RETRIES,
                        timeout=CLIENT_TIMEOUT_S,
                    )
        return self._client

    def _request(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        *,
        effort: Effort,
        output_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Claude Opus 5.5 always thinks (adaptive); effort is the only depth/cost
        # control. Sampling parameters, thinking budgets and assistant prefill are
        # all rejected by the API, so none are sent.
        output_config: dict[str, Any] = {"effort": effort}
        if output_format is not None:
            output_config["format"] = output_format
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
            "output_config": output_config,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def _send(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        *,
        effort: Effort,
        output_format: dict[str, Any] | None = None,
    ) -> Any:
        client = self._get_client()
        request = self._request(system, messages, effort=effort, output_format=output_format)
        try:
            return client.beta.messages.create(**request)
        except Exception as exc:
            friendly = self._friendly_error(exc)
            if friendly is None:
                raise
            # Not chained: the SDK exception carries the request (and its headers).
            raise friendly from None

    def _answer(self, system: list[dict[str, Any]], messages: list[dict[str, Any]], *, effort: Effort) -> str:
        response = self._send(system, messages, effort=effort)
        raw, truncated = _response_text(response)
        text = to_speakable(raw)
        if not text:
            raise AIError(OUT_OF_ROOM_MESSAGE if truncated else EMPTY_ANSWER_MESSAGE)
        if truncated:
            text = f"{text}\n\n{MAX_TOKENS_NOTE}"
        return text

    def _friendly_error(self, exc: BaseException) -> AIError | None:
        """Map an SDK exception to a friendly :class:`AIError` (None if not an SDK error)."""
        try:
            import anthropic
        except ImportError:
            return None
        if not isinstance(exc, anthropic.AnthropicError):
            return None

        status = getattr(exc, "status_code", None)
        logger.warning(
            "Claude request failed: %s%s%s",
            type(exc).__name__,
            f" (HTTP {status})" if status else "",
            f" request id {exc.request_id}" if getattr(exc, "request_id", None) else "",
        )

        if isinstance(exc, anthropic.AuthenticationError):
            message = AUTH_MESSAGE
        elif isinstance(exc, anthropic.PermissionDeniedError):
            message = PERMISSION_MESSAGE
        elif isinstance(exc, anthropic.NotFoundError):
            message = (
                f'The Claude model set in Settings ("{self.model}") isn\'t available to your '
                f"API key. Check the model name in Settings (the default is {DEFAULT_AI_MODEL})."
            )
        elif isinstance(exc, anthropic.RateLimitError):
            message = RATE_LIMIT_MESSAGE
        elif isinstance(exc, anthropic.RequestTooLargeError):
            message = TOO_LONG_MESSAGE
        elif isinstance(exc, anthropic.BadRequestError):
            message = _bad_request_message(_error_reason(exc))
        elif isinstance(exc, anthropic.APIStatusError):
            if exc.status_code >= 500:
                message = SERVER_MESSAGE
            else:
                message = (
                    f"Claude couldn't answer right now (error {exc.status_code}). Try again in a moment."
                )
        elif isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
            message = CONNECTION_MESSAGE
        else:
            message = GENERIC_MESSAGE
        return AIError(self._redact(message))

    def _redact(self, text: str) -> str:
        key = self._settings.resolved_api_key()
        if key and len(key) >= 4:
            text = text.replace(key, "[hidden]")
        return _KEY_PATTERN.sub("[hidden]", text)


# -- module helpers ------------------------------------------------------------


def _import_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise AIUnavailable(NOT_INSTALLED_MESSAGE) from exc
    return anthropic


def _response_text(response: Any) -> tuple[str, bool]:
    """(text, truncated). Raises :class:`AIError` on a refusal.

    Only ``text`` blocks are read: thinking blocks (empty with Claude Opus 5.5)
    and fallback markers are skipped.
    """
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        logger.info("Claude declined (category: %s)", getattr(details, "category", None))
        raise AIError(REFUSAL_MESSAGE)
    parts = [
        block.text
        for block in getattr(response, "content", None) or []
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    ]
    truncated = stop_reason in ("max_tokens", "model_context_window_exceeded")
    return "".join(parts), truncated


def _history_messages(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Clean chat history into alternating user/assistant messages, last turns only.

    Invalid entries are skipped, consecutive same-role messages are merged, and a
    trailing unanswered question is dropped (the new question is sent separately).
    """
    messages: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, Mapping):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + content.strip()
        else:
            messages.append({"role": role, "content": content.strip()})
    if messages and messages[-1]["role"] == "user":
        messages.pop()
    messages = messages[-2 * MAX_HISTORY_TURNS :]
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


def _image_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _error_reason(exc: Any) -> str:
    """The API's own short explanation from an error body, if there is one."""
    body = getattr(exc, "body", None)
    reason: Any = None
    if isinstance(body, Mapping):
        error = body.get("error")
        reason = error.get("message") if isinstance(error, Mapping) else body.get("message")
    if not isinstance(reason, str):
        return ""
    reason = " ".join(reason.split())
    return reason if len(reason) <= 160 else reason[:157].rstrip() + "..."


def _bad_request_message(reason: str) -> str:
    lowered = reason.lower()
    if any(hint in lowered for hint in _TOO_LONG_HINTS):
        return TOO_LONG_MESSAGE
    if "credit balance" in lowered:
        return NO_CREDIT_MESSAGE
    if reason:
        return f"Claude couldn't handle that request ({reason.rstrip('.')}). Try a shorter passage, or try again."
    return "Claude couldn't handle that request. Try a shorter passage, or try again."


def _normalized(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def _strip_letter_prefixes(choices: list[str]) -> list[str]:
    """Drop "A) " / "b. " prefixes, but only when every choice has them in order."""
    matches = [_LETTER_PREFIX.match(choice) for choice in choices]
    if all(matches) and [m.group(1).lower() for m in matches if m] == list("abcdefgh"[: len(choices)]):
        return [choice[m.end() :] for choice, m in zip(choices, matches) if m]
    return choices


def parse_quiz(raw: str, count: int) -> list[QuizQuestion]:
    """Validated, de-duplicated questions from Claude's JSON (``[]`` if unusable)."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    questions: list[QuizQuestion] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        choices_raw = item.get("choices")
        if not isinstance(choices_raw, list) or not all(isinstance(c, str) for c in choices_raw):
            continue
        choices = [clean_inline(c) for c in _strip_letter_prefixes(choices_raw)]
        try:
            question = QuizQuestion(
                question=clean_inline(str(item.get("question", ""))),
                choices=choices,
                answer_index=item.get("answer_index"),
                explanation=clean_inline(str(item.get("explanation", ""))),
            )
        except ValidationError:
            continue
        if len({_normalized(c) for c in question.choices}) != len(question.choices):
            continue  # duplicate choices make the answer ambiguous
        key = _normalized(question.question)
        if key in seen:
            continue
        seen.add(key)
        questions.append(question)
        if len(questions) == count:
            break
    return questions
