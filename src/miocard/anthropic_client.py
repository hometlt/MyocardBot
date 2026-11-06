from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from miocard.analysis_cache import AnalysisCache
from miocard.config import Settings
from miocard.imaging import (
    ECGImageView,
    ProcessedImage,
    create_rhythm_views,
    detect_rhythm_timing_hints,
)
from miocard.models import (
    ANALYSIS_JSON_SCHEMA,
    NON_ECG_IMPRESSION,
    RHYTHM_SCOUT_JSON_SCHEMA,
    ECGAnalysisResponse,
    ECGImageFacts,
    ECGReport,
    FindingBox,
    LocalizedFinding,
    RhythmScoutResponse,
)
from miocard.prompt import SYSTEM_PROMPT, build_user_prompt


class AnthropicServiceError(RuntimeError):
    pass


class OutputLimitExceededError(AnthropicServiceError):
    pass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenBalance:
    used: int
    limit: int
    remaining: int


@dataclass
class ModelUsage:
    """Locally accumulated token counters for one model."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass
class UsageStats:
    """Session-wide usage; the gateway reports only a single aggregate counter."""

    analyses: int = 0
    cache_hits: int = 0
    api_errors: int = 0
    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    @property
    def requests(self) -> int:
        return sum(item.requests for item in self.by_model.values())

    @property
    def input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.by_model.values())

    @property
    def cache_read_tokens(self) -> int:
        return sum(item.cache_read_tokens for item in self.by_model.values())

    @property
    def cache_write_tokens(self) -> int:
        return sum(item.cache_write_tokens for item in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return sum(item.total_tokens for item in self.by_model.values())


@dataclass(frozen=True)
class RhythmCandidate:
    """One temporal event remapped from an enlarged view to the full sheet."""

    finding: LocalizedFinding
    confidence: str
    evidence: str
    center_x: int
    qrs_width: str
    source_view: str
    kind: str = "premature"


def _usage_value(usage: dict[str, object], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
UNAVAILABLE_MARKERS = (
    "не чита",
    "неразбор",
    "плохо чита",
    "обрез",
    "отсутств",
    "не вид",
    "не размеч",
)
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
UNAVAILABLE_LEAD_CLAIM = re.compile(
    r"(?:отведен\w*|крив\w*|строк\w*|канал\w*|верхн\w*)"
    r"[^.!?]{0,60}(?:не\s+чита|неразбор|плохо\s+чита|обрез|отсутств|не\s+вид|не\s+размеч)"
    r"|(?:не\s+видн\w*|отсутств\w*|обрез\w*)\s+"
    r"(?:все\s+)?(?:отведен\w*|крив\w*|строк\w*|канал\w*)",
    re.IGNORECASE,
)
ECTOPY_MARKERS = ("экстрасистол", "преждевременн")
SUPRAVENTRICULAR_MARKERS = ("наджелудочков", "суправентрикуляр", "предсердн")
UNCERTAIN_EVIDENCE_MARKERS = (
    "недостаточно",
    "не различ",
    "неразлич",
    "не вид",
    "нечет",
    "неяс",
    "сомнит",
)
HYPERTROPHY_PATTERN = re.compile(r"гипертроф\w*", re.IGNORECASE)
NAMED_VOLTAGE_CRITERION = re.compile(
    r"(?:соколов(?:а)?[-– ]?лайон|sokolow[-– ]?lyon|корнелл|cornell|"
    r"R\s*(?:V1|aVL)|S\s*(?:V3|V5|V6)|R\s*[/ :]+\s*S)",
    re.IGNORECASE,
)
NUMERIC_AMPLITUDE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:мм|мВ|mm|mV)", re.IGNORECASE)
ATRIAL_ABNORMALITY_PATTERN = re.compile(
    r"(?:p\s*[-– ]?\s*(?:mitrale|pulmonale)|"
    r"(?:увелич|перегруз|гипертроф)\w*.{0,35}(?:лев\w*|прав\w*)\s+предсерд\w*|"
    r"(?:лев\w*|прав\w*)\s+предсерд\w*.{0,35}(?:увелич|перегруз|гипертроф)\w*)",
    re.IGNORECASE,
)
NAMED_ATRIAL_CRITERION = re.compile(
    r"(?:p\s*[-– ]?\s*(?:mitrale|pulmonale)|p(?:tf|[-– ]?terminal\w*)\s*v?1|"
    r"моррис\w*|(?:p|р)(?:[-– ]?зуб\w*)?.{0,20}(?:II|V1|V2))",
    re.IGNORECASE,
)
NUMERIC_P_WAVE_MEASUREMENT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:мм|мВ|мс|mm|mV|ms|мм\s*[*×·]\s*с)",
    re.IGNORECASE,
)
HIGH_VOLTAGE_PHRASE = (
    "Высокий вольтаж грудных комплексов; клиническое значение по фотографии не определяется"
)
IMAGE_INVENTORY_INSTRUCTIONS = """Сначала заполни image_facts, не интерпретируя ритм и не ставя диагноз. В итоговом JSON используй только короткие ключи-алиасы из схемы: верхний уровень f=image_facts и r=report. Длинные смысловые имена из инструкций не дублируй.
Установи is_ecg_image=true только для листа или экрана ЭКГ с несколькими различимыми кривыми и характерными отведениями. Для обычной фотографии, другого документа или изображения без распознаваемых ЭКГ-кривых установи false, кратко заполни non_ecg_reason, очисти остальные факты и верни report=null.
Для ЭКГ независимо от подписей посчитай строки с кривыми сверху вниз. В visible_lead_labels перечисли только читаемые метки I, II, III, aVR, aVL, aVF, V1–V6. Если метка полноценной строки обрезана, но отведение надежно определяется по положению между соседними строками, добавь его в inferred_leads. Не восстанавливай отведение только ради полного стандартного набора.
Каждое найденное отведение помести ровно в один список: readable_waveform_leads или unreadable_waveform_leads. Шум, дрейф, низкая амплитуда и локальный артефакт сами по себе не делают кривую нечитаемой. Высокий или темный QRS не является насыщением без плоской обрезки или утраты части комплекса. Точно перепиши строку скорости и усиления.
После фиксации image_facts заполни report, не меняя эти факты. Пиши максимально компактно: значение observation — до 18 слов, evidence — одно предложение до 24 слов; у каждого possible_interpretation не более одного короткого довода и одного ограничения. Если доверенный контекст перечисляет кандидатов Cn, верни по каждому запись в report.candidate_verdicts (алиас cv, ключи n/v/r); без такого контекста оставь этот список пустым."""
SCHEMA_INSTRUCTION = (
    "\n\nВерни только один JSON-объект без Markdown и пояснений, строго "
    "соответствующий этой JSON Schema:\n"
    + json.dumps(ANALYSIS_JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))
)
RHYTHM_SCOUT_INSTRUCTIONS = """You perform one narrow technical task on an enlarged ECG fragment: inventory abnormally timed beats. Do not diagnose the overall rhythm and do not create a full ECG report.

Scan the complete width from left to right and again from right to left. Compare every QRS with both neighbours and the local dominant template. Search independently for (1) wide or clearly altered premature QRS, (2) narrow premature QRS identified mainly by an abruptly short coupling interval, and (3) abruptly long intervals: a dropped QRS after a visible P wave, a pause that does not follow a premature beat, or grouped conduction with progressive PR lengthening. Recheck the last four visible R–R intervals and do not stop after the first event.

Count distinct temporal events, not affected leads. Vertically aligned manifestations in simultaneous leads are one event. In a sequential layout, use local timing and morphology instead of merging different segments. Reject calibration pulses, lead-switching transients, grid lines, text and isolated one-lead deformation. A true candidate should have premature timing, an abrupt gap or altered morphology supported in the local lead and, when leads are simultaneous, at the same time in another readable lead.

Return one candidate per countable event, maximum five. Each box uses integer coordinates 0–1000 relative to this exact fragment and tightly encloses the event across its clearest adjacent leads; for a pause enclose the gap itself together with the beats on either side. Use `kind` as `premature` or `pause`, and `qrs_width` as `narrow`, `wide` or `uncertain` (`uncertain` for a pause). Keep `finding` and `evidence` short and write them in Russian. Emit only compact schema aliases: top-level `c` means candidates; candidate keys are `f,c,e,w,k,b`; box keys are `t,l,b,r`. If no abnormally timed event is visible, return `{"c":[]}`. Return exactly one minified JSON object and nothing else."""
RHYTHM_SCOUT_SCHEMA_INSTRUCTION = (
    "\n\nThe JSON object must follow this schema:\n"
    + json.dumps(RHYTHM_SCOUT_JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))
)
CACHE_FORMAT_VERSION = "multiscale-v2"
RHYTHM_SCOUT_CACHE_VERSION = "rhythm-scout-v2"


def _extract_json(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _error_detail(response: httpx.Response) -> str:
    """Extract a short provider error from object, string, or plain-text bodies."""
    try:
        payload = response.json()
    except ValueError:
        payload = response.text

    detail: object = payload
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("detail") or error
        elif error:
            detail = error
        else:
            detail = payload.get("message") or payload.get("detail") or "API error"

    if isinstance(detail, (dict, list)):
        text = json.dumps(detail, ensure_ascii=False)
    else:
        text = str(detail or "API error")
    return " ".join(text.split())[:500]


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after", "").strip()
    try:
        if retry_after:
            return min(max(float(retry_after), 0.0), 60.0)
    except ValueError:
        pass
    if response.status_code == 429:
        return float(5 * (2**attempt))
    return float(2**attempt)


class AnthropicClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self.response_models: list[str] = []
        self.response_usage: list[dict[str, object]] = []
        self.usage_stats = UsageStats()
        self._cache = (
            AnalysisCache(settings.analysis_cache_dir)
            if settings.analysis_cache_enabled
            else None
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.anthropic_timeout_seconds, connect=20),
            limits=httpx.Limits(max_connections=settings.max_concurrent_analyses + 2),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get_token_balance(self) -> TokenBalance | None:
        """Return Tokenator package counters; direct Anthropic has no such endpoint."""
        if not self._settings.is_tokenator:
            return None
        api_url = self._settings.effective_api_url.rstrip("/")
        balance_url = (
            api_url[: -len("/messages")] + "/tokens"
            if api_url.endswith("/messages")
            else api_url + "/tokens"
        )
        try:
            response = await self._client.get(
                balance_url,
                headers={
                    "x-api-key": self._settings.anthropic_api_key.get_secret_value(),
                    "anthropic-version": "2023-06-01",
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AnthropicServiceError(
                "Не удалось получить остаток токенов Tokenator."
            ) from exc
        if not isinstance(payload, dict):
            raise AnthropicServiceError(
                "Tokenator вернул остаток токенов в неподдерживаемом формате."
            )
        try:
            balance = TokenBalance(
                used=int(payload["used"]),
                limit=int(payload["limit"]),
                remaining=int(payload["remaining"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AnthropicServiceError(
                "Tokenator вернул неполные данные об остатке токенов."
            ) from exc
        if min(balance.used, balance.limit, balance.remaining) < 0:
            raise AnthropicServiceError(
                "Tokenator вернул некорректные данные об остатке токенов."
            )
        return balance

    def _record_usage(self, model: str, usage: dict[str, object]) -> None:
        entry = self.usage_stats.by_model.setdefault(model, ModelUsage())
        entry.requests += 1
        entry.input_tokens += _usage_value(usage, "input_tokens", "prompt_tokens")
        entry.output_tokens += _usage_value(usage, "output_tokens", "completion_tokens")
        entry.cache_read_tokens += _usage_value(usage, "cache_read_input_tokens")
        entry.cache_write_tokens += _usage_value(usage, "cache_creation_input_tokens")

    def _apply_model_controls(
        self,
        body: dict[str, object],
        *,
        effort: str,
        schema: object,
    ) -> None:
        """Reasoning and output controls, which differ between gateway and direct API."""
        if self._settings.is_tokenator:
            # The gateway takes an OpenAI-shaped reasoning switch and does not constrain
            # the response, so the schema travels inside the prompt instead.
            body["reasoning"] = {"enabled": self._settings.anthropic_thinking_enabled}
            return
        if self._settings.anthropic_thinking_enabled:
            # Adaptive thinking is the only supported on-mode on every current Claude
            # model. The multi-pass beat inventory the specification asks for has
            # nowhere to run without it, because the same specification requires the
            # visible JSON to stay compact. Thinking is omitted rather than explicitly
            # disabled, since an explicit disable is rejected by Fable 5.
            body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        }

    def _system_prompt(self) -> str:
        """Every instruction that does not vary between requests, in one stable block."""
        text = SYSTEM_PROMPT + "\n\n" + IMAGE_INVENTORY_INSTRUCTIONS
        if self._settings.is_tokenator:
            # Tokenator currently accepts output_config but does not constrain the
            # response, so the schema travels in the prompt as a compatibility fallback.
            text += SCHEMA_INSTRUCTION
        return text

    def analysis_system_prompt(self) -> str:
        """Return the exact system prompt used by the main ECG analysis agent."""
        return self._system_prompt()

    def _system_field(self) -> object:
        text = self._system_prompt()
        if self._settings.is_tokenator or not self._settings.prompt_cache_enabled:
            return text
        # The instructions and schema are identical on every analysis, so Anthropic can
        # serve them from its prompt cache instead of re-reading them each time.
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _rhythm_scout_system_prompt(self) -> str:
        text = RHYTHM_SCOUT_INSTRUCTIONS
        if self._settings.is_tokenator:
            text += RHYTHM_SCOUT_SCHEMA_INSTRUCTION
        return text

    def _rhythm_scout_system_field(self) -> object:
        text = self._rhythm_scout_system_prompt()
        if self._settings.is_tokenator or not self._settings.prompt_cache_enabled:
            return text
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _cache_key(
        self,
        image: ProcessedImage,
        model: str,
        context: str | None,
        analysis_context: str | None,
    ) -> str:
        digest = hashlib.sha256()
        components = (
            CACHE_FORMAT_VERSION,
            model,
            # Effort and reasoning change the answer as much as the prompt does, so a
            # report produced under one setting must not be served under another.
            self._settings.anthropic_effort,
            str(self._settings.anthropic_thinking_enabled),
            context or "",
            analysis_context or "",
            image.media_type,
            image.metadata_text(),
            self._system_prompt(),
        )
        for component in components:
            digest.update(component.encode("utf-8"))
            digest.update(b"\0")
        digest.update(image.data)
        return digest.hexdigest()

    def _rhythm_scout_cache_key(
        self,
        image: ProcessedImage,
        model: str,
        view: ECGImageView,
    ) -> str:
        digest = hashlib.sha256()
        components = (
            RHYTHM_SCOUT_CACHE_VERSION,
            model,
            self._settings.anthropic_scout_effort,
            str(self._settings.anthropic_thinking_enabled),
            view.name,
            str(view.left),
            str(view.right),
            image.media_type,
            image.metadata_text(),
            self._rhythm_scout_system_prompt(),
        )
        for component in components:
            digest.update(component.encode("utf-8"))
            digest.update(b"\0")
        digest.update(image.data)
        return digest.hexdigest()

    @staticmethod
    def _image_block(image: ProcessedImage) -> dict[str, object]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": base64.b64encode(image.data).decode("ascii"),
            },
        }

    async def _request_text(self, body: dict[str, object], phase: str) -> str:
        try:
            return await self._request_text_once(body, phase)
        except AnthropicServiceError:
            self.usage_stats.api_errors += 1
            raise

    async def _request_text_once(self, body: dict[str, object], phase: str) -> str:
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                response = await self._client.post(
                    self._settings.effective_api_url,
                    headers={
                        "x-api-key": self._settings.anthropic_api_key.get_secret_value(),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=body,
                )
                response.raise_for_status()
                break
            except httpx.TimeoutException as exc:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise AnthropicServiceError(f"API не ответил вовремя ({phase}).") from exc
            except httpx.HTTPStatusError as exc:
                detail = _error_detail(exc.response)
                output_limit_exhausted = "output limit exceeded" in detail.lower()
                if (
                    exc.response.status_code in RETRYABLE_STATUS_CODES
                    and not output_limit_exhausted
                    and attempt < 2
                ):
                    await asyncio.sleep(_retry_delay(exc.response, attempt))
                    continue
                request_id = exc.response.headers.get("request-id", "unknown")
                if output_limit_exhausted:
                    raise OutputLimitExceededError(
                        "Лимит выходных токенов API-ключа исчерпан. "
                        "Модератору необходимо пополнить лимит или заменить ключ."
                    ) from exc
                if exc.response.status_code == 429:
                    raise AnthropicServiceError(
                        f"API временно ограничил частоту запросов ({phase}): {detail} "
                        f"(request-id: {request_id})"
                    ) from exc
                raise AnthropicServiceError(
                    f"API вернул ошибку {exc.response.status_code} ({phase}): {detail} "
                    f"(request-id: {request_id})"
                ) from exc
            except httpx.HTTPError as exc:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise AnthropicServiceError(f"Не удалось подключиться к API ({phase}).") from exc

        if response is None:
            raise AnthropicServiceError(f"API не вернул ответ ({phase}).")

        try:
            payload = response.json()
        except ValueError as exc:
            raise AnthropicServiceError(
                f"API вернул ответ в неподдерживаемом формате ({phase})."
            ) from exc
        if not isinstance(payload, dict):
            raise AnthropicServiceError(
                f"API вернул ответ в неподдерживаемом формате ({phase})."
            )
        response_model = payload.get("model")
        if isinstance(response_model, str):
            self.response_models.append(response_model)
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self.response_usage.append(usage)
            requested_model = body.get("model")
            self._record_usage(
                requested_model if isinstance(requested_model, str) else "unknown",
                usage,
            )
        if payload.get("stop_reason") in {"max_tokens", "refusal"}:
            raise AnthropicServiceError(
                f"Запрос не завершен ({phase}): stop_reason={payload.get('stop_reason')}."
            )
        text_blocks = [
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        ]
        if not text_blocks:
            raise AnthropicServiceError(f"API вернул пустой ответ ({phase}).")
        return "".join(text_blocks)

    @staticmethod
    def _normalize_image_facts(facts: ECGImageFacts) -> ECGImageFacts:
        if not facts.is_ecg_image:
            return facts.model_copy(
                update={
                    "visible_lead_labels": [],
                    "inferred_leads": [],
                    "missing_standard_leads": list(STANDARD_LEADS),
                    "readable_waveform_leads": [],
                    "unreadable_waveform_leads": [],
                    "calibration_text": "",
                    "speed_mm_s": None,
                    "gain_values_mm_mv": [],
                }
            )
        labeled_values = set(facts.visible_lead_labels)
        inferred_by_lead = {
            item.lead: item for item in facts.inferred_leads if item.lead not in labeled_values
        }
        inferred = [inferred_by_lead[lead] for lead in STANDARD_LEADS if lead in inferred_by_lead]
        visible_values = labeled_values | set(inferred_by_lead)
        visible = [lead for lead in STANDARD_LEADS if lead in visible_values]
        explicitly_unreadable = set(facts.unreadable_waveform_leads)
        explicitly_readable = (
            set(facts.readable_waveform_leads) | set(inferred_by_lead)
        ) - explicitly_unreadable
        readable = [lead for lead in visible if lead in explicitly_readable]
        # The preflight must partition every visible lead. An omitted/ambiguous lead is
        # conservatively treated as unreadable instead of being guessed usable.
        unreadable = [lead for lead in visible if lead not in explicitly_readable]
        return facts.model_copy(
            update={
                "visible_lead_labels": visible,
                "inferred_leads": inferred,
                "missing_standard_leads": [lead for lead in STANDARD_LEADS if lead not in visible],
                "readable_waveform_leads": readable,
                "unreadable_waveform_leads": unreadable,
            }
        )

    @staticmethod
    def _claims_locked_lead_is_unavailable(text: str, readable_leads: set[str]) -> bool:
        if not readable_leads:
            return False
        lowered = text.lower()
        if not any(marker in lowered for marker in UNAVAILABLE_MARKERS):
            return False
        return UNAVAILABLE_LEAD_CLAIM.search(text) is not None

    @classmethod
    def _remove_locked_fact_conflicts(
        cls,
        report: ECGReport,
        facts: ECGImageFacts,
        context: str | None,
    ) -> None:
        readable = set(facts.readable_waveform_leads)
        inferred = {item.lead for item in facts.inferred_leads}

        def conflicts(text: str) -> bool:
            return cls._claims_locked_lead_is_unavailable(text, readable)

        def without_conflicting_clauses(text: str) -> str:
            clauses = re.split(r"(?<=[.!?;])\s+", text.strip())
            return " ".join(clause for clause in clauses if clause and not conflicts(clause)).strip()

        def requests_reacquisition_of_inferred_lead(text: str) -> bool:
            lowered = text.lower()
            if not any(
                marker in lowered
                for marker in ("добав", "включ", "повтор", "пересня", "получ", "запис")
            ):
                return False
            return any(
                re.search(rf"(?<![A-Za-z0-9]){re.escape(lead)}(?![A-Za-z0-9])", text)
                for lead in inferred
            )

        report.image_quality.issues = [item for item in report.image_quality.issues if not conflicts(item)]
        report.limitations = [item for item in report.limitations if not conflicts(item)]
        report.recommended_checks = [
            item
            for item in report.recommended_checks
            if not conflicts(item) and not requests_reacquisition_of_inferred_lead(item)
        ]

        if context:
            context_markers = ("не распоз", "поврежден", "кодиров", "искажен")
            report.limitations = [
                item
                for item in report.limitations
                if not ("контекст" in item.lower() and any(marker in item.lower() for marker in context_markers))
            ]
            report.recommended_checks = [
                item
                for item in report.recommended_checks
                if not ("контекст" in item.lower() and any(marker in item.lower() for marker in context_markers))
            ]

        for field_name in report.observations.__class__.model_fields:
            observation = getattr(report.observations, field_name)
            if conflicts(observation.value):
                cleaned_value = without_conflicting_clauses(observation.value)
                observation.value = cleaned_value or "Не оценивается"
            if conflicts(observation.evidence):
                cleaned_evidence = without_conflicting_clauses(observation.evidence)
                observation.evidence = cleaned_evidence or "Оценка выполнена по доступным кривым."

        report.possible_interpretations = [
            item
            for item in report.possible_interpretations
            if not conflicts(item.label)
            and not any(conflicts(value) for value in item.supporting_evidence)
        ]
        for item in report.possible_interpretations:
            item.contradicting_or_missing_evidence = [
                value for value in item.contradicting_or_missing_evidence if not conflicts(value)
            ]
        report.urgent_findings = [
            item
            for item in report.urgent_findings
            if not conflicts(item.finding) and not conflicts(item.rationale)
        ]

        sentences = re.split(r"(?<=[.!?])\s+", report.concise_impression.strip())
        report.concise_impression = " ".join(
            sentence for sentence in sentences if sentence and not conflicts(sentence)
        ).strip()
        if not report.concise_impression:
            report.concise_impression = "Доступен структурированный предварительный разбор изображения."

    @staticmethod
    def _apply_clinical_consistency_rules(report: ECGReport) -> None:
        """Enforce a few evidence/confidence invariants across model providers."""

        def report_strings() -> list[str]:
            values = [report.concise_impression]
            for field_name in report.observations.__class__.model_fields:
                observation = getattr(report.observations, field_name)
                values.extend((observation.value, observation.evidence))
            for item in report.possible_interpretations:
                values.append(item.label)
                values.extend(item.supporting_evidence)
                values.extend(item.contradicting_or_missing_evidence)
            values.extend(report.limitations)
            values.extend(report.recommended_checks)
            return values

        def evidence_contexts() -> list[str]:
            values = []
            for field_name in report.observations.__class__.model_fields:
                observation = getattr(report.observations, field_name)
                values.append(" ".join((observation.value, observation.evidence)))
            for item in report.possible_interpretations:
                values.append(
                    " ".join(
                        [
                            item.label,
                            *item.supporting_evidence,
                            *item.contradicting_or_missing_evidence,
                        ]
                    )
                )
            return values

        def without_pattern(
            value: str,
            pattern: re.Pattern[str],
            fallback: str = "",
        ) -> str:
            clauses = re.split(r"(?<=[.!?;])\s+", value.strip())
            cleaned = " ".join(
                clause for clause in clauses if clause and not pattern.search(clause)
            ).strip()
            return cleaned or fallback

        def mentions_supraventricular_ectopy(text: str) -> bool:
            lowered = text.lower()
            return any(marker in lowered for marker in ECTOPY_MARKERS) and (
                any(marker in lowered for marker in SUPRAVENTRICULAR_MARKERS)
                or "узк" in lowered
            )

        def says_p_is_unclear(text: str) -> bool:
            lowered = text.lower()
            has_p_reference = bool(
                re.search(
                    r"(?<![a-zа-яё])(?:p|р)(?:[-– ]?зуб\w*)?(?![a-zа-яё])",
                    lowered,
                )
                or "п-зуб" in lowered
            )
            return has_p_reference and (
                any(marker in lowered for marker in UNCERTAIN_EVIDENCE_MARKERS)
                or bool(re.search(r"не\s+.{0,50}(?:различ|прослеж|вид)", lowered))
            )

        rhythm = report.observations.rhythm
        rhythm_text = " ".join((rhythm.value, rhythm.evidence))
        interpretation_text = " ".join(
            " ".join(
                [
                    item.label,
                    *item.supporting_evidence,
                    *item.contradicting_or_missing_evidence,
                ]
            )
            for item in report.possible_interpretations
        )
        all_rhythm_text = f"{rhythm_text} {interpretation_text}"
        if mentions_supraventricular_ectopy(all_rhythm_text) and says_p_is_unclear(
            all_rhythm_text
        ):
            if mentions_supraventricular_ectopy(rhythm_text):
                rhythm.confidence = "low"
            for item in report.possible_interpretations:
                item_text = " ".join(
                    [
                        item.label,
                        *item.supporting_evidence,
                        *item.contradicting_or_missing_evidence,
                    ]
                )
                if mentions_supraventricular_ectopy(item_text):
                    item.confidence = "low"

        rhythm_lower = rhythm.value.lower()
        rhythm_evidence_lower = rhythm.evidence.lower()
        clear_p_before_qrs = bool(
            re.search(
                r"(?:p|р)(?:[-– ]?зуб\w*)?.{0,80}(?:перед|предшеств).{0,60}qrs",
                rhythm_evidence_lower,
            )
        )
        if (
            rhythm.confidence == "low"
            and "синусов" in rhythm_lower
            and clear_p_before_qrs
            and not mentions_supraventricular_ectopy(rhythm_text)
            and not says_p_is_unclear(rhythm_text)
        ):
            rhythm.confidence = "medium"

        # A provider may still rank flutter as probable while admitting that its
        # mandatory organized atrial cycle was not demonstrated. Keep such a report
        # as a low-confidence AF/flutter differential instead of returning an
        # internally contradictory mechanism label.
        flutter_unconfirmed = False
        mandatory_flutter_markers = (
            "организ",
            "стабил",
            "период",
            "цикл",
            "фаз",
            "независ",
            "f-волн",
            "предсердн",
        )
        uncertainty_markers = UNCERTAIN_EVIDENCE_MARKERS + (
            "не подтверж",
            "не доказ",
            "не прослеж",
        )
        for item in report.possible_interpretations:
            if "трепетан" not in item.label.lower():
                continue
            missing = " ".join(item.contradicting_or_missing_evidence).lower()
            if any(marker in missing for marker in mandatory_flutter_markers) and any(
                marker in missing for marker in uncertainty_markers
            ):
                item.confidence = "low"
                flutter_unconfirmed = True

        if flutter_unconfirmed and "трепетан" in rhythm.value.lower():
            base_rhythm = re.split(r"[.;]", rhythm.value, maxsplit=1)[0].strip()
            rhythm.value = (
                f"{base_rhythm}; фибрилляция и трепетание предсердий "
                "требуют дифференциации"
            )
            rhythm.confidence = "low"
            report.overall_confidence = "low"
            report.concise_impression = rhythm.value + "."

        strings = report_strings()
        combined = " ".join(strings)
        has_measured_atrial_criterion = any(
            NAMED_ATRIAL_CRITERION.search(value)
            and NUMERIC_P_WAVE_MEASUREMENT.search(value)
            for value in evidence_contexts()
        )
        if ATRIAL_ABNORMALITY_PATTERN.search(combined) and not has_measured_atrial_criterion:
            for field_name in report.observations.__class__.model_fields:
                observation = getattr(report.observations, field_name)
                if ATRIAL_ABNORMALITY_PATTERN.search(
                    " ".join((observation.value, observation.evidence))
                ):
                    observation.value = without_pattern(
                        observation.value,
                        ATRIAL_ABNORMALITY_PATTERN,
                        "Морфология P-зубца требует ручного измерения",
                    )
                    observation.evidence = without_pattern(
                        observation.evidence,
                        ATRIAL_ABNORMALITY_PATTERN,
                        "Критерий P-зубца вручную не измерен.",
                    )
                    observation.confidence = "low"

            report.possible_interpretations = [
                item
                for item in report.possible_interpretations
                if not ATRIAL_ABNORMALITY_PATTERN.search(
                    " ".join(
                        [
                            item.label,
                            *item.supporting_evidence,
                            *item.contradicting_or_missing_evidence,
                        ]
                    )
                )
            ]
            report.concise_impression = without_pattern(
                report.concise_impression,
                ATRIAL_ABNORMALITY_PATTERN,
                "Доступен структурированный предварительный разбор изображения.",
            )
            report.limitations = [
                value
                for value in report.limitations
                if not ATRIAL_ABNORMALITY_PATTERN.search(value)
            ]
            report.recommended_checks = [
                value
                for value in report.recommended_checks
                if not ATRIAL_ABNORMALITY_PATTERN.search(value)
            ]
            strings = report_strings()
            combined = " ".join(strings)

        has_measured_named_criterion = any(
            NAMED_VOLTAGE_CRITERION.search(value) and NUMERIC_AMPLITUDE.search(value)
            for value in evidence_contexts()
        )
        if not HYPERTROPHY_PATTERN.search(combined) or has_measured_named_criterion:
            return

        qrs = report.observations.qrs_complex
        qrs_lower = qrs.value.lower()
        if "высок" in qrs_lower and ("вольтаж" in qrs_lower or "амплитуд" in qrs_lower):
            qrs.value = HIGH_VOLTAGE_PHRASE
            qrs.confidence = "low"

        for field_name in report.observations.__class__.model_fields:
            observation = getattr(report.observations, field_name)
            if HYPERTROPHY_PATTERN.search(observation.value) or HYPERTROPHY_PATTERN.search(
                observation.evidence
            ):
                observation.value = HIGH_VOLTAGE_PHRASE
                observation.confidence = "low"
                observation.evidence = without_pattern(
                    observation.evidence,
                    HYPERTROPHY_PATTERN,
                    "Амплитудные критерии вручную не измерены.",
                )

        for item in report.possible_interpretations:
            item_text = " ".join(
                [item.label, *item.supporting_evidence, *item.contradicting_or_missing_evidence]
            )
            if not HYPERTROPHY_PATTERN.search(item_text):
                continue
            item_label_lower = item.label.lower()
            if HYPERTROPHY_PATTERN.search(item.label) or (
                "высок" in item_label_lower
                and ("вольтаж" in item_label_lower or "амплитуд" in item_label_lower)
            ):
                item.label = HIGH_VOLTAGE_PHRASE
            item.confidence = "low"
            item.supporting_evidence = [
                value
                for value in item.supporting_evidence
                if not HYPERTROPHY_PATTERN.search(value)
            ]
            item.contradicting_or_missing_evidence = [
                value
                for value in item.contradicting_or_missing_evidence
                if not HYPERTROPHY_PATTERN.search(value)
            ]
            if "Амплитудные критерии вручную не измерены" not in item.contradicting_or_missing_evidence:
                item.contradicting_or_missing_evidence.append(
                    "Амплитудные критерии вручную не измерены"
                )

        if HYPERTROPHY_PATTERN.search(report.concise_impression):
            report.concise_impression = without_pattern(
                report.concise_impression,
                HYPERTROPHY_PATTERN,
                HIGH_VOLTAGE_PHRASE,
            )
        report.limitations = [
            value for value in report.limitations if not HYPERTROPHY_PATTERN.search(value)
        ]
        report.recommended_checks = [
            value for value in report.recommended_checks if not HYPERTROPHY_PATTERN.search(value)
        ]

    @staticmethod
    def _non_ecg_report(facts: ECGImageFacts) -> ECGReport:
        unavailable = {
            "value": "Не оценивается",
            "confidence": "low",
            "evidence": "На изображении не распознана диагностическая 12-канальная ЭКГ.",
        }
        return ECGReport.model_validate(
            {
                "image_quality": {
                    "grade": "non_diagnostic",
                    "issues": [facts.non_ecg_reason] if facts.non_ecg_reason else [],
                    "visible_leads": [],
                    "usable_leads": [],
                    "unreadable_leads": [],
                },
                "calibration": {
                    "raw_text": "",
                    "speed_mm_s": None,
                    "gain_mm_mv": None,
                    "gain_values_mm_mv": [],
                    "gain_notes": "Калибровка ЭКГ не распознана.",
                    "confidence": "low",
                    "source": "Автоматическая проверка изображения",
                },
                "observations": {
                    "rhythm": dict(unavailable),
                    "heart_rate": dict(unavailable),
                    "pr_interval": dict(unavailable),
                    "qrs_complex": dict(unavailable),
                    "electrical_axis": dict(unavailable),
                    "q_waves": dict(unavailable),
                    "r_progression": dict(unavailable),
                    "st_t": dict(unavailable),
                    "qt_qtc": dict(unavailable),
                },
                "possible_interpretations": [],
                "urgent_findings": [],
                "limitations": [],
                "recommended_checks": [],
                "overall_confidence": "low",
                "concise_impression": NON_ECG_IMPRESSION,
            }
        )

    @staticmethod
    def _view_analysis_context(view: ECGImageView) -> str:
        return (
            f"Увеличенный {view.name} фрагмент полного листа: "
            f"x={view.left}–{view.right}/1000, вся полезная высота. "
            "Координаты кандидатов указывай относительно этого фрагмента."
        )

    @staticmethod
    def _finding_is_ectopic(report: ECGReport, finding: LocalizedFinding) -> bool:
        if finding.related_observation not in {"rhythm", "qrs_complex"}:
            return False
        text = finding.finding.lower()
        if finding.related_observation == "rhythm":
            text += " " + report.observations.rhythm.value.lower()
        return any(marker in text for marker in ECTOPY_MARKERS)

    @staticmethod
    def _remap_box(box: FindingBox, view: ECGImageView) -> FindingBox:
        span = view.right - view.left
        left = view.left + round(box.left * span / 1000)
        right = view.left + round(box.right * span / 1000)
        left = min(999, max(0, left))
        right = min(1000, max(left + 1, right))
        return FindingBox(
            top=box.top,
            left=left,
            bottom=box.bottom,
            right=right,
        )

    @classmethod
    def _rhythm_candidates_from_scout(
        cls,
        scout: RhythmScoutResponse,
        view: ECGImageView,
    ) -> list[RhythmCandidate]:
        candidates: list[RhythmCandidate] = []
        for item in scout.candidates:
            mapped_box = cls._remap_box(item.box, view)
            candidates.append(
                RhythmCandidate(
                    finding=LocalizedFinding(
                        finding=item.finding,
                        related_observation="rhythm",
                        box=mapped_box,
                    ),
                    confidence=item.confidence,
                    evidence=item.evidence,
                    center_x=round((mapped_box.left + mapped_box.right) / 2),
                    qrs_width=item.qrs_width,
                    source_view=view.name,
                    kind=item.kind,
                )
            )
        return candidates

    @staticmethod
    def _deduplicate_rhythm_candidates(
        candidates: list[RhythmCandidate],
    ) -> list[RhythmCandidate]:
        confidence_score = {"low": 1, "medium": 2, "high": 3}
        merged: list[RhythmCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.center_x):
            # A pause and a premature complex can legitimately sit next to each other,
            # so only same-kind sightings at the same position are one event.
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if existing.kind == candidate.kind
                    and abs(existing.center_x - candidate.center_x) <= 35
                ),
                None,
            )
            if duplicate_index is None:
                merged.append(candidate)
                continue
            existing = merged[duplicate_index]
            existing_score = confidence_score.get(existing.confidence, 0)
            candidate_score = confidence_score.get(candidate.confidence, 0)
            existing_width = existing.finding.box.right - existing.finding.box.left
            candidate_width = candidate.finding.box.right - candidate.finding.box.left
            if (candidate_score, -candidate_width) > (existing_score, -existing_width):
                merged[duplicate_index] = candidate
        return sorted(merged, key=lambda item: item.center_x)

    @staticmethod
    def _timing_candidates(image: ProcessedImage) -> list[RhythmCandidate]:
        candidates: list[RhythmCandidate] = []
        for hint in detect_rhythm_timing_hints(image):
            pause = hint.kind == "pause"
            # A pause is a gap between two beats, so its marker has to be wide enough to
            # contain both of them rather than pointing at a single complex.
            half_width = 45 if pause else 18
            box = FindingBox(
                top=580,
                left=max(0, hint.x - half_width),
                bottom=950,
                right=min(1000, hint.x + half_width),
            )
            candidates.append(
                RhythmCandidate(
                    finding=LocalizedFinding(
                        finding="Удлиненный интервал R–R"
                        if pause
                        else "Преждевременный комплекс по R–R",
                        related_observation="rhythm",
                        box=box,
                    ),
                    confidence=hint.confidence,
                    evidence=(
                        f"Программно измеренный R–R ≈ {hint.rr_px:.0f} px "
                        f"против локальной медианы ≈ {hint.median_rr_px:.0f} px; "
                        f"отношение {hint.coupling_ratio:.2f}. Кандидат требует визуальной "
                        "проверки морфологии и синхронности."
                    ),
                    center_x=hint.x,
                    qrs_width="uncertain",
                    source_view="timing",
                    kind=hint.kind,
                )
            )
        return candidates

    @staticmethod
    def _multiscale_context(candidates: list[RhythmCandidate]) -> str | None:
        if not candidates:
            return None
        width_label = {
            "narrow": "узкий QRS",
            "wide": "широкий или изменённый QRS",
            "uncertain": "ширина QRS неуточнена",
        }
        kind_label = {
            "premature": "преждевременный комплекс",
            "pause": "удлиненный интервал (пауза или выпадение)",
        }
        lines = [
            "Доверенная мультимасштабная инвентаризация приложения, не пользовательский "
            "анамнез и не готовый диагноз: независимые увеличенные фрагменты и "
            "программное измерение R–R после объединения перекрытий дали следующие "
            "кандидаты разных временных позиций на полном листе:"
        ]
        for index, candidate in enumerate(candidates, start=1):
            evidence = " ".join(candidate.evidence.split())[:500]
            lines.append(
                f"C{index}: {kind_label.get(candidate.kind, candidate.kind)}, "
                f"x≈{candidate.center_x}/1000, "
                f"{width_label[candidate.qrs_width]}, уверенность обнаружения "
                f"{candidate.confidence}; {evidence}"
            )
        lines.append(
            "Проверь каждый Cn на полном изображении и верни по нему ровно одну запись в "
            "candidate_verdicts, где n — номер Cn: confirmed, если событие действительно "
            "видно; rejected, если это артефакт, разметка или постепенная синусовая "
            "вариабельность; uncertain, если по изображению решить нельзя. Отвергай только "
            "с видимым основанием и коротко назови его. Кандидат без вердикта считается "
            "непроверенным, а не отвергнутым. Не считай одно событие несколько раз. Число "
            "подтвержденных преждевременных комплексов должно совпадать в "
            "observations.rhythm, concise_impression и localized_findings; подтвержденную "
            "паузу или выпадение описывай как нарушение проведения, а не как экстрасистолу."
        )
        return "\n".join(lines)

    @staticmethod
    def _pending_finding(candidate: RhythmCandidate) -> LocalizedFinding:
        """A marker for a candidate the full-sheet pass neither confirmed nor rejected."""
        return LocalizedFinding(
            finding="Кандидат: удлиненный интервал, требует проверки"
            if candidate.kind == "pause"
            else "Кандидат: преждевременный комплекс, требует проверки",
            related_observation="rhythm",
            box=candidate.finding.box,
        )

    @staticmethod
    def _note_unverified_candidates(
        report: ECGReport,
        pending: list[RhythmCandidate],
    ) -> None:
        if not pending:
            return
        positions = ", ".join(str(candidate.center_x) for candidate in pending)
        note = (
            "Мультимасштабный проход выделил непроверенные кандидаты в позициях "
            f"x≈{positions}/1000; они отмечены на изображении и требуют ручной проверки."
        )
        rhythm = report.observations.rhythm
        if note not in rhythm.evidence:
            rhythm.evidence = rhythm.evidence.rstrip(". ") + ". " + note

    @classmethod
    def _reconcile_multiscale_report(
        cls,
        report: ECGReport,
        candidates: list[RhythmCandidate],
    ) -> ECGReport:
        """Reconcile the pre-scan inventory with what the full-sheet pass actually saw.

        The candidates are collected before any clinical judgement, so they may not
        overrule it: only a candidate the full-sheet pass explicitly confirmed changes
        the reported count. A rejected one disappears, and an unjudged one is shown as a
        marked candidate for the doctor instead of being asserted as a finding.
        """
        if not candidates:
            return report

        verdicts = {item.candidate: item.verdict for item in report.candidate_verdicts}
        numbered = list(enumerate(candidates, start=1))
        confirmed = [item for number, item in numbered if verdicts.get(number) == "confirmed"]
        pending = [
            item
            for number, item in numbered
            if verdicts.get(number, "uncertain") == "uncertain"
        ]

        own_events = [
            finding
            for finding in report.localized_findings
            if cls._finding_is_ectopic(report, finding)
        ]
        other_findings = [
            finding
            for finding in report.localized_findings
            if not cls._finding_is_ectopic(report, finding)
        ]
        marked = [item.finding for item in confirmed] or own_events
        marked += [cls._pending_finding(item) for item in pending]
        report.localized_findings = (marked + other_findings)[:3]

        cls._note_unverified_candidates(report, pending)

        count = sum(1 for item in confirmed if item.kind == "premature")
        if not 1 <= count <= 5:
            return report

        # Russian numerals need three agreeing forms here; a single-form table used to
        # produce "с одним преждевременными комплексами" for a single beat.
        count_words = {
            1: ("один", "одним преждевременным комплексом"),
            2: ("два", "двумя преждевременными комплексами"),
            3: ("три", "тремя преждевременными комплексами"),
            4: ("четыре", "четырьмя преждевременными комплексами"),
            5: ("пять", "пятью преждевременными комплексами"),
        }
        nominative, instrumental = count_words[count]
        report_count_text = (
            report.observations.rhythm.value + " " + report.concise_impression
        ).lower()
        count_pattern = rf"(?<!\w)(?:{nominative}|{count})(?!\w)"
        if re.search(count_pattern, report_count_text):
            return report

        current_value = report.observations.rhythm.value
        if "синусов" in current_value.lower():
            rhythm_value = f"Синусовый ритм с {instrumental}"
        else:
            rhythm_value = (
                current_value.rstrip(". ")
                + f"; подтвержденных преждевременных комплексов: {count}"
            )
        positions = ", ".join(
            str(item.center_x) for item in confirmed if item.kind == "premature"
        )
        report.observations.rhythm.value = rhythm_value
        report.observations.rhythm.evidence = (
            report.observations.rhythm.evidence.rstrip(". ")
            + ". Подтверждены раздельные преждевременные временные позиции "
            f"x≈{positions}/1000; число событий: {count}; "
            "происхождение комплексов оценивается отдельно."
        )
        report.concise_impression = rhythm_value + "."
        return report

    async def _analyze_rhythm_view(
        self,
        view: ECGImageView,
        model: str,
    ) -> RhythmScoutResponse:
        cache_key = self._rhythm_scout_cache_key(view.image, model, view)
        if self._cache is not None:
            cached = await asyncio.to_thread(self._cache.get_rhythm_scout, cache_key)
            if cached is not None:
                return cached

        body: dict[str, object] = {
            "model": model,
            "max_tokens": min(
                self._settings.anthropic_max_tokens,
                self._settings.anthropic_scout_max_tokens,
            ),
            "stream": False,
            "system": self._rhythm_scout_system_field(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        self._image_block(view.image),
                        {"type": "text", "text": self._view_analysis_context(view)},
                    ],
                }
            ],
        }
        self._apply_model_controls(
            body,
            effort=self._settings.anthropic_scout_effort,
            schema=RHYTHM_SCOUT_JSON_SCHEMA,
        )

        result: RhythmScoutResponse | None = None
        structure_error: json.JSONDecodeError | ValidationError | None = None
        for attempt in range(2):
            try:
                result = RhythmScoutResponse.model_validate(
                    _extract_json(
                        await self._request_text(
                            body,
                            "поиск комплексов во фрагменте"
                            if attempt == 0
                            else "повторный поиск комплексов во фрагменте",
                        )
                    )
                )
                break
            except (json.JSONDecodeError, ValidationError) as exc:
                structure_error = exc
                if attempt == 0:
                    content = body["messages"][0]["content"]  # type: ignore[index]
                    content.append(  # type: ignore[union-attr]
                        {
                            "type": "text",
                            "text": (
                                "Предыдущий ответ нарушил компактную JSON Schema. "
                                "Верни только компактный объект по схеме с верхним ключом c."
                            ),
                        }
                    )
        if result is None:
            raise AnthropicServiceError(
                "Ответ поиска комплексов не прошел проверку структуры."
            ) from structure_error
        if self._cache is not None:
            await asyncio.to_thread(self._cache.put_rhythm_scout, cache_key, result)
        return result

    async def analyze_multiscale(
        self,
        image: ProcessedImage,
        context: str | None = None,
        model: str | None = None,
    ) -> ECGReport:
        """Inventory overlapping enlarged views, then validate candidates on the full sheet."""
        selected_model = model or self._settings.anthropic_model
        self.usage_stats.analyses += 1
        views = create_rhythm_views(image)

        async def scan(view: ECGImageView) -> tuple[ECGImageView, RhythmScoutResponse]:
            return view, await self._analyze_rhythm_view(view, selected_model)

        scan_results = await asyncio.gather(
            *(scan(view) for view in views),
            return_exceptions=True,
        )
        candidates = self._timing_candidates(image)
        for result in scan_results:
            if isinstance(result, BaseException):
                logger.warning("Enlarged ECG rhythm pass failed: %s", result)
                continue
            view, scan_report = result
            view_candidates = self._rhythm_candidates_from_scout(scan_report, view)
            candidates.extend(view_candidates)
            logger.debug("Rhythm view %s produced %d candidates", view.name, len(view_candidates))
        candidates = self._deduplicate_rhythm_candidates(candidates)
        trusted_context = self._multiscale_context(candidates)
        report = await self.analyze(
            image,
            context,
            model=selected_model,
            analysis_context=trusted_context,
            _record_analysis=False,
        )
        return self._reconcile_multiscale_report(report, candidates)

    async def analyze(
        self,
        image: ProcessedImage,
        context: str | None = None,
        model: str | None = None,
        *,
        analysis_context: str | None = None,
        _record_analysis: bool = True,
    ) -> ECGReport:
        selected_model = model or self._settings.anthropic_model
        if _record_analysis:
            self.usage_stats.analyses += 1
        cache_key = self._cache_key(image, selected_model, context, analysis_context)
        if self._cache is not None:
            cached = await asyncio.to_thread(self._cache.get, cache_key)
            if cached is not None:
                self.usage_stats.cache_hits += 1
                self._apply_clinical_consistency_rules(cached)
                return cached
        user_prompt = build_user_prompt(image.metadata_text(), context, analysis_context)
        body = {
            "model": selected_model,
            "max_tokens": self._settings.anthropic_max_tokens,
            # Some Anthropic-compatible gateways stream by default unless this is explicit.
            "stream": False,
            "system": self._system_field(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            **self._image_block(image),
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                }
            ],
        }
        self._apply_model_controls(
            body,
            effort=self._settings.anthropic_effort,
            schema=ANALYSIS_JSON_SCHEMA,
        )
        result: ECGAnalysisResponse | None = None
        structure_error: json.JSONDecodeError | ValidationError | None = None
        for attempt in range(2):
            try:
                result = ECGAnalysisResponse.model_validate(
                    _extract_json(
                        await self._request_text(
                            body,
                            "анализ изображения"
                            if attempt == 0
                            else "повторный структурированный анализ",
                        )
                    )
                )
                break
            except (json.JSONDecodeError, ValidationError) as exc:
                structure_error = exc
                if attempt == 0:
                    content = body["messages"][0]["content"]  # type: ignore[index]
                    content.append(  # type: ignore[union-attr]
                        {
                            "type": "text",
                            "text": (
                                "Предыдущая попытка этого запроса нарушила JSON Schema. "
                                "Повтори анализ с нуля и верни ровно один полностью валидный "
                                "компактный JSON-объект с короткими ключами из схемы, без "
                                "Markdown, незакрытых строк и второго объекта."
                            ),
                        }
                    )
        if result is None:
            raise AnthropicServiceError(
                "Ответ модели не прошел проверку структуры после повторной попытки."
            ) from structure_error

        facts = self._normalize_image_facts(result.image_facts)
        if not facts.is_ecg_image:
            report = self._non_ecg_report(facts)
            if self._cache is not None:
                await asyncio.to_thread(self._cache.put, cache_key, report)
            return report
        if result.report is None:
            raise AnthropicServiceError("API не вернул отчет для распознанной ЭКГ.")
        report = result.report

        # Facts are generated first in the structured response and then locked here so
        # interpretation text cannot contradict the inventory fields.
        report.image_quality.visible_leads = facts.visible_lead_labels
        report.calibration.raw_text = facts.calibration_text
        report.calibration.speed_mm_s = facts.speed_mm_s
        report.calibration.gain_values_mm_mv = facts.gain_values_mm_mv
        report.calibration.gain_mm_mv = (
            facts.gain_values_mm_mv[0] if len(facts.gain_values_mm_mv) == 1 else None
        )
        report.calibration.source = "Автоматическая проверка изображения"
        if len(facts.gain_values_mm_mv) > 1:
            report.calibration.gain_notes = (
                "На записи указано несколько значений усиления; соответствие группам "
                "отведений не установлено. Амплитудные критерии отключены."
            )
        elif len(facts.gain_values_mm_mv) == 1:
            report.calibration.gain_notes = "На записи указано одно значение усиления."
        else:
            report.calibration.gain_notes = "Значение усиления не распознано."

        report.image_quality.usable_leads = facts.readable_waveform_leads
        report.image_quality.unreadable_leads = facts.unreadable_waveform_leads
        self._remove_locked_fact_conflicts(report, facts, context)
        self._apply_clinical_consistency_rules(report)
        if self._cache is not None:
            await asyncio.to_thread(self._cache.put, cache_key, report)
        return report
