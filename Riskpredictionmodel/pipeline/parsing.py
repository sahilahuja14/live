from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_NO_PAYMENT_MARKERS = {
    "",
    "nan",
    "none",
    "na",
    "#n/a",
    "not received",
    "nil",
    "pending",
}

_FULL_DATE_RE = re.compile(r"\b\d{1,2}[-/ ](?:[A-Za-z]{3,9}|\d{1,2})[-/ ]\d{2,4}\b")
_TEXT_DAY_MONTH_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s*[A-Za-z]{3,9}\b", re.IGNORECASE)
_NUMERIC_DAY_MONTH_RE = re.compile(r"\b\d{1,2}/\d{1,2}\b")


@dataclass(frozen=True)
class PaymentParseResult:
    raw_value: str
    payment_date: pd.Timestamp | pd.NaT
    parsed_dates: tuple[str, ...]
    installment_count: int
    partial_payment_flag: bool
    parsing_status: str


def safe_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def safe_numeric(value) -> float | None:
    text = safe_text(value)
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("%", "").strip()
    if cleaned in {"", "nan", "None"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_main_date(value) -> pd.Timestamp | pd.NaT:
    text = safe_text(value)
    numeric = safe_numeric(value)
    if numeric is not None and numeric > 1000:
        parsed = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).tz_localize(None) if getattr(parsed, "tzinfo", None) else pd.Timestamp(parsed)
    if text is None:
        return pd.NaT
    try:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_localize(None) if getattr(parsed, "tzinfo", None) else pd.Timestamp(parsed)


def _normalize_token(token: str) -> str:
    token = token.strip()
    token = token.replace("&", " ")
    token = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", token, flags=re.IGNORECASE)
    token = re.sub(r"(\d)([A-Za-z])", r"\1 \2", token)
    token = re.sub(r"([A-Za-z])(\d)", r"\1 \2", token)
    token = token.replace(".", " ")
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def _resolve_yearless_date(day: int, month: int, invoice_date: pd.Timestamp | pd.NaT, due_date: pd.Timestamp | pd.NaT) -> pd.Timestamp | pd.NaT:
    reference = due_date if pd.notna(due_date) else invoice_date
    if pd.isna(reference):
        reference = pd.Timestamp.utcnow().normalize().tz_localize(None)
    candidate_years = [reference.year - 1, reference.year, reference.year + 1, reference.year + 2]
    candidates: list[pd.Timestamp] = []
    for year in candidate_years:
        try:
            candidates.append(pd.Timestamp(year=year, month=month, day=day))
        except ValueError:
            continue
    if not candidates:
        return pd.NaT

    invoice_floor = invoice_date - pd.Timedelta(days=60) if pd.notna(invoice_date) else None
    invoice_cap = invoice_date + pd.Timedelta(days=550) if pd.notna(invoice_date) else None

    def _score(ts: pd.Timestamp) -> tuple[float, float]:
        penalty = 0.0
        if invoice_floor is not None and ts < invoice_floor:
            penalty += 365.0
        if invoice_cap is not None and ts > invoice_cap:
            penalty += 180.0
        return penalty, abs((ts - reference).days)

    return min(candidates, key=_score)


def _parse_payment_token(token: str, invoice_date: pd.Timestamp | pd.NaT, due_date: pd.Timestamp | pd.NaT) -> pd.Timestamp | pd.NaT:
    cleaned = _normalize_token(token)
    if not cleaned:
        return pd.NaT

    full = parse_main_date(cleaned)
    if pd.notna(full) and re.search(r"(?:-|/| )\d{2,4}$", cleaned):
        return full

    match = re.match(r"^(?P<day>\d{1,2})[/-](?P<month>\d{1,2})$", cleaned)
    if match:
        return _resolve_yearless_date(
            day=int(match.group("day")),
            month=int(match.group("month")),
            invoice_date=invoice_date,
            due_date=due_date,
        )

    match = re.match(r"^(?P<day>\d{1,2})[- ](?P<month>[A-Za-z]{3,9})$", cleaned)
    if match:
        month = _MONTH_MAP.get(match.group("month").lower())
        if month is None:
            return pd.NaT
        return _resolve_yearless_date(
            day=int(match.group("day")),
            month=month,
            invoice_date=invoice_date,
            due_date=due_date,
        )

    fallback = parse_main_date(cleaned)
    return fallback if pd.notna(fallback) else pd.NaT


def _extract_payment_tokens(raw_value: str) -> list[str]:
    normalized = _normalize_token(raw_value)
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for pattern in (_FULL_DATE_RE, _TEXT_DAY_MONTH_RE, _NUMERIC_DAY_MONTH_RE):
        for match in pattern.finditer(normalized):
            spans.append(match.span())
            tokens.append(match.group(0))
    if tokens:
        return tokens
    return [part.strip() for part in re.split(r"[,&;]+", normalized) if part.strip()]


def parse_payment_value(
    raw_value,
    invoice_date: pd.Timestamp | pd.NaT = pd.NaT,
    due_date: pd.Timestamp | pd.NaT = pd.NaT,
    paid_amount: float | None = None,
    invoice_amount: float | None = None,
) -> PaymentParseResult:
    text = safe_text(raw_value) or ""
    lowered = text.lower().strip()
    if lowered in _NO_PAYMENT_MARKERS:
        return PaymentParseResult(
            raw_value=text,
            payment_date=pd.NaT,
            parsed_dates=tuple(),
            installment_count=0,
            partial_payment_flag=False,
            parsing_status="blank_or_missing",
        )

    numeric = safe_numeric(raw_value)
    if numeric is not None and numeric > 1000:
        parsed_numeric = parse_main_date(numeric)
        return PaymentParseResult(
            raw_value=text,
            payment_date=parsed_numeric,
            parsed_dates=tuple([parsed_numeric.strftime("%Y-%m-%d")]) if pd.notna(parsed_numeric) else tuple(),
            installment_count=1 if pd.notna(parsed_numeric) else 0,
            partial_payment_flag=False,
            parsing_status="parsed" if pd.notna(parsed_numeric) else "unparsed_non_blank",
        )

    tokens = _extract_payment_tokens(text)
    parsed: list[pd.Timestamp] = []
    for token in tokens:
        candidate = _parse_payment_token(token, invoice_date=invoice_date, due_date=due_date)
        if pd.notna(candidate):
            parsed.append(candidate)

    parsed = sorted(set(parsed))
    installment_count = len(parsed)
    partial_from_amount = False
    if invoice_amount is not None and paid_amount is not None and invoice_amount > 0:
        partial_from_amount = float(paid_amount) + 1e-9 < float(invoice_amount)
    partial_payment_flag = installment_count > 1 or partial_from_amount

    if not parsed:
        return PaymentParseResult(
            raw_value=text,
            payment_date=pd.NaT,
            parsed_dates=tuple(),
            installment_count=0,
            partial_payment_flag=partial_payment_flag,
            parsing_status="unparsed_non_blank",
        )

    return PaymentParseResult(
        raw_value=text,
        payment_date=parsed[-1],
        parsed_dates=tuple(ts.strftime("%Y-%m-%d") for ts in parsed),
        installment_count=installment_count,
        partial_payment_flag=partial_payment_flag,
        parsing_status="parsed",
    )
