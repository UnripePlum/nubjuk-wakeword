from __future__ import annotations

import hashlib
import re

_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3

_CHO = [
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
]
_JUNG = [
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "eu",
    "ui",
    "i",
]
_JONG = [
    "",
    "k",
    "k",
    "ks",
    "n",
    "nj",
    "nh",
    "t",
    "l",
    "lk",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "p",
    "ps",
    "t",
    "t",
    "ng",
    "t",
    "t",
    "k",
    "t",
    "p",
    "h",
]


def _is_hangul_syllable(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return _HANGUL_BASE <= cp <= _HANGUL_LAST


def _romanize_hangul_syllable(ch: str) -> str:
    code = ord(ch) - _HANGUL_BASE
    cho = code // 588
    jung = (code % 588) // 28
    jong = code % 28
    return f"{_CHO[cho]}{_JUNG[jung]}{_JONG[jong]}"


def _normalize_slug(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def target_word_to_slug(target_word: str, *, explicit_slug: str | None = None) -> str:
    if explicit_slug and explicit_slug.strip():
        normalized = _normalize_slug(explicit_slug.strip())
        if normalized:
            return normalized

    text = target_word.strip()
    if not text:
        return "wakeword"

    if all(ch.isascii() for ch in text):
        normalized = _normalize_slug(text)
        if normalized:
            return normalized

    romanized_parts: list[str] = []
    for ch in text:
        if _is_hangul_syllable(ch):
            romanized_parts.append(_romanize_hangul_syllable(ch))
        elif ch.isascii():
            romanized_parts.append(ch.lower())
        elif ch.isspace():
            romanized_parts.append("_")

    normalized = _normalize_slug("".join(romanized_parts))
    if normalized:
        return normalized

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
    return f"wakeword_{digest}"
