"""Translation for the assistant's own prose.

The dictionaries live in `data/i18n/<code>.json` and are shared with the
browser, which loads the same files for its interface chrome. One source of
truth, so the two halves of the product cannot drift into saying different
things in the same language.

Three rules this module holds to:

  * **English is a language, not a special case.** The English strings sit in
    `en.json` alongside the rest and go through the same lookup, so a missing
    key is visible in development rather than silently working.
  * **Numbers never get translated.** Templates carry named placeholders and
    the caller formats them in. Digits, units, currency and botanical names
    pass through untouched — a mistranslated species name reaches a planting
    list, and that is a real-world cost.
  * **A missing translation falls back to English rather than to nothing.**
    A part-translated reply is readable; a blank one is not.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "i18n"

# code -> loaded dict, or None when the file is absent/unreadable. Only codes
# that survive the validation in `_load` are ever admitted; see the note there.
_CACHE: dict[str, dict[str, Any] | None] = {}

DEFAULT_LANG = "en"


def available() -> list[dict[str, str]]:
    """The language registry, from data/i18n/index.json."""
    try:
        idx = json.loads((_DIR / "index.json").read_text(encoding="utf-8"))
        return idx.get("languages", [])
    except Exception:
        return [{"code": "en", "name": "English", "native": "English", "dir": "ltr"}]


def _load(code: str) -> dict[str, Any] | None:
    # Defend the path FIRST, and never let a rejected code near the cache.
    # `code` arrives from an HTTP body — /api/assistant and /api/species both
    # forward the caller's `lang` here verbatim. Testing the cache before
    # validating, and then storing the rejects, turned this module-level dict
    # into unbounded storage keyed by whatever a caller cared to send: nothing
    # evicts it and nothing bounds the key. Re-running this test on every bad
    # request costs a string scan; caching it cost the process.
    if not code or not code.replace("-", "").isalnum() or len(code) > 8:
        return None
    if code in _CACHE:
        return _CACHE[code]
    # A well-formed code we have no file for still caches its None, so a
    # missing translation is one failed read rather than one per request.
    path = _DIR / (code + ".json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = None
    _CACHE[code] = data
    return data


def normalise(code: Any) -> str:
    """Accept 'gu', 'gu-IN', 'GU' and land on a language we actually ship."""
    c = str(code or "").strip().lower().replace("_", "-")
    if not c:
        return DEFAULT_LANG
    if _load(c) is not None:
        return c
    base = c.split("-")[0]
    return base if _load(base) is not None else DEFAULT_LANG


def t(key: str, lang: str = DEFAULT_LANG, **kw: Any) -> str:
    """Render template `key` in `lang`, falling back to English per-key.

    Formatting failures never raise: a template with a placeholder the caller
    did not supply returns the English rendering, and if that also fails, the
    raw template. A reply that reads oddly beats a 500."""
    for code in (lang, DEFAULT_LANG):
        data = _load(code)
        if not data:
            continue
        tpl = (data.get("assistant") or {}).get(key)
        if not tpl:
            continue
        try:
            return tpl.format(**kw) if kw else tpl
        except (KeyError, IndexError, ValueError):
            log.warning("i18n: template %r in %r does not fit its arguments", key, code)
            continue
    return key


def species_name(common: str, lang: str = DEFAULT_LANG) -> str:
    """Local common name for a species, or the English one unchanged.

    Never touches the botanical name — that is the unambiguous key a nursery
    or a forest department list is actually ordered against."""
    data = _load(lang)
    if not data:
        return common
    return (data.get("species") or {}).get(common) or common


def nlu(lang: str = DEFAULT_LANG) -> dict[str, list[str]]:
    """Intent -> keywords in this language's own script.

    Empty when a language file has no `nlu` block yet, in which case the
    English patterns and the scored fallback still run."""
    data = _load(lang)
    if not data:
        return {}
    table = data.get("nlu") or {}
    return table if isinstance(table, dict) else {}


def direction(lang: str = DEFAULT_LANG) -> str:
    for row in available():
        if row.get("code") == lang:
            return row.get("dir", "ltr")
    return "ltr"
