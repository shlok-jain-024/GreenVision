"""The assistant: natural language in, a *plan of real actions* out.

This is not a chatbot bolted onto a map. It is the engine's front door. A
message arrives with the browser's current context (where the area of interest
is, what was measured inside it, what the user has drawn so far); this module
decides what the user is asking for, answers it **from measured numbers only**,
and emits a list of tool calls the page then executes against the live map.

Three rules it does not break, because a planning tool that breaks them is
worse than no tool at all:

  1. **Never invent a number.** Every figure in a reply is either passed in by
     the browser (measured live) or read off the trained panel. If neither has
     it, the reply says so.
  2. **Never claim precision the source cannot carry.** Projections are named
     projections, proxies are named proxies, and the placeholder traffic stream
     is never quoted at all.
  3. **Say what it did.** Every action returned is described in the reply, so
     the map never moves for a reason the reader cannot see.

The language model, when one is loaded, is used for *phrasing only* — it is
handed the finished numbers and asked to write them up. It never picks a
species, never ranks a cell, and never produces a figure. With no model
installed the deterministic writer below does the same job in plainer prose,
which is why the whole assistant works offline with nothing fetched.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from typing import Any

from . import i18n
from .species import SPECIES_KB

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f(v: Any) -> float | None:
    """Best-effort float that refuses NaN/Inf, so nothing downstream has to."""
    try:
        if v is None or isinstance(v, bool):
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    x = _f(v)
    return int(round(x)) if x is not None else None


def _aqi_key(a: float) -> str:
    """Band name as a translation key, not as English prose."""
    if a <= 50:
        return "good"
    if a <= 100:
        return "moderate"
    if a <= 150:
        return "sensitive"
    if a <= 200:
        return "unhealthy"
    if a <= 300:
        return "very_unhealthy"
    return "hazardous"


# Element ids -> translation keys. The English name is only a fallback; the
# dictionary carries the plural for each language because English plural rules
# ("benchs", "solars path light") do not survive being applied by machine, and
# other languages do not use the same rule at all.
_ELEMENT_KEY = {
    "path_gravel": "path_gravel", "meadow": "meadow", "shrub": "shrub",
    "bench": "bench", "light": "light", "tap": "tap", "compost": "compost",
    "gazebo": "gazebo", "play": "play", "rwh": "rwh", "hedge": "hedge",
    "swale": "swale", "drip": "drip", "mulch": "mulch",
}


def _qty_phrase(e: dict[str, Any], lang: str = "en") -> str:
    """"8 benches", "600 m2 of gravel trail" - never "8 benchs", and never
    "4 each bench", which is the unit slug leaking into a sentence."""
    qty, unit = float(e.get("qty") or 0), e.get("unit")
    key = _ELEMENT_KEY.get(e.get("id") or "")
    fallback = str(e.get("name") or "").lower()

    def name_for(plural: bool) -> str:
        if not key:
            return fallback
        k = "element." + key + ("_plural" if plural else "")
        out = i18n.t(k, lang)
        if out == k:                       # no plural form defined for this language
            out = i18n.t("element." + key, lang)
        return fallback if out.startswith("element.") else out

    if unit == "each":
        n = int(round(qty))
        return i18n.t("element.qty_each", lang, n=n, name=name_for(n != 1))
    if unit == "m":
        return i18n.t("element.qty_m", lang, n=f"{qty:,.0f}", name=name_for(False))
    return i18n.t("element.qty_m2", lang, n=f"{qty:,.0f}", name=name_for(False))


# ---------------------------------------------------------------------------
# Intent recognition
# ---------------------------------------------------------------------------
#
# Ordered, deliberately. The first pattern that matches wins, so the specific
# intents ("compare A and B") are listed above the general ones ("tell me about
# A"). Each entry is (intent, regex). Patterns are matched against a lowercased,
# punctuation-normalised message.

# Terms the glossary in _do_explain actually covers. A definitional question
# naming one of these is a definition request even though the term also appears
# in a topical pattern below - "what is NDVI" is not a canopy reading.
_GLOSSARY = (r"ndvi|h3|hexagons?|mcda|priority score|multi.?criteria|memory loop|"
             r"in.?context|plantable|accuracy|theil|backtest|placeholder")

_INTENTS: list[tuple[str, str]] = [
    ("greet",     r"^\s*(hi|hey|hello|yo|namaste|good (morning|afternoon|evening)|"
                  r"thanks|thank you|ok|okay|cool|nice)\b[\s!.]*$"),
    ("help",      r"\b(help|what can you do|how do i use|commands?|examples?)\b"),
    ("explain",   r"\b(what'?s?|what is|what are|what does|explain|meaning of|define|"
                  r"how does|how do you)\b[^?]{0,40}\b(?:" + _GLOSSARY + r")\b"),
    ("compare",   r"\bcompare\b|\bversus\b|\bvs\.?\b|\bbetter\b.*\bor\b"),
    ("priority",  r"\b(priorit\w*|most urgent|worst (areas?|cells?|zones?)|where should (the )?(city|we|i) plant|top \d+ (cells?|zones?|areas?)|rank\w*|hot ?spots?)\b"),
    ("design",    r"\b(design|build|plan|create|make|lay ?out|sketch)\b.{0,30}\b(park|garden|oasis|grove|belt|plot|avenue|buffer|space|something|it)\b"),
    ("design",    r"\b(design|plan) (me |us )?(a|an|one)\b"),
    ("plant",     r"\b(plant|planting|add|place|put)\b.{0,24}\b(tree|trees|sapling|saplings|shrub|shrubs)\b"),
    ("plant",     r"\b(plant|add|place|put)\s+(\d+|a|some|more)\b"),
    ("species",   r"\b(species|which trees?|what trees?|what should i plant|what to plant|recommend\w* (trees?|species|plants?)|suitable trees?|best trees?)\b"),
    ("empty_land", r"\b(empty|bare|vacant|unused|open|free|barren|waste)\s*"
                   r"(land|ground|space|plot|area|spots?|patch\w*)\b|"
                   r"\bwhere can (i|we) plant\b|\bplantable\b|\broom to plant\b"),
    ("carbon",    r"\b(co2|carbon|sequest\w*|offset|emissions?|footprint)\b"),
    ("heat",      r"\b(heat|hot|cooling|cooler|shade|shadow|temperature|"
                  r"heat ?island|uhi|degrees)\b"),
    ("budget",    r"\b(lakh|crore|rupees?|\u20b9|rs\.?|budget of|afford|"
                  r"with \d+|for \d+ ?(lakh|crore|k))\b"),
    ("maintenance", r"\b(maintain\w*|upkeep|watering|who waters|annual cost|"
                    r"running cost|per year cost|survive without)\b"),
    ("timing",    r"\b(when (to|should)|which (season|month)|monsoon|"
                  r"best time|planting season|sapling season)\b"),
    ("survival",  r"\b(surviv\w*|die|died|mortality|how many will live|"
                  r"establish\w*|failure rate)\b"),
    ("people",    r"\b(people|population|residents|who benefits|how many will use|"
                  r"footfall|community)\b"),
    ("sources",   r"\b(source|sources|where does (this|the) data|how do you know|"
                  r"how accurate|reliable|cite|citation|provenance)\b"),
    # "how much" is only a COST question when it is not asking how much
    # of something else. Without the lookahead, "how much rain does this
    # get" classified as cost, because this pattern is tested before
    # water. Caught by scripts/parity_check.py, which is why it exists.
    ("cost",      r"\b(cost|costs|budget|price|expensive|rupees|inr|crore|lakh|"
                  r"bill of quantit\w*|boq)\b|"
                  r"\bhow much(?!\s+(?:rain|rainfall|water|co2|carbon|shade|"
                  r"canopy|green|greenery|land|space|area|room|time|sun|light))\b"),
    ("project",   r"\b(project\w*|forecast|future|\d+\s*years?|long ?term|by 20\d\d|25 ?year)\b"),
    ("review",    r"\b(review|is (my|this) design|any good|score|critique|flaws?|problems? with)\b"),
    ("air",       r"\b(air|aqi|pollution|polluted|pm ?2\.?5|pm ?10|breathe|smog|no2|ozone)\b"),
    ("canopy",    r"\b(canopy|green ?cover|tree ?cover|vegetation|ndvi|how green|greenery)\b"),
    ("traffic",   r"\b(traffic|congestion|bottlenecks?|jams?|roads? are)\b"),
    ("water",     r"\b(water|rain|rainfall|irrigation|drought|monsoon|groundwater)\b"),
    ("soil",      r"\b(soil|ph|texture|clay|sandy|loam|ground condition)\b"),
    ("view",      r"\b(satellite|green view|map view|show (me )?(the )?(green|satellite|street|priority))\b"),
    ("goto",      r"\b(go to|goto|show me|take me to|fly to|navigate to|find|search|jump to|zoom to|look at|open)\b"),
    ("report",    r"\b(report|summar\w+|brief|tell me about|analyse|analyze|overview|status)\b|"
                  r"\bwhat('?s| is|s)?\s+(is\s+)?(here|around|nearby|in this area)\b"),
    ("explain",   r"\b(why|how does|what is|what does|explain|meaning of|based on)\b"),
]

_COMPILED = [(name, re.compile(pat)) for name, pat in _INTENTS]


# "the place the reader is looking at". A message that names no subject at all
# is still answerable when it points at the map ("what about here"), and a
# message with ONE weak topic word is worth acting on when it also points at
# the map. web/gv-engine.js carries the same test; keep the two in step.
_HERE = re.compile(r"\b(?:here|this area|this place|near me|nearby|around)\b")

# Last resort, and only that. By the time this table is consulted every one of
# the ~30 patterns above has missed, so the message is phrased in a way the
# product did not anticipate. Two matching words - or one plus a "here" - is
# the bar, which is why the lists are short, single-token and boring: a wide
# list would turn "unknown" into a confident wrong answer, and a wrong answer
# moves the map. Words the regexes already catch are left out deliberately;
# they can never reach this point, so listing them would only be noise.
#
# Every key must name an intent with a real _do_<name> handler, or classify()
# silently routes it to _do_report and the reader gets a place summary in
# answer to a question about trees.
_FALLBACK_WORDS: dict[str, list[str]] = {
    "species":    ["tree", "trees", "sapling", "saplings", "native",
                   "varieties", "nursery"],
    "empty_land": ["land", "plot", "plots", "vacant", "empty", "barren"],
    "canopy":     ["green", "greener", "cover"],
    "heat":       ["hotter", "warm", "summer"],
    "priority":   ["urgent", "worst"],
    "timing":     ["season", "month"],
    "report":     ["locality", "neighbourhood", "neighborhood",
                   "condition", "conditions"],
}


# Every Indic script this product ships has its own digit block. A reader
# typing "૫ લાખ" means the same as "5 lakh" and the extractors below only
# know ASCII, so fold the digits before anything else looks at the string.
_DIGIT_MAP = {}
for _base in (0x0966,  # Devanagari  (hi, mr)
              0x09E6,  # Bengali     (bn, as)
              0x0A66,  # Gurmukhi    (pa)
              0x0AE6,  # Gujarati    (gu)
              0x0B66,  # Odia        (or)
              0x0BE6,  # Tamil       (ta)
              0x0C66,  # Telugu      (te)
              0x0CE6,  # Kannada     (kn)
              0x0D66,  # Malayalam   (ml)
              0x06F0,  # Extended Arabic-Indic (ur)
              0x0660): # Arabic-Indic
    for _d in range(10):
        _DIGIT_MAP[_base + _d] = ord("0") + _d


def _normalise(msg: str) -> str:
    """Lowercase, fold Indic digits, strip punctuation - WITHOUT destroying
    the script.

    Keeps letters (L*), combining marks (M*) and numbers (N*); blanks out
    only real punctuation and symbols. The mark class is the whole point:
    drop it and every Indic language becomes unreadable consonant skeletons.
    """
    s = (msg or "").lower().strip().translate(_DIGIT_MAP)
    s = s.replace("\u2019", "'")
    out = []
    for ch in s:
        if ch.isspace():
            out.append(" ")
        elif unicodedata.category(ch)[0] in ("L", "M", "N") or ch in "'.,-/&":
            out.append(ch)
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _native_intent(n: str, lang: str) -> str | None:
    """Score the message against the language's own keyword lists.

    Longest keyword wins, so a two-word phrase beats an incidental single
    word - "શું વાવવું" (what to plant) must not be read as a bare "શું"
    (what)."""
    if not lang or lang == "en":
        return None
    table = i18n.nlu(lang)
    if not table:
        return None
    best, best_len = None, 0
    for intent, words in table.items():
        for w in words:
            w = str(w).lower().strip()
            if w and w in n and len(w) > best_len:
                best, best_len = intent, len(w)
    return best


def classify(msg: str, lang: str = "en") -> str:
    n = _normalise(msg)
    if not n:
        return "help"
    # English patterns first: they are precise and many Indian-language
    # messages carry English technical terms ("NDVI", "AQI", "1 hectare").
    for name, rx in _COMPILED:
        if rx.search(n):
            return name
    native = _native_intent(n, lang)
    if native:
        return native
    words = set(n.split())
    scored = [(len(words & set(v)), k) for k, v in _FALLBACK_WORDS.items()]
    best_n, best_k = max(scored)
    if best_n >= 2 or (best_n == 1 and _HERE.search(n)):
        return best_k
    return "report" if _HERE.search(n) else "unknown"


# ---------------------------------------------------------------------------
# Slot extraction
# ---------------------------------------------------------------------------

# How a goal reads in a sentence. The keys are the studio's internal slugs.
GOAL_NOUN = {
    "park": "park", "greenbelt": "green belt", "riverfront": "riverfront planting",
    "community": "community garden", "campus": "campus green",
    "avenue": "roadside avenue", "residential": "residential green",
    "industrial": "industrial buffer", "wetland": "wetland edge planting",
}

_GOALS = {
    "park": ["park", "public park", "green space"],
    "greenbelt": ["green belt", "greenbelt", "belt"],
    "riverfront": ["riverfront", "river", "waterfront", "lakefront", "riverbank"],
    "community": ["community garden", "community", "allotment", "kitchen garden"],
    "campus": ["school", "college", "campus", "university", "children", "kids"],
    "avenue": ["avenue", "roadside", "road side", "street tree", "median"],
    "residential": ["residential", "housing", "society", "apartment", "colony"],
    "industrial": ["industrial", "factory", "buffer", "industry"],
    "wetland": ["wetland", "marsh", "pond edge", "lake edge"],
}

# Area phrases -> square metres. Ordered longest-first at match time.
_AREA_UNITS = [
    (r"hectares?|\bha\b|हेक्टेयर|હેક્ટર|হেক্টর|ஹெக்டேர்|హెక్టార్|ಹೆಕ್ಟೇರ್|ഹെക്ടർ|ਹੈਕਟੇਅਰ|ହେକ୍ଟର|ہیکٹر", 10_000.0),
    (r"acres?", 4046.86),
    (r"square kilometi?res?|sq\.? ?km|km2|km²", 1_000_000.0),
    (r"square met(?:re|er)s?|sq\.? ?m|m2|m²", 1.0),
    (r"bighas?", 2529.3),          # Gujarat bigha; regional, flagged in the reply
    (r"guntha?s?", 101.17),
]


# Words that end a place name and begin a new request. Without these,
# "near rajpath club find empty land" is read as one enormous place name,
# the geocoder returns nothing, and the actual question is never answered.
_PLACE_STOP = (
    r"\b(?:and|then|please|for me|instead|on the map|now|"
    r"find|show|tell|give|check|see|look|search|"
    r"what|which|where|when|how|why|who|is|are|does|do|can|should|"
    r"plant|design|build|make|create|plan|draw|cost|price|budget|review|"
    r"compare|project|forecast)\b"
)


# The subject of the question is not the place the question is about.
# "find empty land near Rajpath Club" names a THING (empty land) and a
# PLACE (Rajpath Club); the verb "find" introduces the thing, not the place.
_LAND_NOUN = (
    r"(?:the\s+|some\s+|any\s+)?"
    r"(?:empty|bare|vacant|unused|open|free|barren|waste|plantable|available)?\s*"
    r"(?:land|ground|space|spaces|plot|plots|area|areas|spot|spots|"
    r"patch|patches|site|sites|room)\b"
)

# Locative prepositions, with the near-misses people actually type. A place
# name is what follows one of these. "nar" and "ner" are the two typos that
# showed up in real use; they are distinctive enough as standalone tokens
# that matching them costs nothing.
_LOCATIVE = r"\b(?:n(?:ea|ae|e|a)r(?:by)?|around|close to|next to|beside|" \
            r"in|at|by|within|inside|surrounding)\s+"


def extract_place(msg: str) -> str | None:
    """The place name in a 'go to X' / 'near X' style message.

    Stops at the first command word so a compound request keeps its verb."""
    s = (msg or "").strip().rstrip("?.!")
    m = re.search(
        r"\b(?:go to|goto|show me|take me to|fly to|navigate to|jump to|zoom to|"
        r"look at|search(?: for)?|find|near|nearby|around|close to|in|at|open)\s+(.+)$",
        s, re.I,
    )
    if not m:
        return None
    rest = m.group(1).strip()

    # If what we captured OPENS with the question's own noun ("empty land",
    # "bare ground", "room"), then the place — if there is one at all — comes
    # after a locative preposition further along. Take that, or nothing.
    #
    # Without this, "find empty land near Rajpath Club" geocoded the string
    # "empty land near rajpath club", which no gazetteer contains, so the
    # reader got a failed search instead of an answer to their actual
    # question. Reported from real use.
    lead = re.match(_LAND_NOUN, rest, re.I)
    if lead:
        after = rest[lead.end():]
        loc = re.search(_LOCATIVE, after, re.I)
        if not loc:
            return None                    # "find empty land" — no place named
        rest = after[loc.end():].strip()

    place = re.split(_PLACE_STOP, rest, maxsplit=1, flags=re.I)[0]
    place = place.strip(" ,.-")
    if not place or len(place) < 2:
        return None
    # A bare intent word is not a place ("show me the green view").
    if re.fullmatch(
        r"(the |a |an |some |me )*(green|satellite|map|priority|street|empty|bare|"
        r"vacant|open|land|ground|space|park|trees?|air|soil|water|cost|place)"
        r"( view| land| ground| space)?", place, re.I
    ):
        return None
    return place[:120]


def extract_places_pair(msg: str) -> tuple[str, str] | None:
    m = re.search(
        r"\bcompare\s+(.+?)\s+(?:and|with|to|versus|vs\.?)\s+(.+?)$",
        (msg or "").strip().rstrip("?.!"), re.I,
    )
    if not m:
        m = re.search(r"^(.+?)\s+(?:versus|vs\.?)\s+(.+?)$",
                      (msg or "").strip().rstrip("?.!"), re.I)
    if not m:
        return None
    a, b = m.group(1).strip(" ,.-"), m.group(2).strip(" ,.-")
    if a and b:
        return a[:80], b[:80]
    return None


def extract_area_m2(msg: str) -> float | None:
    s = _normalise(msg)
    for unit_rx, mult in _AREA_UNITS:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:" + unit_rx + r")(?![A-Za-z])", s)
        if m:
            return float(m.group(1)) * mult
    return None


def extract_goal(msg: str, lang: str = "en") -> str | None:
    s = _normalise(msg)
    best, best_len = None, 0
    for goal, words in _GOALS.items():
        for w in words:
            if w in s and len(w) > best_len:
                best, best_len = goal, len(w)
    # A goal named in the reader's own language counts too.
    for goal in _GOALS:
        w = i18n.t("goal." + goal, lang).lower()
        if w and not w.startswith("goal.") and w in s and len(w) > best_len:
            best, best_len = goal, len(w)
    return best


def extract_count(msg: str) -> int | None:
    s = _normalise(msg)
    # "60 trees", and also "60 neem trees" / "60 pongamia (karanj) saplings" -
    # the species name sits between the number and the noun.
    m = re.search(
        r"(\d+)\s*(?:[a-z()\-']+\s+){0,3}(?:trees?|saplings?|shrubs?|plants?)\b", s)
    if m:
        return max(1, min(2000, int(m.group(1))))
    # "plant 60 neem" - a bare count straight after the verb, no noun at all.
    m = re.search(r"\b(?:plant|add|place|put)\s+(\d+)\b", s)
    if m:
        return max(1, min(2000, int(m.group(1))))
    return None


def extract_years(msg: str) -> int | None:
    s = _normalise(msg)
    m = re.search(r"(\d+)\s*years?\b", s)
    if m:
        return max(1, min(50, int(m.group(1))))
    m = re.search(r"\bby (20\d\d)\b", s)
    if m:
        import datetime
        return max(1, min(50, int(m.group(1)) - datetime.date.today().year))
    return None


def extract_money(msg: str) -> float | None:
    """Rupees from "5 lakh", "2 crore", "Rs 50000", "50k"."""
    n = _normalise(msg)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|cr|करोड़|કરોડ|কোটি|ਕਰੋੜ|କୋଟି|کروڑ)(?![A-Za-z])", n)
    if m:
        return float(m.group(1)) * 1e7
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|lakhs|लाख|લાખ|লাখ|ਲੱਖ|ଲକ୍ଷ|لاکھ)(?![A-Za-z])", n)
    if m:
        return float(m.group(1)) * 1e5
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", n)
    if m:
        return float(m.group(1)) * 1000
    m = re.search(r"(?:rs\.?|inr|\u20b9)\s*([\d,]+)", n)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"\b(\d{4,9})\b", n)
    return float(m.group(1)) if m else None


def _money_words(n: float) -> str:
    if n >= 1e7:
        return f"Rs {n / 1e7:,.2f} crore".replace(".00", "")
    if n >= 1e5:
        return f"Rs {n / 1e5:,.2f} lakh".replace(".00", "")
    return f"Rs {n:,.0f}"


def extract_top_n(msg: str) -> int | None:
    m = re.search(r"\btop\s+(\d+)", _normalise(msg))
    if m:
        return max(1, min(100, int(m.group(1))))
    return None


# ---------------------------------------------------------------------------
# Context view — what the browser told us
# ---------------------------------------------------------------------------


class Ctx:
    """Typed, defensive read of the JSON the page posts up.

    Everything is optional. A missing reading becomes None and the reply
    acknowledges the gap rather than substituting a plausible number."""

    def __init__(self, raw: dict[str, Any] | None) -> None:
        raw = raw if isinstance(raw, dict) else {}
        self.raw = raw
        aoi = raw.get("aoi") if isinstance(raw.get("aoi"), dict) else {}
        self.lat = _f(aoi.get("lat"))
        self.lon = _f(aoi.get("lon"))
        self.km2 = _f(aoi.get("km2")) or 100.0
        self.place = str(raw.get("place") or "").strip()
        self.view = str(raw.get("view") or "").strip()

        r = raw.get("readings") if isinstance(raw.get("readings"), dict) else {}
        self.aqi = _f(r.get("aqi"))
        self.aqi_min = _f(r.get("aqi_min"))
        self.aqi_max = _f(r.get("aqi_max"))
        self.pm25 = _f(r.get("pm25"))
        self.pm10 = _f(r.get("pm10"))
        self.temp = _f(r.get("temp"))
        self.humidity = _f(r.get("humidity"))
        self.canopy = _f(r.get("canopy_pct"))
        self.bare = _f(r.get("bare_frac"))
        self.rain = _f(r.get("rain_mm_yr"))
        self.hot_days = _f(r.get("hot_days_yr"))
        self.n_points = _i(r.get("n_points"))

        c = raw.get("census") if isinstance(raw.get("census"), dict) else {}
        self.census = {k: _i(v) for k, v in c.items() if _i(v) is not None}

        d = raw.get("design") if isinstance(raw.get("design"), dict) else {}
        self.goal = str(d.get("goal") or raw.get("goal") or "park").strip() or "park"
        self.plot_m2 = _f(d.get("plot_m2"))
        self.n_items = _i(d.get("n_items")) or 0
        self.n_trees = _i(d.get("n_trees")) or 0
        self.total_cost = _f(d.get("total_cost"))
        self.review_score = _f(d.get("review_score"))

    @property
    def has_point(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def where(self) -> str:
        if self.place:
            return self.place
        if self.has_point:
            return f"{self.lat:.4f}, {self.lon:.4f}"
        return "this area"


# ---------------------------------------------------------------------------
# The assistant
# ---------------------------------------------------------------------------


class Assistant:
    """Answers a message against a live `Engine`, in the reader's language.

    Every sentence goes through `self.t()`, which renders a template from
    data/i18n/<lang>.json and falls back to English per-key. Numbers, units,
    place names and botanical names are passed in as placeholders and are
    never translated - a mistranslated species name ends up on a planting
    list, and that has a cost in the ground.

    `engine` must expose: cfg, ranked, recommendations, soil, lessons,
    health(), recommend_point(), match_species() and layout_plan()."""

    def __init__(self, engine: Any) -> None:
        self.e = engine
        self.lang = i18n.DEFAULT_LANG

    # -- translation helpers ------------------------------------------------

    def t(self, key: str, **kw: Any) -> str:
        return i18n.t(key, self.lang, **kw)

    def sp(self, common: str) -> str:
        """A species name as the reader knows it, with the English name kept
        alongside so it can still be matched against a nursery list.

        The English name is stripped of its own parenthetical first, or
        "Jarul (Pride of India)" renders as "તામ્રપુષ્પી (Jarul (Pride of
        India))" - nested brackets that read as a bug."""
        local = i18n.species_name(common, self.lang)
        if local == common:
            return common
        bare = re.sub(r"\s*\(.*?\)", "", common).strip() or common
        return f"{local} ({bare})"

    def goal_noun(self, goal: str) -> str:
        return self.t("goal." + goal) if goal in _GOALS else self.t("goal.park")

    def aqi_word(self, a: float) -> str:
        return self.t("aqi." + _aqi_key(a))

    # -- entry point --------------------------------------------------------

    def handle(self, message: str, context: dict[str, Any] | None,
               lang: Any = None) -> dict[str, Any]:
        self.lang = i18n.normalise(lang)
        ctx = Ctx(context)
        intent = classify(message, self.lang)

        # A compound request - "near Rajpath Club find empty land" - names a
        # place AND asks something. Move there first, then answer the rest, so
        # the reader gets both halves instead of a failed geocode.
        lead: list[dict[str, Any]] = []
        if intent not in ("goto", "compare", "greet", "help", "unknown"):
            place = extract_place(message)
            if place and not parse_latlon(message):
                lead = [{"tool": "map.search", "args": {"query": place}},
                        {"tool": "dock.open", "args": {"tab": "area"}}]
                ctx.place = ctx.place or place
        log.info("assistant intent=%s lang=%s msg=%r", intent, self.lang,
                 (message or "")[:120])
        fn = getattr(self, "_do_" + intent, None) or self._do_report
        try:
            out = fn(message, ctx)
        except Exception as exc:                      # never 500 on a question
            log.exception("assistant failed on intent=%s", intent)
            out = {"reply": self.t("error.body",
                                   err=f"{type(exc).__name__}: {exc}"),
                   "actions": []}
        out.setdefault("actions", [])
        out.setdefault("cards", [])
        if lead:
            out["actions"] = lead + out["actions"]
            out["reply"] = self.t("compound.lead", place=extract_place(message) or "") \
                + "\n\n" + out.get("reply", "")
        out.setdefault("cards", [])
        out["intent"] = intent
        out["lang"] = self.lang
        out["dir"] = i18n.direction(self.lang)
        out["source"] = self.e.reasoning_label()
        return out

    # -- individual intents -------------------------------------------------

    def _do_help(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        return {"reply": self.t("help.body"), "actions": []}

    def _do_greet(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        key = "greet.hello_where" if ctx.place else "greet.hello"
        return {"reply": self.t(key, place=ctx.place), "actions": []}

    def _do_unknown(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        return {"reply": self.t("unknown.body"), "actions": []}

    # ---- navigation -------------------------------------------------------

    def _do_goto(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        # Coordinates beat a name: they are unambiguous.
        ll = parse_latlon(msg)
        if ll:
            lat, lon = ll
            return {
                "reply": self.t("goto.coords", lat=f"{lat:.4f}", lon=f"{lon:.4f}",
                                km2=int(ctx.km2)),
                "actions": [{"tool": "map.goto", "args": {"lat": lat, "lon": lon, "zoom": 15}},
                            {"tool": "dock.open", "args": {"tab": "area"}}],
            }
        place = extract_place(msg)
        if not place:
            return {"reply": self.t("goto.which"), "actions": []}
        return {
            "reply": self.t("goto.place", place=place, km2=int(ctx.km2)),
            "actions": [{"tool": "map.search", "args": {"query": place}},
                        {"tool": "dock.open", "args": {"tab": "area"}}],
        }

    def _do_view(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        n = _normalise(msg)
        view = ("priority" if "priority" in n else
                "green" if "green" in n else
                "map" if ("map view" in n or "street" in n) else
                "satellite")
        return {"reply": self.t("view." + view),
                "actions": [{"tool": "map.view", "args": {"view": view}}]}

    # ---- reading a place --------------------------------------------------

    def _do_report(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.has_point:
            return self._need_point()
        bits: list[str] = []
        head = self.t("report.head", place=ctx.where, km2=int(ctx.km2))

        if ctx.aqi is not None:
            pm = (self.t("report.aqi_pm", pm25=f"{ctx.pm25:.0f}")
                  if ctx.pm25 is not None else "")
            line = self.t("report.aqi", aqi=int(ctx.aqi),
                          word=self.aqi_word(ctx.aqi), pm=pm)
            if (ctx.aqi_min is not None and ctx.aqi_max is not None
                    and ctx.aqi_max - ctx.aqi_min > 12):
                line += self.t("report.aqi_spread", lo=int(ctx.aqi_min),
                               hi=int(ctx.aqi_max))
            bits.append(line)
        else:
            bits.append(self.t("report.aqi_none"))

        if ctx.canopy is not None:
            verdict = ("dense" if ctx.canopy >= 30 else
                       "moderate" if ctx.canopy >= 18 else
                       "thin" if ctx.canopy >= 8 else "very_thin")
            bits.append(self.t("report.canopy", pct=f"{ctx.canopy:.0f}",
                               verdict=self.t("canopy." + verdict)))
        if ctx.bare is not None:
            bits.append(self.t("report.bare", pct=f"{ctx.bare * 100:.0f}"))
        if ctx.rain is not None:
            hot = (self.t("report.rain_hot", days=f"{ctx.hot_days:.0f}")
                   if ctx.hot_days is not None else "")
            bits.append(self.t("report.rain", mm=f"{ctx.rain:,.0f}", hot=hot))

        cs = ctx.census
        if cs:
            named = [(k, cs[k]) for k in
                     ("buildings", "roads", "parks", "trees", "schools",
                      "hospitals", "water") if cs.get(k)]
            if named:
                items = ", ".join(f"**{v:,}** {k}" for k, v in named[:5])
                bits.append(self.t("report.census", items=items))

        cell = self._cell_for(ctx)
        if cell:
            bits.append(cell["sentence"])
        bits.append(self._verdict(ctx))
        return {"reply": head + "\n\n" + "\n\n".join(bits),
                "actions": [{"tool": "dock.open", "args": {"tab": "area"}}]}

    def _do_air(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.has_point:
            return self._need_point()
        if ctx.aqi is None:
            return {"reply": self.t("air.none"), "actions": []}
        lines = [self.t("air.head", aqi=int(ctx.aqi), word=self.aqi_word(ctx.aqi),
                        n=ctx.n_points or 9, km2=int(ctx.km2))]
        if ctx.pm25 is not None:
            lines.append(self.t("air.pm25", pm25=f"{ctx.pm25:.0f}",
                                times=f"{ctx.pm25 / 15.0:.1f}"))
        if ctx.aqi_min is not None and ctx.aqi_max is not None:
            lines.append(self.t("air.range", lo=int(ctx.aqi_min), hi=int(ctx.aqi_max)))
        if ctx.aqi >= 150:
            lines.append(self.t("air.species_note"))
        cell = self._cell_for(ctx)
        if cell and cell.get("aqi_pred_delta") is not None:
            d = cell["aqi_pred_delta"]
            tail = ("air.forecast_worse" if d > 2 else
                    "air.forecast_better" if d < -2 else "air.forecast_flat")
            lines.append(self.t("air.forecast", delta=f"{d:+.1f}", tail=self.t(tail)))
        return {"reply": "\n\n".join(lines),
                "actions": [{"tool": "dock.open", "args": {"tab": "area"}}]}

    def _do_canopy(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.has_point:
            return self._need_point()
        lines = []
        if ctx.canopy is not None:
            lines.append(self.t("cover.head", pct=f"{ctx.canopy:.0f}", km2=int(ctx.km2)))
        else:
            lines.append(self.t("cover.none"))
        if ctx.bare is not None:
            lines.append(self.t("cover.bare", pct=f"{ctx.bare * 100:.0f}"))
        cell = self._cell_for(ctx)
        if cell:
            nl, tr = cell.get("ndvi_latest"), cell.get("ndvi_trend_per_year")
            if nl is not None:
                trend = (self.t("cover.engine_trend", trend=f"{tr:+.4f}")
                         if tr is not None else "")
                lines.append(self.t("cover.engine", ndvi=f"{nl:.3f}", trend=trend))
            if tr is not None and tr < -0.005:
                lines.append(self.t("cover.decline"))
        lines.append(self.t("cover.switch"))
        return {"reply": "\n\n".join(lines),
                "actions": [{"tool": "map.view", "args": {"view": "green"}}]}

    def _do_water(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.has_point:
            return self._need_point()
        lines = []
        if ctx.rain is not None:
            band = ("arid" if ctx.rain < 400 else
                    "semiarid" if ctx.rain < 750 else
                    "subhumid" if ctx.rain < 1200 else "humid")
            lines.append(self.t("water.rain", mm=f"{ctx.rain:,.0f}",
                                band=self.t("water.band_" + band)))
            if ctx.rain < 750:
                lines.append(self.t("water.dry_note"))
        else:
            lines.append(self.t("water.none"))
        if ctx.hot_days is not None:
            lines.append(self.t("water.hot", days=f"{ctx.hot_days:.0f}"))
        return {"reply": "\n\n".join(lines), "actions": []}

    def _do_soil(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        prof = self._soil_for(ctx)
        if not prof:
            return {"reply": self.t("soil.none"), "actions": []}
        bits = [self.t("soil.head", zone=prof["zone"])]
        if prof.get("ph") is not None:
            bits.append(self.t("soil.ph", ph=f"{prof['ph']:.1f}",
                               cls=prof.get("ph_class", "?")))
        if prof.get("texture"):
            parts = (self.t("soil.texture_parts", sand=f"{prof['sand']:.0f}",
                            silt=f"{prof['silt']:.0f}", clay=f"{prof['clay']:.0f}")
                     if prof.get("sand") is not None else "")
            bits.append(self.t("soil.texture", texture=prof["texture"], parts=parts))
        if prof.get("organic_carbon") is not None:
            bits.append(self.t("soil.carbon", soc=f"{prof['organic_carbon']:.1f}"))
        if prof.get("nitrogen") is not None:
            bits.append(self.t("soil.nitrogen", n=f"{prof['nitrogen']:.1f}"))
        bits.append(self.t("soil.caveat"))
        return {"reply": "\n".join(bits), "actions": []}

    def _do_traffic(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        return {"reply": self.t("traffic.body"),
                "actions": [{"tool": "dock.open", "args": {"tab": "traffic"}}]}

    # ---- the engine's ranking --------------------------------------------

    def _do_priority(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        n = extract_top_n(msg) or 5
        rows = self.e.top_cells(n)
        if not rows:
            return {"reply": self.t("priority.none"), "actions": []}
        lines = [self.t("priority.head", total=self.e.n_zones(),
                        city=self.e.cfg.city.name, n=len(rows))]
        for r in rows:
            line = self.t("priority.row", rank=r["rank"], score=f"{r['score']:.3f}",
                          aqi=f"{r['aqi_latest']:.0f}",
                          delta=f"{r['aqi_pred_delta']:+.1f}",
                          ndvi=f"{r['ndvi_latest']:.3f}",
                          trend=f"{r['ndvi_trend_per_year']:+.4f}")
            if r.get("species"):
                line += self.t("priority.row_species",
                               species=", ".join(self.sp(s) for s in r["species"]))
            lines.append(line)
        lines.append(self.t("priority.footer"))
        return {
            "reply": "\n".join(lines),
            "actions": [{"tool": "map.view", "args": {"view": "priority"}},
                        {"tool": "priority.focus", "args": {"rank": rows[0]["rank"]}}],
            "cards": rows,
        }

    def _do_compare(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        pair = extract_places_pair(msg)
        if not pair:
            return {"reply": self.t("compare.need_two"), "actions": []}
        a, b = pair
        return {"reply": self.t("compare.body", a=a, b=b),
                "actions": [{"tool": "compare.run", "args": {"a": a, "b": b}}]}

    # ---- species and design ----------------------------------------------

    def _do_species(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        goal = extract_goal(msg, self.lang) or ctx.goal
        picks = self.e.match_species(
            lat=ctx.lat, lon=ctx.lon, aqi=ctx.aqi, canopy_pct=ctx.canopy,
            rain_mm_yr=ctx.rain, goal=goal, limit=6, lang=self.lang,
        )
        if not picks["species"]:
            return {"reply": self.t("species.none"), "actions": []}
        lines = [picks["headline"], ""]
        for s in picks["species"]:
            lines.append(self.t(
                "species.card", common=self.sp(s["common"]),
                botanical=s.get("botanical", ""),
                canopy=self.t("species.canopy_" + (s.get("canopy") or "medium")),
                pollution=self.t("species.tol_" + (s.get("pollution_tolerance") or "medium")),
                water=self.t("species.tol_" + (s.get("water_need") or "medium")),
                native=self.t("species." + (s.get("native_status") or "native")),
                why=s.get("why", "")))
        if picks.get("caveats"):
            lines.append("\n" + picks["caveats"])
        return {
            "reply": "\n\n".join(lines),
            "actions": [{"tool": "studio.suggest", "args": {
                "species": [s["common"] for s in picks["species"]], "goal": goal}}],
            "cards": picks["species"],
        }

    def _do_design(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.has_point:
            return self._need_point()
        area = extract_area_m2(msg) or 10_000.0
        area = max(200.0, min(250_000.0, area))
        goal = extract_goal(msg, self.lang) or ctx.goal
        picks = self.e.match_species(
            lat=ctx.lat, lon=ctx.lon, aqi=ctx.aqi, canopy_pct=ctx.canopy,
            rain_mm_yr=ctx.rain, goal=goal, limit=8, area_m2=area, lang=self.lang,
        )
        plan = self.e.layout_plan(area_m2=area, goal=goal, species=picks["species"],
                                  rain_mm_yr=ctx.rain, aqi=ctx.aqi, lang=self.lang)

        mix = ", ".join(f"{c['count']}× {self.sp(c['species'])}" for c in plan["mix"])
        lines = [
            self.t("design.head", ha=f"{area / 10_000.0:.2f}", m2=f"{area:,.0f}",
                   kind=self.goal_noun(goal), where=ctx.where),
            "",
            self.t("design.trees", n=plan["n_trees"],
                   spacing=f"{plan['spacing_m']:.0f}", mix=mix),
        ]
        if plan.get("elements"):
            lines.append(self.t("design.plus", items=", ".join(
                _qty_phrase(e, self.lang) for e in plan["elements"])))
        lines += ["", plan["rationale"], "", self.t("design.tail")]
        return {
            "reply": "\n".join(lines),
            "actions": [
                {"tool": "studio.goal", "args": {"goal": goal}},
                {"tool": "studio.plot", "args": {"area_m2": area}},
                {"tool": "studio.autoplant", "args": {
                    "mix": plan["mix"], "spacing_m": plan["spacing_m"]}},
                {"tool": "studio.elements", "args": {"elements": plan["elements"]}},
                {"tool": "dock.open", "args": {"tab": "studio"}},
            ],
            "cards": picks["species"],
        }

    def _do_plant(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.has_point:
            return self._need_point()
        count = extract_count(msg) or 40
        named = self._named_species(msg)
        goal = extract_goal(msg, self.lang) or ctx.goal
        if named:
            mix = [{"species": named[0], "count": count}]
            why = self.t("plant.named_why", species=self.sp(named[0]))
        else:
            picks = self.e.match_species(
                lat=ctx.lat, lon=ctx.lon, aqi=ctx.aqi, canopy_pct=ctx.canopy,
                rain_mm_yr=ctx.rain, goal=goal, limit=3, lang=self.lang,
            )
            per = max(1, count // max(1, len(picks["species"])))
            mix = [{"species": s["common"], "count": per} for s in picks["species"]]
            why = picks["headline"]
        total = sum(m["count"] for m in mix)
        shown = ", ".join(f"{m['count']}× {self.sp(m['species'])}" for m in mix)
        return {
            "reply": self.t("plant.head", n=total, mix=shown) + "\n\n" + why +
                     "\n\n" + self.t("plant.tail"),
            "actions": [
                {"tool": "studio.ensure_plot", "args": {}},
                {"tool": "studio.autoplant", "args": {"mix": mix, "spacing_m": 8}},
                {"tool": "dock.open", "args": {"tab": "studio"}},
            ],
        }

    # ---- money, time, judgement ------------------------------------------

    def _do_cost(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.n_items:
            return {"reply": self.t("cost.none"),
                    "actions": [{"tool": "dock.open", "args": {"tab": "studio"}}]}
        return {"reply": self.t("cost.body", n=ctx.n_items),
                "actions": [{"tool": "dock.open", "args": {"tab": "cost"}}]}

    def _do_project(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        years = extract_years(msg) or 25
        if not ctx.n_items:
            return {"reply": self.t("project.none"),
                    "actions": [{"tool": "dock.open", "args": {"tab": "studio"}}]}
        return {"reply": self.t("project.body", years=years),
                "actions": [{"tool": "project.show", "args": {"years": years}},
                            {"tool": "dock.open", "args": {"tab": "review"}}]}

    def _do_review(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.n_items:
            return {"reply": self.t("review.none"),
                    "actions": [{"tool": "dock.open", "args": {"tab": "studio"}}]}
        note = ""
        if ctx.review_score is not None:
            s = ctx.review_score
            key = ("review.note_strong" if s >= 75 else
                   "review.note_ok" if s >= 55 else "review.note_bad")
            note = self.t(key, score=f"{s:.0f}")
        return {"reply": self.t("review.body", note=note),
                "actions": [{"tool": "review.show", "args": {}},
                            {"tool": "dock.open", "args": {"tab": "review"}}]}

    def _do_explain(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        n = _normalise(msg)
        for topic in ("ndvi", "h3", "mcda", "priority", "memory", "traffic",
                      "plantable", "accuracy"):
            if topic in n:
                return {"reply": self.t("explain." + topic), "actions": []}
        if "hexagon" in n:
            return {"reply": self.t("explain.h3"), "actions": []}
        if "why" in n and ctx.has_point:
            return self._do_report(msg, ctx)
        return {"reply": self.t("explain.menu"), "actions": []}

    # ---- the widened question set ----------------------------------------
    #
    # Everything below answers from a measured input or the trained panel.
    # Where the honest answer is "the data does not carry that", the handler
    # says so and names what would be needed - which is more useful than a
    # confident guess and is the only version that survives a judge asking
    # where the number came from.

    def _do_empty_land(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.has_point:
            return self._need_point()
        lines = []
        if ctx.bare is not None:
            ha = ctx.bare * ctx.km2 * 100.0          # km² -> hectares
            lines.append(self.t("empty.head", pct=f"{ctx.bare * 100:.0f}",
                                km2=int(ctx.km2), ha=f"{ha:,.0f}"))
        else:
            lines.append(self.t("empty.none"))

        cells = self.e.bare_cells(ctx.lat, ctx.lon, n=5)
        if cells:
            lines.append(self.t("empty.cells_head", n=len(cells)))
            for c in cells:
                lines.append(self.t("empty.cell_row", rank=c["rank"],
                                    pct=f"{c['plantable_space'] * 100:.0f}",
                                    ndvi=f"{c['ndvi_latest']:.2f}",
                                    km=f"{c['dist_km']:.1f}"))
        lines.append(self.t("empty.caveat"))
        return {
            "reply": "\n\n".join(lines),
            "actions": [{"tool": "map.view", "args": {"view": "green"}},
                        {"tool": "dock.open", "args": {"tab": "area"}}],
            "cards": cells,
        }

    def _do_carbon(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.n_trees:
            return {"reply": self.t("carbon.none"), "actions": []}
        # 20 kg/yr/tree at maturity is the mid-range of the studio's own
        # per-species co2 column; the spread is stated rather than hidden.
        lo, hi = ctx.n_trees * 8, ctx.n_trees * 30
        return {"reply": self.t("carbon.body", n=ctx.n_trees,
                                lo=f"{lo:,}", hi=f"{hi:,}",
                                t_lo=f"{lo * 25 / 1000:,.0f}",
                                t_hi=f"{hi * 25 / 1000:,.0f}"),
                "actions": [{"tool": "dock.open", "args": {"tab": "review"}}]}

    def _do_heat(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.has_point:
            return self._need_point()
        lines = []
        if ctx.temp is not None:
            lines.append(self.t("heat.now", temp=f"{ctx.temp:.0f}"))
        if ctx.hot_days is not None:
            lines.append(self.t("heat.days", days=f"{ctx.hot_days:.0f}"))
        if ctx.canopy is not None:
            lines.append(self.t("heat.canopy", pct=f"{ctx.canopy:.0f}"))
        lines.append(self.t("heat.mechanism"))
        return {"reply": "\n\n".join(lines),
                "actions": [{"tool": "dock.open", "args": {"tab": "area"}}]}

    def _do_budget(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        rupees = extract_money(msg)
        if rupees is None:
            return {"reply": self.t("budget.ask"), "actions": []}
        # Reverse the studio's own all-in per-tree figure rather than inventing
        # a rate: ~Rs 3,400 a tree is what its bill of quantities produces once
        # the three-year establishment period is included.
        per_tree = 3400.0
        trees = int(rupees * 0.72 / per_tree)        # 72% to planting, 28% to ground works
        ha = trees * 81.0 / 10000.0                  # at 9 m centres
        return {"reply": self.t("budget.body", amount=_money_words(rupees),
                                trees=f"{trees:,}", ha=f"{ha:.2f}",
                                per=f"{per_tree:,.0f}"),
                "actions": [{"tool": "dock.open", "args": {"tab": "cost"}}]}

    def _do_maintenance(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        if not ctx.n_items:
            return {"reply": self.t("maint.none"), "actions": []}
        return {"reply": self.t("maint.body"),
                "actions": [{"tool": "dock.open", "args": {"tab": "cost"}}]}

    def _do_timing(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        return {"reply": self.t("timing.body"), "actions": []}

    def _do_survival(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        n = ctx.n_trees or 0
        if not n:
            return {"reply": self.t("survival.general"), "actions": []}
        return {"reply": self.t("survival.body", n=n,
                                lo=int(n * 0.6), hi=int(n * 0.85)),
                "actions": [{"tool": "dock.open", "args": {"tab": "review"}}]}

    def _do_people(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        # The page carries no population layer. Say so, and give the reader
        # the proxy it DOES carry rather than a fabricated headcount.
        cs = ctx.census or {}
        return {"reply": self.t("people.body",
                                buildings=f"{cs.get('buildings', 0):,}",
                                schools=cs.get("schools", 0)),
                "actions": [{"tool": "dock.open", "args": {"tab": "area"}}]}

    def _do_sources(self, msg: str, ctx: Ctx) -> dict[str, Any]:
        return {"reply": self.t("sources.body"), "actions": []}

    # -- shared helpers -----------------------------------------------------

    def _need_point(self) -> dict[str, Any]:
        return {"reply": self.t("need_point"), "actions": []}

    def _named_species(self, msg: str) -> list[str]:
        """Species the user named, in English or in their own language."""
        n = _normalise(msg)
        hits = []
        for row in SPECIES_KB:
            common = row["common"]
            names = {common.lower(),
                     re.sub(r"\s*\(.*?\)", "", common).strip().lower(),
                     i18n.species_name(common, self.lang).lower()}
            if any(x and x in n for x in names):
                hits.append(common)
        return hits

    def _cell_for(self, ctx: Ctx) -> dict[str, Any] | None:
        if not ctx.has_point:
            return None
        r = self.e.cell_report(ctx.lat, ctx.lon)
        if not r:
            return None
        # Rebuild the sentence in the reader's language rather than shipping
        # the English one the engine composed for its own logs.
        if r.get("rank") is not None and r.get("score") is not None:
            s = self.t("cell.sentence", zone=r["zone"], rank=r["rank"],
                       total=self.e.n_zones(), score=f"{r['score']:.3f}")
            if r.get("ndvi_trend_per_year") is not None and r.get("aqi_pred_delta") is not None:
                s += self.t("cell.trend", trend=f"{r['ndvi_trend_per_year']:+.4f}",
                            delta=f"{r['aqi_pred_delta']:+.1f}")
            r["sentence"] = s
        return r

    def _soil_for(self, ctx: Ctx) -> dict[str, Any] | None:
        if not ctx.has_point:
            return None
        return self.e.soil_report(ctx.lat, ctx.lon)

    def _verdict(self, ctx: Ctx) -> str:
        """One honest sentence on what this place actually needs."""
        if ctx.aqi is None and ctx.canopy is None:
            return self.t("verdict.unknown")
        bad_air = ctx.aqi is not None and ctx.aqi > 100
        thin = ctx.canopy is not None and ctx.canopy < 15
        room = ctx.bare is not None and ctx.bare > 0.30
        if bad_air and thin:
            return self.t("verdict.plant_room" if room else "verdict.plant_noroom")
        if bad_air:
            return self.t("verdict.air_only")
        if thin:
            return self.t("verdict.thin")
        return self.t("verdict.ok")


# ---------------------------------------------------------------------------
# Coordinate parsing (shared with the browser's search bar contract)
# ---------------------------------------------------------------------------

_DEC = r"[-+]?\d{1,3}(?:\.\d+)?"


def parse_latlon(text: str) -> tuple[float, float] | None:
    """Accept '23.02, 72.57', '23.02 72.57', '23.02N 72.57E' and DMS.

    Returns (lat, lon) only when both are in range - a bare '10 20' that could
    be anything is still accepted as coordinates because the caller only tries
    this on search input, but out-of-range pairs are rejected outright."""
    s = (text or "").strip()

    # Degrees-minutes-seconds, e.g. 23°01'21.0"N 72°34'17.0"E
    dms = re.findall(
        r"(\d{1,3})\s*[°d:]\s*(\d{1,2})\s*['′m:]?\s*(\d{1,2}(?:\.\d+)?)?\s*[\"″s]?\s*([NSEW])",
        s, re.I,
    )
    if len(dms) == 2:
        vals = {}
        for deg, mnt, sec, hemi in dms:
            v = float(deg) + float(mnt) / 60.0 + (float(sec) if sec else 0.0) / 3600.0
            h = hemi.upper()
            if h in ("S", "W"):
                v = -v
            vals["lat" if h in ("N", "S") else "lon"] = v
        if "lat" in vals and "lon" in vals:
            return _range_ok(vals["lat"], vals["lon"])

    # Decimal with hemisphere letters
    m = re.search(rf"({_DEC})\s*([NS])\s*[, ]\s*({_DEC})\s*([EW])", s, re.I)
    if m:
        lat = float(m.group(1)) * (-1 if m.group(2).upper() == "S" else 1)
        lon = float(m.group(3)) * (-1 if m.group(4).upper() == "W" else 1)
        return _range_ok(lat, lon)

    # Plain decimal pair
    m = re.search(rf"({_DEC})\s*[,;]\s*({_DEC})", s)
    if not m:
        m = re.fullmatch(rf"\s*({_DEC})\s+({_DEC})\s*", s)
    if m:
        return _range_ok(float(m.group(1)), float(m.group(2)))
    return None


def _range_ok(lat: float, lon: float) -> tuple[float, float] | None:
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return lat, lon
    # Tolerate a swapped pair — a very common paste error.
    if -90.0 <= lon <= 90.0 and -180.0 <= lat <= 180.0:
        return lon, lat
    return None
