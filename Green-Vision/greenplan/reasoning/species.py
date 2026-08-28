"""Curated species knowledge base for India-appropriate urban planting.

The reasoning model may ONLY select species from this table — it is passed
verbatim as context and any name outside it is rejected. It must never invent
or override the traits below.

# TODO(you): VERIFY every row against a horticulture source before operational
# use — e.g., your state Forest Department nursery list, ICFRE/FRI guidance, or
# a municipal tree authority. These values (including the soil_ph / soil_texture
# / salinity_tolerance columns) are indicative defaults drafted for your review,
# not verified silviculture data. Soil tolerances especially vary by cultivar
# and rootstock — confirm before planting.
"""

from __future__ import annotations

# columns: common, botanical, native_status (native|naturalized|exotic),
# canopy (small|medium|large), pollution_tolerance (low|medium|high),
# water_need (low|medium|high), context (where it belongs in a city),
# soil_ph (tolerated pH range "lo-hi"), soil_texture (sandy|loamy|clay|any,
# '|'-separated), salinity_tolerance (low|medium|high)
SPECIES_KB: list[dict[str, str]] = [
    {"common": "Neem", "botanical": "Azadirachta indica", "native_status": "native", "canopy": "large", "pollution_tolerance": "high", "water_need": "low", "context": "avenue, park", "soil_ph": "6.0-8.5", "soil_texture": "any", "salinity_tolerance": "high"},
    {"common": "Peepal", "botanical": "Ficus religiosa", "native_status": "native", "canopy": "large", "pollution_tolerance": "high", "water_need": "low", "context": "park, open grounds (aggressive roots: keep off narrow pavements)", "soil_ph": "6.5-8.5", "soil_texture": "any", "salinity_tolerance": "medium"},
    {"common": "Banyan", "botanical": "Ficus benghalensis", "native_status": "native", "canopy": "large", "pollution_tolerance": "high", "water_need": "low", "context": "large park, open grounds only", "soil_ph": "6.5-8.0", "soil_texture": "any", "salinity_tolerance": "medium"},
    {"common": "Pongamia (Karanj)", "botanical": "Millettia pinnata", "native_status": "native", "canopy": "medium", "pollution_tolerance": "high", "water_need": "low", "context": "avenue, road median", "soil_ph": "6.5-8.5", "soil_texture": "any", "salinity_tolerance": "high"},
    {"common": "Jamun", "botanical": "Syzygium cumini", "native_status": "native", "canopy": "large", "pollution_tolerance": "medium", "water_need": "medium", "context": "avenue, park", "soil_ph": "6.0-8.0", "soil_texture": "loamy|clay", "salinity_tolerance": "medium"},
    {"common": "Arjun", "botanical": "Terminalia arjuna", "native_status": "native", "canopy": "large", "pollution_tolerance": "high", "water_need": "medium", "context": "avenue, riverbank, lakefront", "soil_ph": "6.0-8.0", "soil_texture": "loamy|clay", "salinity_tolerance": "medium"},
    {"common": "Gulmohar", "botanical": "Delonix regia", "native_status": "naturalized", "canopy": "medium", "pollution_tolerance": "medium", "water_need": "low", "context": "avenue (brittle limbs: away from parking)", "soil_ph": "6.0-7.5", "soil_texture": "sandy|loamy", "salinity_tolerance": "low"},
    {"common": "Amaltas", "botanical": "Cassia fistula", "native_status": "native", "canopy": "small", "pollution_tolerance": "medium", "water_need": "low", "context": "avenue, median, small streets", "soil_ph": "6.0-8.0", "soil_texture": "any", "salinity_tolerance": "medium"},
    {"common": "Rain Tree", "botanical": "Samanea saman", "native_status": "exotic", "canopy": "large", "pollution_tolerance": "medium", "water_need": "medium", "context": "wide avenue, park", "soil_ph": "6.0-7.5", "soil_texture": "loamy", "salinity_tolerance": "low"},
    {"common": "Sita Ashok", "botanical": "Saraca asoca", "native_status": "native", "canopy": "small", "pollution_tolerance": "low", "water_need": "medium", "context": "shaded park, courtyard", "soil_ph": "5.5-7.0", "soil_texture": "loamy", "salinity_tolerance": "low"},
    {"common": "False Ashoka", "botanical": "Polyalthia longifolia", "native_status": "native", "canopy": "small", "pollution_tolerance": "medium", "water_need": "medium", "context": "boundary screen, narrow verges", "soil_ph": "6.0-7.8", "soil_texture": "loamy", "salinity_tolerance": "low"},
    {"common": "Kadamb", "botanical": "Neolamarckia cadamba", "native_status": "native", "canopy": "large", "pollution_tolerance": "medium", "water_need": "medium", "context": "park, wide avenue", "soil_ph": "5.5-7.5", "soil_texture": "loamy|clay", "salinity_tolerance": "low"},
    {"common": "Mahua", "botanical": "Madhuca longifolia", "native_status": "native", "canopy": "large", "pollution_tolerance": "medium", "water_need": "low", "context": "peri-urban, large park", "soil_ph": "6.0-8.0", "soil_texture": "sandy|loamy", "salinity_tolerance": "medium"},
    {"common": "Bakul (Maulsari)", "botanical": "Mimusops elengi", "native_status": "native", "canopy": "medium", "pollution_tolerance": "high", "water_need": "medium", "context": "avenue, compact streets", "soil_ph": "6.0-7.8", "soil_texture": "loamy", "salinity_tolerance": "medium"},
    {"common": "Son Champa", "botanical": "Magnolia champaca", "native_status": "native", "canopy": "medium", "pollution_tolerance": "low", "water_need": "medium", "context": "park, institutional campus", "soil_ph": "5.5-7.0", "soil_texture": "loamy", "salinity_tolerance": "low"},
    {"common": "Amla", "botanical": "Phyllanthus emblica", "native_status": "native", "canopy": "small", "pollution_tolerance": "medium", "water_need": "low", "context": "park, institutional campus", "soil_ph": "6.0-8.5", "soil_texture": "any", "salinity_tolerance": "medium"},
    {"common": "Bael", "botanical": "Aegle marmelos", "native_status": "native", "canopy": "small", "pollution_tolerance": "medium", "water_need": "low", "context": "institutional campus, park edge", "soil_ph": "6.0-8.5", "soil_texture": "any", "salinity_tolerance": "high"},
    {"common": "Tamarind", "botanical": "Tamarindus indica", "native_status": "naturalized", "canopy": "large", "pollution_tolerance": "high", "water_need": "low", "context": "avenue, park", "soil_ph": "6.0-8.0", "soil_texture": "sandy|loamy", "salinity_tolerance": "medium"},
    {"common": "Indian Cork Tree", "botanical": "Millingtonia hortensis", "native_status": "native", "canopy": "medium", "pollution_tolerance": "medium", "water_need": "low", "context": "avenue (tall, narrow crown)", "soil_ph": "6.0-7.5", "soil_texture": "loamy", "salinity_tolerance": "low"},
    {"common": "Putranjiva", "botanical": "Putranjiva roxburghii", "native_status": "native", "canopy": "medium", "pollution_tolerance": "high", "water_need": "medium", "context": "avenue, boundary screen", "soil_ph": "6.5-8.0", "soil_texture": "loamy", "salinity_tolerance": "medium"},
    {"common": "Semal", "botanical": "Bombax ceiba", "native_status": "native", "canopy": "large", "pollution_tolerance": "medium", "water_need": "medium", "context": "large park, open grounds", "soil_ph": "6.0-7.5", "soil_texture": "sandy|loamy", "salinity_tolerance": "low"},
    {"common": "Palash", "botanical": "Butea monosperma", "native_status": "native", "canopy": "small", "pollution_tolerance": "high", "water_need": "low", "context": "peri-urban, open ground, poor soils", "soil_ph": "6.0-8.5", "soil_texture": "any", "salinity_tolerance": "high"},
    {"common": "Shisham", "botanical": "Dalbergia sissoo", "native_status": "native", "canopy": "large", "pollution_tolerance": "high", "water_need": "low", "context": "avenue, peri-urban belt", "soil_ph": "6.5-8.5", "soil_texture": "sandy|loamy", "salinity_tolerance": "medium"},
    {"common": "Siris", "botanical": "Albizia lebbeck", "native_status": "native", "canopy": "large", "pollution_tolerance": "high", "water_need": "low", "context": "avenue, park", "soil_ph": "6.0-9.0", "soil_texture": "any", "salinity_tolerance": "high"},
    {"common": "Kachnar", "botanical": "Bauhinia variegata", "native_status": "native", "canopy": "small", "pollution_tolerance": "medium", "water_need": "low", "context": "median, small streets", "soil_ph": "6.0-7.5", "soil_texture": "loamy", "salinity_tolerance": "low"},
    {"common": "Jarul (Pride of India)", "botanical": "Lagerstroemia speciosa", "native_status": "native", "canopy": "medium", "pollution_tolerance": "medium", "water_need": "medium", "context": "avenue, park, lakefront", "soil_ph": "5.5-7.5", "soil_texture": "loamy|clay", "salinity_tolerance": "low"},
    {"common": "Moringa", "botanical": "Moringa oleifera", "native_status": "native", "canopy": "small", "pollution_tolerance": "medium", "water_need": "low", "context": "compact streets, home gardens", "soil_ph": "6.3-8.0", "soil_texture": "sandy|loamy", "salinity_tolerance": "medium"},
    {"common": "Jackfruit", "botanical": "Artocarpus heterophyllus", "native_status": "native", "canopy": "large", "pollution_tolerance": "low", "water_need": "medium", "context": "park, institutional campus", "soil_ph": "6.0-7.5", "soil_texture": "loamy", "salinity_tolerance": "low"},
    {"common": "Mango", "botanical": "Mangifera indica", "native_status": "native", "canopy": "large", "pollution_tolerance": "medium", "water_need": "medium", "context": "park, wide avenue", "soil_ph": "5.5-7.5", "soil_texture": "loamy", "salinity_tolerance": "low"},
    {"common": "Copper Pod", "botanical": "Peltophorum pterocarpum", "native_status": "exotic", "canopy": "large", "pollution_tolerance": "high", "water_need": "low", "context": "avenue, park", "soil_ph": "6.0-8.0", "soil_texture": "sandy|loamy", "salinity_tolerance": "high"},
]

_COLUMNS = [
    "common", "botanical", "native_status", "canopy", "pollution_tolerance",
    "water_need", "context", "soil_ph", "soil_texture", "salinity_tolerance",
]


def kb_names() -> list[str]:
    return [s["common"] for s in SPECIES_KB]


def kb_by_name() -> dict[str, dict[str, str]]:
    return {s["common"]: s for s in SPECIES_KB}


def kb_markdown_table() -> str:
    lines = ["| " + " | ".join(_COLUMNS) + " |", "|" + "---|" * len(_COLUMNS)]
    for s in SPECIES_KB:
        lines.append("| " + " | ".join(s[c] for c in _COLUMNS) + " |")
    return "\n".join(lines)


def validate_selection(names: list[str], hi: int = 5) -> list[str]:
    """Keep only exact KB names, deduplicated, clipped to at most `hi`."""
    valid = set(kb_names())
    seen: list[str] = []
    for n in names:
        if n in valid and n not in seen:
            seen.append(n)
    return seen[:hi]
