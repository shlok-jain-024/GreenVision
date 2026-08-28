/* =====================================================================
   Green Vision — client-side engine and assistant
   =====================================================================
   Loaded only by the STATIC build (Cloudflare Pages, GitHub Pages, or a
   plain file server). When greenplan.server is running, none of this is
   used: the Python assistant answers and this file stands down.

   Why it exists
   -------------
   Cloudflare Pages serves files. It will not run pandas, numpy and h3.
   But almost nothing the page asks the engine for is dynamic — for a
   fixed city the ranking, the forecast, the soil table and the species
   knowledge base are constants, and scripts/build_static.py bakes them
   to engine/*.json.

   The one live part is the assistant, and it is deterministic: no model,
   no weights, just intent matching and planning. So it ports.

   What this is NOT
   ----------------
   It is not a second, competing implementation of the studio. Every
   action it emits is a tool call the page already executes for the
   Python assistant, against the same GV.design state, so anything it
   builds stays editable and costs, reviews and projects identically.

   Honest scope
   ------------
   This covers the intents people actually type. It is deliberately
   narrower than greenplan/reasoning/assistant.py — the Python keeps
   compound place-extraction, the full glossary, comparison and several
   long-tail intents. Where this cannot answer, it says so and names the
   command that starts the full engine, rather than inventing a reply.
   `source` on every answer says which brain produced it, and the UI
   prints that, so the difference is never hidden from the reader.
   ===================================================================== */

(function () {
"use strict";

const GVE = window.GVE = { data: null, loaded: false, loading: null };

/* ---------- baked engine payloads ---------------------------------- */

const FILES = {
  cells:     "engine/cells.json",
  greenloss: "engine/greenloss.json",
  soil:      "engine/soil.json",
  species:   "engine/species.json",
  meta:      "engine/meta.json"
};

async function loadJSON(u) {
  try {
    const r = await fetch(u);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) { return null; }
}

/* Load once, share the promise so eight simultaneous questions cause one
   set of requests rather than eight. */
GVE.load = function () {
  if (GVE.loaded) return Promise.resolve(GVE.data);
  if (GVE.loading) return GVE.loading;
  GVE.loading = (async () => {
    const keys = Object.keys(FILES);
    const vals = await Promise.all(keys.map(k => loadJSON(FILES[k])));
    const d = {};
    keys.forEach((k, i) => (d[k] = vals[i]));
    GVE.data = d;
    GVE.loaded = !!(d.cells && d.cells.length);
    return d;
  })();
  return GVE.loading;
};

/* ---------- geometry: which baked cell is this point in? ------------
   The Python side calls h3.latlng_to_cell. Shipping an H3 library to do
   that in the browser would be ~100 KB for one function, and we already
   have every cell's polygon in greenloss.json — so: point in polygon,
   exact, over at most a few hundred hexagons. */

function pointInRing(lat, lon, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];      // GeoJSON is [lon, lat]
    const xj = ring[j][0], yj = ring[j][1];
    if ((yi > lat) !== (yj > lat) &&
        lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function cellAt(lat, lon) {
  const gl = GVE.data && GVE.data.greenloss;
  if (!gl || !gl.features) return null;
  for (const f of gl.features) {
    if (!f.geometry || !f.geometry.coordinates) continue;
    if (pointInRing(lat, lon, f.geometry.coordinates[0])) return f.properties.zone;
  }
  return null;
}

function rowFor(zone) {
  const cells = (GVE.data && GVE.data.cells) || [];
  return cells.find(c => c.zone === zone) || null;
}

function km(aLat, aLon, bLat, bLon) {
  const R = 6371, dLat = (bLat - aLat) * Math.PI / 180, dLon = (bLon - aLon) * Math.PI / 180;
  const s = Math.sin(dLat / 2) ** 2 +
            Math.cos(aLat * Math.PI / 180) * Math.cos(bLat * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

/* ---------- the engine surface the assistant needs ------------------ */

/* The engine's own zone count, not the number of rows that survived the
   finite-score filter. cells.json omits cells with no real NDVI coverage
   (they cannot be ranked), so counting its rows under-reports the panel —
   132 instead of 146 — and disagrees with what the Python assistant says
   about the same city. meta.zones is the engine's figure; fall back to the
   row count only if meta failed to load. */
GVE.nZones = function () {
  const m = GVE.data && GVE.data.meta;
  if (m && isFinite(m.zones)) return m.zones;
  return ((GVE.data && GVE.data.cells) || []).length;
};

/* How many of those the browser can actually show a score for. Used where
   the distinction matters, so neither number is ever quietly wrong. */
GVE.nRanked = () => ((GVE.data && GVE.data.cells) || []).length;

GVE.cellReport = function (lat, lon) {
  if (lat == null || lon == null) return null;
  const z = cellAt(lat, lon);
  return z ? rowFor(z) : null;
};

GVE.soilReport = function (lat, lon) {
  if (lat == null || lon == null) return null;
  const z = cellAt(lat, lon);
  if (!z) return null;
  const s = GVE.data.soil && GVE.data.soil[z];
  return s ? Object.assign({ zone: z }, s) : null;
};

GVE.topCells = function (n) {
  const cells = ((GVE.data && GVE.data.cells) || []).slice();
  cells.sort((a, b) => a.rank - b.rank);
  return cells.slice(0, Math.max(1, Math.min(100, n || 5)));
};

/* Ported from Engine.bare_cells in server.py, which is canonical.

   This used to take the 40 emptiest cells in the city FIRST and only then
   sort them by distance, so a 46%-plantable cell across the road was thrown
   away whenever forty emptier cells existed anywhere else — and the two
   builds answered "where is there empty land" with different cells. Python
   keeps every cell over the 0.45 bar and orders by distance, emptiest first
   only as the tiebreak. */
GVE.bareCells = function (lat, lon, n) {
  const cells = ((GVE.data && GVE.data.cells) || [])
    .filter(c => c.lat != null && c.plantable_space != null &&
                 isFinite(c.plantable_space) && c.plantable_space >= 0.45);
  // Python sets dist_km to 0.0 for every cell when there is no point, so the
  // tiebreak governs. Always assign, or a stale _km from a previous call
  // with a point would survive into a call without one.
  const hasPt = lat != null && lon != null;
  cells.forEach(c => (c._km = hasPt ? km(lat, lon, c.lat, c.lon) : 0));
  cells.sort((a, b) => (a._km - b._km) || (b.plantable_space - a.plantable_space));
  return cells.slice(0, Math.max(1, Math.min(20, n || 5)));
};

/* ---------- text helpers ------------------------------------------- */

const nf = n => Number(n).toLocaleString("en-IN");
const pm = (v, d) => (v > 0 ? "+" : "") + Number(v).toFixed(d == null ? 1 : d);

/* Six bands, same boundaries as _aqi_key in reasoning/assistant.py — all
   `<=`, and everything past 300 is hazardous. The JS stopped at five and
   called a 340 reading "very unhealthy" while the Python called it
   "hazardous"; Ahmedabad reaches that range, so the two builds were
   describing the same number differently. */
function aqiWord(a) {
  return a <= 50 ? "good" : a <= 100 ? "moderate" :
         a <= 150 ? "unhealthy for sensitive groups" :
         a <= 200 ? "unhealthy" :
         a <= 300 ? "very unhealthy" : "hazardous";
}

function inr(n) {
  if (n >= 1e7) return "₹" + (n / 1e7).toFixed(2).replace(/\.00$/, "") + " crore";
  if (n >= 1e5) return "₹" + (n / 1e5).toFixed(2).replace(/\.00$/, "") + " lakh";
  return "₹" + Math.round(n).toLocaleString("en-IN");
}

/* ---------- intent classification ----------------------------------
   Ported from greenplan/reasoning/assistant.py::_INTENTS. Order is
   load-bearing: the first match wins, so the specific patterns must
   precede the general ones exactly as they do in the Python. */

const INTENTS = [
  ["greet",     /^\s*(hi|hey|hello|yo|namaste|good (morning|afternoon|evening)|thanks|thank you|ok|okay|cool|nice)\b[\s!.]*$/],
  ["help",      /\b(help|what can you do|how do i use|commands?|examples?)\b/],
  ["compare",   /\bcompare\b|\bversus\b|\bvs\.?\b/],
  ["priority",  /\b(priorit\w*|most urgent|worst (areas?|cells?|zones?)|where should (the )?(city|we|i) plant|top \d+ (cells?|zones?|areas?)|rank\w*|hot ?spots?)\b/],
  ["design",    /\b(design|build|plan|create|make|lay ?out|sketch)\b.{0,30}\b(park|garden|oasis|grove|belt|plot|avenue|buffer|space|something|it)\b/],
  ["design",    /\b(design|plan) (me |us )?(a|an|one)\b/],
  ["plant",     /\b(plant|planting|add|place|put)\b.{0,24}\b(tree|trees|sapling|saplings|shrub|shrubs)\b/],
  ["plant",     /\b(plant|add|place|put)\s+(\d+|a|some|more)\b/],
  ["species",   /\b(species|which trees?|what trees?|what should i plant|what to plant|recommend\w* (trees?|species|plants?)|suitable trees?|best trees?)\b/],
  ["empty_land",/\b(empty|bare|vacant|unused|open|free|barren|waste)\s*(land|ground|space|plot|area|spots?|patch\w*)\b|\bwhere can (i|we) plant\b|\bplantable\b|\broom to plant\b/],
  // "how much" is only a cost question when it is not asking how much of
  // something else — see the matching comment in assistant.py.
  ["cost",      /\b(cost|costs|budget|price|expensive|rupees|inr|crore|lakh|bill of quantit\w*|boq)\b|\bhow much(?!\s+(?:rain|rainfall|water|co2|carbon|shade|canopy|green|greenery|land|space|area|room|time|sun|light))\b/],
  ["project",   /\b(project\w*|forecast|future|\d+\s*years?|long ?term|by 20\d\d|25 ?year)\b/],
  ["review",    /\b(review|is (my|this) design|any good|score|critique|flaws?|problems? with)\b/],
  ["air",       /\b(air|aqi|pollution|polluted|pm ?2\.?5|pm ?10|breathe|smog|no2|ozone)\b/],
  ["canopy",    /\b(canopy|green ?cover|tree ?cover|vegetation|ndvi|how green|greenery)\b/],
  ["traffic",   /\b(traffic|congestion|bottlenecks?|jams?)\b/],
  ["water",     /\b(water|rain|rainfall|irrigation|drought|monsoon|groundwater)\b/],
  ["soil",      /\b(soil|ph|texture|clay|sandy|loam|ground condition)\b/],
  ["view",      /\b(satellite|green view|map view|show (me )?(the )?(green|satellite|street|priority))\b/],
  ["goto",      /\b(go to|goto|show me|take me to|fly to|navigate to|find|search|jump to|zoom to|look at|open)\b/],
  ["report",    /\b(report|summar\w+|brief|tell me about|analyse|analyze|overview|status)\b|\bwhat('?s| is|s)?\s+(is\s+)?(here|around|nearby|in this area)\b/]
];

/* Mirrors _normalise(): lowercase, fold Indic digits, keep letters, marks
   and numbers, blank out punctuation. \p{M} is the load-bearing class —
   drop it and every Indic script becomes unreadable. */
const DIGIT_BASES = [0x0966,0x09E6,0x0A66,0x0AE6,0x0B66,0x0BE6,0x0C66,0x0CE6,0x0D66,0x06F0,0x0660];
function normalise(msg) {
  let s = String(msg || "").toLowerCase().trim().replace(/’/g, "'");
  s = s.replace(/[٠-٩۰-۹०-९০-৯੦-੯૦-૯୦-୯௦-௯౦-౯೦-೯൦-൯]/g,
    ch => {
      const c = ch.codePointAt(0);
      for (const b of DIGIT_BASES) if (c >= b && c <= b + 9) return String(c - b);
      return ch;
    });
  s = s.replace(/[^\p{L}\p{M}\p{N}\s'.,\-\/&]/gu, " ");
  return s.replace(/\s+/g, " ").trim();
}

GVE.classify = function (msg) {
  const n = normalise(msg);
  if (!n) return "help";
  for (const [name, rx] of INTENTS) if (rx.test(n)) return name;
  return /\b(here|this area|nearby|around)\b/.test(n) ? "report" : "unknown";
};

/* ---------- slot extraction (ported) -------------------------------- */

const LAND_NOUN = /^(?:the\s+|some\s+|any\s+)?(?:empty|bare|vacant|unused|open|free|barren|waste|plantable|available)?\s*(?:land|ground|space|spaces|plot|plots|area|areas|spot|spots|patch|patches|site|sites|room)\b/i;
const LOCATIVE  = /\b(?:n(?:ea|ae|e|a)r(?:by)?|around|close to|next to|beside|in|at|by|within|inside|surrounding)\s+/i;
const PLACE_STOP = /\b(?:and|then|please|for me|instead|on the map|now|find|show|tell|give|check|see|look|search|what|which|where|when|how|why|who|is|are|does|do|can|should|plant|design|build|make|create|plan|draw|cost|price|budget|review|compare|project|forecast)\b/i;

GVE.extractPlace = function (msg) {
  const s = String(msg || "").trim().replace(/[?.!]+$/, "");
  const m = s.match(/\b(?:go to|goto|show me|take me to|fly to|navigate to|jump to|zoom to|look at|search(?: for)?|find|near|nearby|around|close to|in|at|open)\s+(.+)$/i);
  if (!m) return null;
  let rest = m[1].trim();
  // The subject of the question is not the place. "find empty land near X"
  // names a THING and a PLACE; only the second is geocodable.
  const lead = rest.match(LAND_NOUN);
  if (lead) {
    const after = rest.slice(lead[0].length);
    const loc = after.match(LOCATIVE);
    if (!loc) return null;
    rest = after.slice(loc.index + loc[0].length).trim();
  }
  let place = rest.split(PLACE_STOP)[0].replace(/^[\s,.\-]+|[\s,.\-]+$/g, "");
  if (!place || place.length < 2) return null;
  if (/^(the |a |an |some |me )*(green|satellite|map|priority|street|empty|bare|vacant|open|land|ground|space|park|trees?|air|soil|water|cost|place)( view| land| ground| space)?$/i.test(place)) return null;
  return place.slice(0, 120);
};

const AREA_UNITS = [
  [/(\d+(?:\.\d+)?)\s*(?:hectares?|\bha\b)(?![a-z])/i, 10000],
  [/(\d+(?:\.\d+)?)\s*(?:acres?)(?![a-z])/i, 4046.86],
  [/(\d+(?:\.\d+)?)\s*(?:sq\.? ?km|km2|km²)(?![a-z])/i, 1e6],
  [/(\d+(?:\.\d+)?)\s*(?:square met(?:re|er)s?|sq\.? ?m|m2|m²)(?![a-z])/i, 1]
];
GVE.extractArea = function (msg) {
  const s = normalise(msg);
  for (const [rx, mult] of AREA_UNITS) { const m = s.match(rx); if (m) return parseFloat(m[1]) * mult; }
  return null;
};

const GOALS = {
  park: ["park", "public park", "green space"],
  greenbelt: ["green belt", "greenbelt"],
  riverfront: ["riverfront", "river", "waterfront", "lakefront", "riverbank"],
  community: ["community garden", "community", "allotment", "kitchen garden"],
  campus: ["school", "college", "campus", "university", "children", "kids"],
  avenue: ["avenue", "roadside", "road side", "street tree", "median"],
  residential: ["residential", "housing", "society", "apartment", "colony"],
  industrial: ["industrial", "factory", "buffer", "industry"],
  wetland: ["wetland", "marsh", "pond edge", "lake edge"]
};
const GOAL_NOUN = {
  park: "park", greenbelt: "green belt", riverfront: "riverfront planting",
  community: "community garden", campus: "campus green", avenue: "roadside avenue",
  residential: "residential green", industrial: "industrial buffer",
  wetland: "wetland edge planting"
};
GVE.extractGoal = function (msg) {
  const s = normalise(msg);
  let best = null, bestLen = 0;
  for (const g in GOALS) for (const w of GOALS[g])
    if (s.indexOf(w) >= 0 && w.length > bestLen) { best = g; bestLen = w.length; }
  return best;
};
GVE.extractCount = function (msg) {
  const s = normalise(msg);
  let m = s.match(/(\d+)\s*(?:[a-z()\-']+\s+){0,3}(?:trees?|saplings?|shrubs?|plants?)\b/);
  if (m) return Math.max(1, Math.min(2000, parseInt(m[1], 10)));
  m = s.match(/\b(?:plant|add|place|put)\s+(\d+)\b/);
  return m ? Math.max(1, Math.min(2000, parseInt(m[1], 10))) : null;
};
GVE.extractTopN = function (msg) {
  const m = normalise(msg).match(/\btop\s+(\d+)/);
  return m ? Math.max(1, Math.min(100, parseInt(m[1], 10))) : null;
};
GVE.extractYears = function (msg) {
  const m = normalise(msg).match(/(\d+)\s*years?\b/);
  return m ? Math.max(1, Math.min(50, parseInt(m[1], 10))) : null;
};

/* ---------- context view -------------------------------------------- */

function Ctx(raw) {
  raw = raw || {};
  const aoi = raw.aoi || {}, r = raw.readings || {}, d = raw.design || {};
  const f = v => (v == null || !isFinite(v) ? null : Number(v));
  return {
    lat: f(aoi.lat), lon: f(aoi.lon), km2: f(aoi.km2) || 100,
    place: String(raw.place || "").trim(),
    aqi: f(r.aqi), aqiMin: f(r.aqi_min), aqiMax: f(r.aqi_max),
    pm25: f(r.pm25), temp: f(r.temp), canopy: f(r.canopy_pct),
    bare: f(r.bare_frac), rain: f(r.rain_mm_yr), hotDays: f(r.hot_days_yr),
    nPoints: f(r.n_points), census: raw.census || {},
    goal: String(d.goal || "park"), plotM2: f(d.plot_m2),
    nTrees: f(d.n_trees) || 0, nItems: f(d.n_items) || 0,
    totalCost: f(d.total_cost), reviewScore: f(d.review_score),
    get hasPoint() { return this.lat != null && this.lon != null; },
    get where() {
      return this.place || (this.lat != null
        ? this.lat.toFixed(4) + ", " + this.lon.toFixed(4) : "this area");
    }
  };
}

const NEED_POINT = {
  reply: "Click anywhere on the map first — I read a 100 km² circle around " +
         "that point, and every number I give you comes from inside it.",
  actions: []
};

/* ---------- handlers ------------------------------------------------ */

const H = {};

H.help = () => ({ reply:
  "I can drive this map and build on it. Try:\n\n" +
  "- **Show me Bopal** — flies there and reads the 100 km²\n" +
  "- **What should I plant here** — species matched to this air, rain and soil\n" +
  "- **Design a 1 hectare park for a school** — draws it, plants it, furnishes it\n" +
  "- **Where are the top 5 cells to plant in** — the engine's ranking\n" +
  "- **Find empty land near Rajpath Club** — where there is actually room\n" +
  "- **What does this cost** / **review my design** / **project 25 years**",
  actions: [] });

H.greet = (m, c) => ({
  reply: c.place ? "Ready when you are — reading **" + c.place + "** now. Ask me what to plant, or tell me to design something."
                 : "Click a point on the map and I will read the 100 km² around it. Or tell me a place to go to.",
  actions: [] });

H.unknown = () => ({ reply:
  "I did not follow that. I can move the map, read a place, pick species, " +
  "design and cost a planting, or show the engine's ranking — ask for one of " +
  "those, or type **help**.", actions: [] });

H.goto = function (msg, c) {
  const ll = (typeof window.parseLatLon === "function") ? window.parseLatLon(msg) : null;
  if (ll) return {
    reply: "Going to **" + ll[0].toFixed(4) + ", " + ll[1].toFixed(4) + "** and reading the " + Math.round(c.km2) + " km² around it.",
    actions: [{ tool: "map.goto", args: { lat: ll[0], lon: ll[1], zoom: 15 } },
              { tool: "dock.open", args: { tab: "area" } }] };
  const place = GVE.extractPlace(msg);
  if (!place) return { reply: "Which place? Give me a name, or paste a coordinate pair.", actions: [] };
  return {
    reply: "Searching for **" + place + "**, then reading the " + Math.round(c.km2) + " km² around it.",
    actions: [{ tool: "map.search", args: { query: place } },
              { tool: "dock.open", args: { tab: "area" } }] };
};

H.view = function (msg) {
  const n = normalise(msg);
  const view = n.indexOf("priority") >= 0 ? "priority" :
               n.indexOf("green") >= 0 ? "green" :
               (n.indexOf("map view") >= 0 || n.indexOf("street") >= 0) ? "map" : "satellite";
  const words = {
    priority: "The engine's planting priority — warm where it is most urgent.",
    green: "Canopy now, plus the engine's forecast: amber is green today and predicted to lose it.",
    map: "Street map.", satellite: "Satellite imagery."
  };
  return { reply: words[view], actions: [{ tool: "map.view", args: { view } }] };
};

H.report = function (msg, c) {
  if (!c.hasPoint) return NEED_POINT;
  const bits = ["Reading **" + c.where + "** across " + Math.round(c.km2) + " km²."];
  if (c.aqi != null) {
    let l = "Air quality is **" + Math.round(c.aqi) + "** — " + aqiWord(c.aqi) +
            (c.pm25 != null ? ", PM2.5 at **" + Math.round(c.pm25) + " µg/m³**" : "") + ".";
    if (c.aqiMin != null && c.aqiMax != null && c.aqiMax - c.aqiMin > 12)
      l += " It ranges " + Math.round(c.aqiMin) + " to " + Math.round(c.aqiMax) +
           " across the perimeter — a single centre reading would have missed that.";
    bits.push(l);
  }
  if (c.canopy != null) {
    const v = c.canopy >= 30 ? "dense" : c.canopy >= 18 ? "moderate" : c.canopy >= 8 ? "thin" : "very thin";
    bits.push("Tree canopy is **" + Math.round(c.canopy) + "%** — " + v + " for an urban area.");
  }
  if (c.bare != null) bits.push("About **" + Math.round(c.bare * 100) + "%** reads as bare or near-bare ground.");
  if (c.rain != null) bits.push("Rainfall is **" + nf(Math.round(c.rain)) + " mm/yr**" +
    (c.hotDays != null ? ", with **" + Math.round(c.hotDays) + "** days a year over 40 °C" : "") + ".");
  const named = ["buildings", "roads", "parks", "trees", "schools", "hospitals"]
    .filter(k => c.census[k]).slice(0, 5)
    .map(k => "**" + nf(c.census[k]) + "** " + k);
  if (named.length) bits.push("Inside the perimeter: " + named.join(", ") + ".");
  const cell = GVE.cellReport(c.lat, c.lon);
  if (cell) bits.push("The engine ranks this cell **#" + cell.rank + "** of " + GVE.nZones() +
    " on 42 months of history — NDVI " + cell.ndvi_latest.toFixed(3) +
    " trending " + pm(cell.ndvi_trend_per_year, 4) + "/yr.");
  return { reply: bits.join("\n\n"), actions: [{ tool: "dock.open", args: { tab: "area" } }] };
};

H.air = function (msg, c) {
  if (!c.hasPoint) return NEED_POINT;
  if (c.aqi == null) return { reply: "No air-quality reading has loaded for this area yet.", actions: [] };
  const l = ["Air quality is **" + Math.round(c.aqi) + "** (" + aqiWord(c.aqi) + "), averaged over " +
             (c.nPoints || 9) + " points across " + Math.round(c.km2) + " km²."];
  if (c.pm25 != null) l.push("PM2.5 is **" + Math.round(c.pm25) + " µg/m³** — about **" +
    (c.pm25 / 15).toFixed(1) + "×** the WHO annual guideline of 15.");
  if (c.aqiMin != null && c.aqiMax != null) l.push("It runs " + Math.round(c.aqiMin) + " to " + Math.round(c.aqiMax) + " within the perimeter.");
  if (c.aqi >= 150) l.push("At this level, species choice matters: pick high pollution tolerance, and prefer dense evergreen crowns near the road edge.");
  const cell = GVE.cellReport(c.lat, c.lon);
  if (cell && cell.aqi_pred_delta != null) {
    const d = cell.aqi_pred_delta;
    l.push("The engine forecasts **" + pm(d) + "** AQI over its horizon for this cell — " +
      (d > 2 ? "worsening." : d < -2 ? "improving." : "roughly flat."));
  }
  return { reply: l.join("\n\n"), actions: [{ tool: "dock.open", args: { tab: "area" } }] };
};

H.canopy = function (msg, c) {
  if (!c.hasPoint) return NEED_POINT;
  const l = [];
  l.push(c.canopy != null
    ? "Tree canopy covers about **" + Math.round(c.canopy) + "%** of this " + Math.round(c.km2) + " km²."
    : "Canopy could not be read from imagery here.");
  if (c.bare != null) l.push("**" + Math.round(c.bare * 100) + "%** reads as bare or near-bare — that is the plantable share.");
  const cell = GVE.cellReport(c.lat, c.lon);
  if (cell) {
    l.push("The engine's own NDVI for this cell is **" + cell.ndvi_latest.toFixed(3) +
      "**, trending **" + pm(cell.ndvi_trend_per_year, 4) + "/yr**.");
    if (cell.ndvi_trend_per_year < -0.005)
      l.push("That is a real decline. This is the case the Green view's amber cells are for — green today, forecast to lose it.");
  }
  l.push("Switching to the Green view so you can see it.");
  return { reply: l.join("\n\n"), actions: [{ tool: "map.view", args: { view: "green" } }] };
};

H.water = function (msg, c) {
  if (!c.hasPoint) return NEED_POINT;
  const l = [];
  if (c.rain != null) {
    const band = c.rain < 400 ? "arid" : c.rain < 750 ? "semi-arid" : c.rain < 1200 ? "sub-humid" : "humid";
    l.push("Rainfall here is **" + nf(Math.round(c.rain)) + " mm/yr** — " + band + ".");
    if (c.rain < 750) l.push("Below about 750 mm, irrigation is not a rounding error in the budget: it is the largest line in the three-year establishment phase. Drought-tolerant species and drip both pay for themselves.");
  } else l.push("No rainfall normal has loaded for this area yet.");
  if (c.hotDays != null) l.push("**" + Math.round(c.hotDays) + "** days a year go over 40 °C, which is what actually kills a sapling in its first summer.");
  return { reply: l.join("\n\n"), actions: [] };
};

H.soil = function (msg, c) {
  const p = GVE.soilReport(c.lat, c.lon);
  if (!p) return { reply:
    "No soil profile covers this point. SoilGrids masks built-up land, and this build only carries the configured city bbox — so species matching falls back to pollution and rainfall rather than inventing a pH.",
    actions: [] };
  const b = ["Soil for cell `" + p.zone + "`:"];
  if (p.ph != null) b.push("- pH **" + p.ph.toFixed(1) + "** (" + (p.ph_class || "?") + ")");
  if (p.texture) b.push("- Texture **" + p.texture + "**" +
    (p.sand != null ? " — " + Math.round(p.sand) + "% sand, " + Math.round(p.silt) + "% silt, " + Math.round(p.clay) + "% clay" : ""));
  if (p.organic_carbon != null) b.push("- Organic carbon **" + p.organic_carbon.toFixed(1) + " g/kg**");
  if (p.nitrogen != null) b.push("- Nitrogen **" + p.nitrogen.toFixed(1) + " g/kg**");
  b.push("\nISRIC SoilGrids v2.0, 250 m, modelled — not a site test. Confirm with an auger before you order stock.");
  return { reply: b.join("\n"), actions: [] };
};

H.traffic = () => ({ reply:
  "Traffic here is **modelled** from OpenStreetMap road topology and a time-of-day curve — it is not measured flow unless you set a TomTom key. Opening the traffic panel.",
  actions: [{ tool: "dock.open", args: { tab: "traffic" } }] });

H.priority = function (msg) {
  const rows = GVE.topCells(GVE.extractTopN(msg) || 5);
  if (!rows.length) return { reply: "The engine's ranking has not loaded.", actions: [] };
  const meta = (GVE.data && GVE.data.meta) || {};
  const scored = GVE.nRanked(), total = GVE.nZones();
  const l = ["The engine ranked **" + total + " H3 cells** across " +
             (meta.city || "the city") + " on " + (meta.months_history || 42) +
             " months of MODIS NDVI and Open-Meteo AQI" +
             (scored < total
               ? " — " + scored + " of them have enough NDVI coverage to score"
               : "") +
             ". Top " + rows.length + ":", ""];
  for (const r of rows) {
    let s = "**#" + r.rank + "** — score **" + r.score.toFixed(3) + "** — AQI " +
            Math.round(r.aqi_latest) + " forecast " + pm(r.aqi_pred_delta) +
            ", NDVI " + r.ndvi_latest.toFixed(3) + " trending " + pm(r.ndvi_trend_per_year, 4) + "/yr";
    if (r.species && r.species.length) s += "\n  Plant: " + r.species.join(", ");
    l.push(s);
  }
  l.push("", "Switching to the Priority view and focusing rank 1. Click any hexagon for its full history.");
  return { reply: l.join("\n"),
    actions: [{ tool: "map.view", args: { view: "priority" } },
              { tool: "priority.focus", args: { rank: rows[0].rank } }],
    cards: rows };
};

H.empty_land = function (msg, c) {
  // _do_empty_land in assistant.py guards on has_point and this did not, so
  // the static build answered with the bare-ground share of a circle nobody
  // had chosen yet. Python is canonical.
  if (!c.hasPoint) return NEED_POINT;
  const l = [];
  if (c.bare != null) l.push("About **" + Math.round(c.bare * 100) + "%** of this " +
    Math.round(c.km2) + " km² scans as bare, plantable ground — roughly **" +
    nf(Math.round(c.bare * c.km2 * 100)) + " hectares**. That is modelled from current " +
    "satellite imagery, not a land survey, and it counts anything without vegetation: " +
    "rooftops, car parks and construction sites are in that figure alongside genuinely open soil.");
  const rows = GVE.bareCells(c.lat, c.lon, 5);
  if (rows.length) {
    l.push("", "The cells with the most room to plant, from the engine's panel:");
    for (const r of rows) l.push("**#" + r.rank + "** — **" +
      Math.round(r.plantable_space * 100) + "%** plantable, NDVI " + r.ndvi_latest.toFixed(2) +
      (r._km != null ? ", " + r._km.toFixed(1) + " km away" : ""));
  }
  l.push("", "The Green view's red cells are the same signal on the map.");
  return { reply: l.join("\n"), actions: [{ tool: "map.view", args: { view: "green" } }] };
};

H.species = function (msg, c) {
  const goal = GVE.extractGoal(msg) || c.goal;
  const ctx = window.GV && GV.ctx;
  const ranked = (window.GV && GV.rankedSpecies) ? GV.rankedSpecies(ctx || {}, goal) : [];
  if (!ranked.length) return { reply: "The species table has not loaded.", actions: [] };
  const top = ranked.slice(0, 6);
  const head = "Matched against " +
    (c.aqi != null ? "AQI " + Math.round(c.aqi) : "no AQI reading") + ", " +
    (c.rain != null ? nf(Math.round(c.rain)) + " mm/yr rainfall" : "no rainfall figure") + ", " +
    (c.canopy != null ? Math.round(c.canopy) + "% existing canopy" : "unknown canopy") +
    ", for a " + (GOAL_NOUN[goal] || goal) + ".";
  const l = [head, ""];
  for (const s of top) {
    l.push("**" + s.name + "** (*" + s.bot + "*) — " +
      Math.round(s.fit * 100) + "% fit" +
      (s.why && s.why.length ? ". " + s.why.join(", ") : "") +
      (s.warn ? ". ⚠ " + s.warn : "") + ".");
  }
  const prof = GVE.soilReport(c.lat, c.lon);
  l.push("", prof && prof.ph != null
    ? "Filtered against this cell's soil: pH " + prof.ph.toFixed(1) + ", " + (prof.texture || "unknown texture") + "."
    : "No soil profile covers this point, so pH and texture did not filter the list.");
  l.push("Species traits are indicative defaults, not verified silviculture — confirm against your state Forest Department nursery list before ordering.");
  return { reply: l.join("\n"),
    actions: [{ tool: "studio.suggest", args: { species: top.map(s => s.name), goal } }],
    cards: top };
};

/* Element schedule per goal — mirrors the Python layout_plan closely
   enough to give the same shape of scheme. */
function elementsFor(goal, area, rain) {
  const e = [];
  const path = Math.round(area * 0.06);
  e.push({ id: "path_gravel", qty: path, unit: "m2" });
  e.push({ id: "meadow", qty: Math.round(area * 0.15), unit: "m2" });
  e.push({ id: "shrub", qty: Math.round(area * 0.08), unit: "m2" });
  if (area >= 3000) e.push({ id: "bench", qty: Math.max(2, Math.round(area / 1250)), unit: "each" });
  if (area >= 3000) e.push({ id: "light", qty: Math.max(2, Math.round(area / 2000)), unit: "each" });
  if (area >= 2000) e.push({ id: "tap", qty: 1, unit: "each" });
  if (area >= 4000) e.push({ id: "compost", qty: 1, unit: "each" });
  if (area >= 300) e.push({ id: "rwh", qty: Math.max(1, Math.round(area / 8000)), unit: "each" });
  if (rain != null && rain < 750) e.push({ id: "drip", qty: Math.round(area * 0.25), unit: "m2" });
  if (goal === "campus" || goal === "community") e.push({ id: "play", qty: 1, unit: "each" });
  return e;
}

H.design = function (msg, c) {
  if (!c.hasPoint) return NEED_POINT;
  let area = GVE.extractArea(msg) || 10000;
  area = Math.max(200, Math.min(250000, area));
  const goal = GVE.extractGoal(msg) || c.goal;
  const ctx = (window.GV && GV.ctx) || {};
  const ranked = (window.GV && GV.rankedSpecies) ? GV.rankedSpecies(ctx, goal) : [];
  const pool = ranked.filter(s => s.type === "tree" && !s.warn).slice(0, 6);
  if (!pool.length) return { reply: "The species table has not loaded, so I will not guess a mix.", actions: [] };

  const spacing = area < 2000 ? 6 : area < 20000 ? 8 : 10;
  const fits = Math.max(1, Math.floor((area * 0.55) / (spacing * spacing)));
  /* Keep no species above the cap — Santamour's rule is the whole reason the
     review engine exists; the assistant should not hand you a design that
     fails it on arrival.

     This used to end `if (left > 0) mix[0].count += left`, which dumped every
     unplaced tree on the FIRST species and sailed straight past the cap. It
     looked fine at pool=6 and produced 44% / 72% / 100% single-species stands
     at pool=3 / 2 / 1 — and `pool` shrinks exactly when the site is harsh,
     because bad air and poor soil set `warn` on species. So the worst sites
     got the monoculture, and reviewDesign() then flagged the design the
     assistant had just written. Mirrors server.py layout_plan() now:
     redistribute round-robin onto whoever is still under the cap. */
  const cap = Math.max(1, Math.floor(fits * 0.28));
  const per = Math.max(1, Math.round(fits / pool.length));
  const counts = pool.map(() => 0);
  let left = fits;
  for (let i = 0; i < pool.length && left > 0; i++) {
    // `left` is in the min so the counter never goes negative — the old
    // `left -= take` after a `Math.min(take, left)` count let the two drift
    // apart, and the emitted mix silently undershot the number reported.
    const take = Math.min(cap, per, left);
    counts[i] = take;
    left -= take;
  }
  let guard = 0;
  while (left > 0 && guard++ < 10000) {
    const room = [];
    for (let i = 0; i < counts.length; i++) if (counts[i] > 0 && counts[i] < cap) room.push(i);
    if (!room.length) break;          // everyone is at the cap
    for (const i of room) { if (left <= 0) break; counts[i]++; left--; }
  }
  /* If trees are still unplaced here, every species is at its cap and the
     only ways forward are to break the rule or to plant fewer. Plant fewer,
     and say so below rather than quietly reporting the spacing figure. */
  const mix = [];
  for (let i = 0; i < pool.length; i++)
    if (counts[i] > 0) mix.push({ species: pool[i].name, count: counts[i] });
  const nTrees = mix.reduce((t, m) => t + m.count, 0);

  const els = elementsFor(goal, area, c.rain);
  const elNames = { path_gravel: "gravel trail", meadow: "native grass meadow",
    shrub: "shrub massing", bench: "benches", light: "solar path lights",
    tap: "drinking water point", compost: "composting bay",
    rwh: "rainwater recharge pit", drip: "drip irrigation", play: "play equipment" };

  const l = [
    "Laying out a **" + (area / 10000).toFixed(2) + " ha** (" + nf(Math.round(area)) +
      " m²) " + (GOAL_NOUN[goal] || goal) + " at " + c.where + ".",
    "",
    "**" + nTrees + " trees** at " + spacing + " m centres — " +
      mix.map(m => m.count + "× " + m.species).join(", ") + ".",
    "Plus " + els.map(e => (e.unit === "m2" ? nf(e.qty) + " m² of " : e.qty + " ") +
      (elNames[e.id] || e.id)).join(", ") + ".",
    ""
  ];
  if (nTrees < fits)
    l.push("The spacing would take " + fits + " trees, but only " + pool.length +
      (pool.length === 1 ? " species survives" : " species survive") +
      " this site's air and soil filters, and holding each of them under the " +
      cap + "-tree diversity cap leaves room for " + nTrees +
      ". A thinner stand is recoverable; a monoculture that a single pest clears is not.");
  if (c.aqi != null && c.aqi >= 120)
    l.push("Species are weighted for pollution tolerance because AQI here is " + Math.round(c.aqi) + ".");
  if (c.rain != null && c.rain < 750)
    l.push("Drip irrigation is in the schedule because " + nf(Math.round(c.rain)) +
      " mm/yr will not carry this planting through its first three summers on its own.");
  if (area >= 300)
    l.push("A recharge pit is included because most Indian municipal codes require one above a 300 m² plot.");
  l.push("", "Drawing it now. Everything is editable — click any tree to remove it, or redraw the plot. Ask me for the cost when you want the bill of quantities.");

  return {
    reply: l.join("\n"),
    actions: [
      { tool: "studio.goal", args: { goal } },
      { tool: "studio.plot", args: { area_m2: area } },
      { tool: "studio.autoplant", args: { mix, spacing_m: spacing } },
      { tool: "studio.elements", args: { elements: els } },
      { tool: "dock.open", args: { tab: "studio" } }
    ]
  };
};

H.plant = function (msg, c) {
  const n = GVE.extractCount(msg) || 30;
  const goal = GVE.extractGoal(msg) || c.goal;
  const ctx = (window.GV && GV.ctx) || {};
  const ranked = (window.GV && GV.rankedSpecies) ? GV.rankedSpecies(ctx, goal) : [];
  // Honour a named species if the message contains one.
  const named = ranked.find(s => normalise(msg).indexOf(s.name.toLowerCase().replace(/\s*\(.*?\)/, "")) >= 0);
  const pool = named ? [named] : ranked.filter(s => s.type === "tree" && !s.warn).slice(0, 4);
  if (!pool.length) return { reply: "The species table has not loaded.", actions: [] };
  const per = Math.max(1, Math.floor(n / pool.length));
  const mix = pool.map((s, i) => ({ species: s.name, count: i === pool.length - 1 ? n - per * (pool.length - 1) : per }));
  return {
    reply: "Planting **" + n + "** — " + mix.map(m => m.count + "× " + m.species).join(", ") +
      ".\n\nIf there is no plot yet I will square one off inside the perimeter first. Click any tree to remove it.",
    actions: [{ tool: "studio.autoplant", args: { mix, spacing_m: 8 } },
              { tool: "dock.open", args: { tab: "studio" } }]
  };
};

H.cost = function (msg, c) {
  if (!c.nItems) return { reply:
    "There is nothing placed yet to cost. Tell me to design something — " +
    "*design a 1 hectare park* — and I will draw it, then price it.", actions: [] };
  const l = [];
  if (c.totalCost != null) {
    l.push("This design comes to **" + inr(c.totalCost) + "** all in — direct cost plus contingency, design fee and GST.");
    if (c.nTrees) l.push("That is **" + c.nTrees + " trees** across " +
      (c.plotM2 ? nf(Math.round(c.plotM2)) + " m²" : "the plot") + ".");
  }
  l.push("Opening the full bill of quantities. Every line comes from something actually placed — nothing is a percentage of a guess except contingency and the design fee, which are the industry conventions.");
  if (c.rain != null) l.push("The three-year establishment water is scaled to this site's **" +
    nf(Math.round(c.rain)) + " mm/yr** rainfall, so the same design costs differently in a different city.");
  l.push("Rates are indicative 2026 Indian figures — planning-grade, not a quotation.");
  return { reply: l.join("\n\n"), actions: [{ tool: "dock.open", args: { tab: "cost" } }] };
};

H.review = function (msg, c) {
  if (!c.nItems) return { reply:
    "Nothing to review yet — design something first and I will score it.", actions: [] };
  const l = [];
  if (c.reviewScore != null) l.push("This design scores **" + Math.round(c.reviewScore) + " / 100**.");
  l.push("Twelve checks, weighted: Santamour's 10/20/30 rule for species and genus share, " +
    "mature-crown spacing, water balance against this site's own rainfall, shade over walking " +
    "routes, permeable ground, and safety.");
  l.push("It knows nothing about ownership, utilities or drainage surveys — a good score means " +
    "the planting logic holds up, not that this is buildable.");
  return { reply: l.join("\n\n"), actions: [{ tool: "review.show", args: {} }] };
};

H.project = function (msg, c) {
  if (!c.nItems) return { reply: "Design something first and I will project it forward.", actions: [] };
  const y = GVE.extractYears(msg) || 25;
  return {
    reply: "Projecting **" + y + " years**: logistic canopy growth, survival curves, species " +
      "lifespan and saturating cooling.\n\nThis is labelled **PROJECTED, not forecast** and " +
      "**UNVALIDATED** in the source, and it means it — no observation exists that far out, so " +
      "error compounds and cannot be checked. Treat it as a defensible shape, not a prediction.",
    actions: [{ tool: "project.show", args: { years: y } }]
  };
};

H.compare = function (msg) {
  const s = String(msg || "").trim().replace(/[?.!]+$/, "");
  let m = s.match(/\bcompare\s+(.+?)\s+(?:and|with|to|versus|vs\.?)\s+(.+?)$/i) ||
          s.match(/^(.+?)\s+(?:versus|vs\.?)\s+(.+?)$/i);
  if (!m) return { reply: "Name two places — *compare Bopal and Vastrapur*.", actions: [] };
  return { reply: "Reading **" + m[1].trim() + "** and **" + m[2].trim() +
      "** in turn and putting the two side by side.",
    actions: [{ tool: "compare.run", args: { a: m[1].trim(), b: m[2].trim() } }] };
};

/* ---------- entry point --------------------------------------------- */

GVE.handle = async function (message, context) {
  await GVE.load();
  const c = Ctx(context);
  const intent = GVE.classify(message);

  // Compound request: "near Rajpath Club find empty land" names a place AND
  // asks something. Move first, then answer — same as the Python.
  let lead = [];
  if (["goto", "compare", "greet", "help", "unknown"].indexOf(intent) < 0) {
    const place = GVE.extractPlace(message);
    const ll = (typeof window.parseLatLon === "function") ? window.parseLatLon(message) : null;
    if (place && !ll) {
      lead = [{ tool: "map.search", args: { query: place } },
              { tool: "dock.open", args: { tab: "area" } }];
      if (!c.place) c.place = place;
    }
  }

  let out;
  try {
    out = (H[intent] || H.report)(message, c);
  } catch (e) {
    out = { reply: "That question hit an error in the offline planner: " + e.message +
      "\n\nThe full engine handles more than this build does — start it with " +
      "`python -m greenplan.server --config config/city.yaml`.", actions: [] };
  }
  out.actions = out.actions || [];
  out.cards = out.cards || [];
  if (lead.length) {
    out.actions = lead.concat(out.actions);
    out.reply = "Moving to **" + GVE.extractPlace(message) + "** first, then answering that.\n\n" + out.reply;
  }
  out.intent = intent;
  out.lang = "en";
  out.dir = "ltr";
  // Named honestly, and the UI prints it: this is the browser's planner
  // working off baked engine output, not the trained Python process.
  out.source = "offline-planner (static build)";
  return out;
};

})();
