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
   no weights, just intent matching, a short conversational memory, and
   planning. So it ports.

   What the assistant can do that a bare intent-matcher cannot
   ----------------------------------------------------------
   - Carry context across turns. "and the cost?", "why those?", "make it
     bigger", "do that there", "the second one", "yes" all resolve
     against what was just said or shown (see MEM and followUp()).
   - Hold a short back-and-forth. Vague asks ("design something") get a
     question back, not a guessed answer; the reply is remembered so the
     next message finishes the request.
   - Small talk. "thanks", "who are you", "nice" get a human reply, not
     the help card.
   - Reason from partial input. A message no pattern catches is scanned
     for weak signals ("shade", "kids", "afford") and either routed with
     a stated assumption or met with one targeted question.

   What this is NOT
   ----------------
   It is not a language model and does not pretend to be. Every number
   still comes from the baked engine payloads; the assistant only chooses
   which number to show and how to phrase it. `source` on every answer
   says which brain produced it, and the UI prints that.

   Honest scope
   ------------
   Narrower than greenplan/reasoning/assistant.py — the Python keeps the
   full glossary and several long-tail intents. Where this cannot answer
   it says so and names the command that starts the full engine.
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
const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;

/* Deterministic rotation, not randomness: seeded by the turn counter so a
   reader who asks the same thing twice gets different words, but a test
   run is still reproducible. */
function pick(arr, salt) {
  return arr[(MEM.turns + (salt || 0)) % arr.length];
}

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
   precede the general ones exactly as they do in the Python.

   scripts/parity_check.py runs a corpus of real messages through this
   and the Python classifier and fails on divergence — so this list and
   its order are frozen against that file, and the conversational layer
   below never edits a classify() result, it only runs earlier. */

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
  // "a quarter hectare", "half a hectare", "half acre" — words people use.
  if (/\b(quarter|1\/4)\s+(?:of\s+)?(?:an?\s+)?(hectare|ha|acre)\b/.test(s)) return /acre/.test(s) ? 1012 : 2500;
  if (/\b(half|1\/2)\s+(?:of\s+)?(?:an?\s+)?(hectare|ha|acre)\b/.test(s))    return /acre/.test(s) ? 2023 : 5000;
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
    get hasDesign() { return this.nItems > 0 || this.plotM2 != null; },
    get where() {
      return this.place || (this.lat != null
        ? this.lat.toFixed(4) + ", " + this.lon.toFixed(4) : "this area");
    }
  };
}

/* ---------- conversational memory ----------------------------------
   Session-scoped and in-memory: a page reload starts a fresh conversation,
   which matches what a reader expects. Nothing here is persisted and
   nothing leaves the tab. Everything the follow-up resolver needs to make
   "and the cost?" or "the second one" mean something lives on this object. */

const MEM = GVE.mem = {
  turns: 0,
  greeted: false,
  lastIntent: null,     // the intent the previous turn resolved to
  lastTopic: null,      // coarse subject, for "why?": air|water|canopy|soil|species|priority|design|empty_land
  species: [],          // [{name, bot}] shown in the last species/design answer, best first
  focusSpecies: null,   // the one species singled out by an ordinal ref, for "plant it"
  cells: [],            // [rank...] shown in the last priority/empty-land answer
  lastPlace: null,      // last place name a goto resolved, for "go back"
  proposal: null,       // { run: () => ({reply,actions}) } — awaiting "yes"
  pending: null,        // { intent, text } — a half-specified request awaiting more detail
  design: { area: null, goal: null }   // running design spec across turns
};

GVE.reset = function () {
  MEM.turns = 0; MEM.greeted = false; MEM.lastIntent = null; MEM.lastTopic = null;
  MEM.species = []; MEM.focusSpecies = null; MEM.cells = []; MEM.lastPlace = null;
  MEM.proposal = null; MEM.pending = null;
  MEM.design = { area: null, goal: null };
};

const TOPIC_OF = {
  air: "air", water: "water", canopy: "canopy", soil: "soil",
  species: "species", plant: "species", priority: "priority",
  empty_land: "empty_land", design: "design", cost: "cost", report: "report"
};

/* ---------- handlers ------------------------------------------------ */

const H = {};

const NEED_POINT = () => ({
  reply: pick([
    "Click a point on the map first — everything I read comes from the 100 km² circle around it.",
    "Pick a spot on the map and I will read the 100 km² around it. Or tell me a place to go to.",
    "I need a point on the map before I can answer that — the whole read is scoped to a 100 km² circle."
  ]),
  actions: []
});

H.help = (m, c) => {
  const lines = [
    "- **Show me Bopal** — flies there and reads the 100 km²",
    "- **What should I plant here** — species matched to this air, rain and soil",
    "- **Design a 1 hectare park for a school** — draws it, plants it, furnishes it",
    "- **Where are the top 5 cells to plant** — the engine's ranking",
    "- **Find empty land near Rajpath Club** — where there is actually room",
    "- **What does this cost** / **review my design** / **project 25 years**"
  ];
  const head = c.hasDesign
    ? "You have a design open. I can cost it, review it, project it 25 years, or keep editing it — just say. Other things I do:"
    : c.hasPoint
      ? "You are on **" + c.where + "**. From here you can ask what to plant, where the city should plant first, or tell me to design something. The full list:"
      : "I drive this map and build on it. Try one of these:";
  return { reply: head + "\n\n" + lines.join("\n") +
    "\n\nI answer from measured data and the baked engine — where there is no number I say so rather than invent one.",
    actions: [] };
};

H.greet = (m, c) => {
  if (MEM.greeted) {
    return { reply: pick([
      c.hasDesign ? "Still here. Want the cost, a review, or more edits to the design?"
                  : c.hasPoint ? "Still here — what about **" + c.where + "** do you want to know?"
                               : "Still here. Pick a point on the map, or name a place to go to.",
      c.hasPoint ? "Go on — ask me about the air, the canopy, or what to plant on **" + c.where + "**."
                 : "Ready when you are. Click the map, or tell me where to go."
    ]), actions: [] };
  }
  MEM.greeted = true;
  return { reply: c.place
    ? "Hello. You are looking at **" + c.place + "**. Ask me what is here, where the city should plant first, or tell me to design something — say **help** for the full list."
    : "Hello. Click a point on the map and I will read the 100 km² around it, or name a place to go to. Say **help** for what I can do.",
    actions: [] };
};

H.thanks = () => ({
  reply: pick([
    "Anytime.", "Happy to help.", "Sure thing.", "You got it.",
    "Glad it helped."
  ]) + (MEM.lastIntent === "species" || MEM.lastIntent === "design"
    ? " Tell me to plant it, or ask what it would cost."
    : MEM.lastIntent === "priority"
      ? " Say the rank number to jump to a cell."
      : ""),
  actions: []
});

H.identity = () => ({
  reply: "I am Green Vision's built-in assistant — a small deterministic planner, not a language model. " +
    "I read the baked engine (42 months of MODIS NDVI, Open-Meteo air, ISRIC soil) and the live map, " +
    "and I operate the studio for you. Every figure I give comes from that data; I only choose which one " +
    "to show. Nothing you type leaves this tab.",
  actions: []
});

H.smalltalk = (m, c) => {
  const n = normalise(m);
  if (/\b(how are you|how's it going|hows it going|you (ok|good|alright))\b/.test(n))
    return { reply: "Running fine, thanks. " + (c.hasPoint ? "Ask me anything about **" + c.where + "**." : "Click the map and I will get to work."), actions: [] };
  if (/\b(good (job|work|bot)|well done|nice one|you'?re (smart|clever|good|helpful)|amazing|impressive|love (this|it))\b/.test(n))
    return { reply: pick(["Kind of you — it is really the engine doing the work.", "Thanks. The numbers are all from measured data, I just pick which to show."]), actions: [] };
  if (/\b(bye|goodbye|see ya|see you|later|cya)\b/.test(n))
    return { reply: "See you. Your designs stay in this browser until you clear its data.", actions: [] };
  if (/\b(sorry|my bad|oops)\b/.test(n))
    return { reply: "No trouble. What would you like to do?", actions: [] };
  return null;
};

/* Reason from a message no pattern caught: scan for weak signals, and
   either route it with the assumption stated, or ask exactly one
   question. Never the bare "I did not understand" unless nothing at all
   is recognisable. */
const SOFT = [
  [/\b(shade|shady|cooler|cool it|hot|heat|temperature|sun)\b/, "heat", "the heat and shade here"],
  [/\b(kids?|children|playground|school|students?)\b/, "design-campus", "a green space for children"],
  [/\b(afford|budget|money|lakh|crore|rupees?|spend|cheap|expensive)\b/, "cost", "the cost"],
  [/\b(survive|survival|die|dying|dead|alive|keep alive)\b/, "survival", "whether a planting would survive"],
  [/\b(carbon|co2|sequest|offset)\b/, "carbon", "carbon uptake"],
  [/\b(dust|particulate|smog|breathe|breathing|lungs?)\b/, "air", "the air here"],
  [/\b(flood|runoff|waterlog|drain|drainage|recharge)\b/, "water", "water and drainage"],
  [/\b(bird|butterfl|pollinat|wildlife|habitat|biodiversity)\b/, "species", "planting for wildlife"],
  [/\b(neem|peepal|banyan|gulmohar|amaltas|jamun|tamarind|mango)\b/, "species", "a species you named"],
  [/\b(picnic|walk|jog|sit|bench|seating|path)\b/, "design", "a usable green space"]
];

H.unknown = function (msg, c) {
  const n = normalise(msg);
  for (const [rx, route, phrase] of SOFT) {
    if (!rx.test(n)) continue;
    if (route === "heat")
      return { reply: "I think you are asking about **" + phrase + "**. " +
        (c.hasPoint
          ? "Canopy cuts street-level temperature 2–4 °C under a closed cover on a clear afternoon. " +
            (c.canopy != null ? "Cover here is **" + Math.round(c.canopy) + "%** right now" +
              (c.hotDays != null ? ", against **" + Math.round(c.hotDays) + "** days a year over 40 °C" : "") + "." : "Click the map and I will read the current cover.") +
            " Ask me to **design a park** and the review projects the cooling."
          : "Click a point on the map and I will read the current canopy and hot-day count."),
        actions: c.hasPoint ? [{ tool: "map.view", args: { view: "green" } }] : [] };
    if (route === "design-campus")
      return { reply: "Sounds like you want **" + phrase + "**. Tell me the size — a quarter hectare, half, one — and I will lay out a campus green: shade over the routes children walk, nothing toxic near play, and the cost.",
        actions: [], _pending: { intent: "design", text: "design a campus green for a school" } };
    if (route === "cost")
      return H.cost(msg, c);
    if (route === "survival")
      return { reply: "You are asking about **" + phrase + "**. Realistic urban survival in Indian municipal plantings is **60–85% at three years** when watering is budgeted, well under half when it is not. The three-year establishment period is where the money and nearly all the mortality sit. Design something and the review carries survival forward.", actions: [] };
    if (route === "carbon")
      return { reply: "You are asking about **" + phrase + "**. A mixed urban planting absorbs roughly **10–30 kg CO₂ per tree per year at maturity** — wide on purpose, because it depends on survival and water. Design a planting and the projection totals it over 25 years.", actions: [] };
    // air / water / species — route straight through with a note
    const h = H[route === "species" ? "species" : route];
    if (h) {
      const out = h(msg, c);
      out.reply = "I read that as a question about **" + phrase + "**.\n\n" + out.reply;
      return out;
    }
  }
  // Nothing recognisable.
  return { reply: pick([
    "I did not catch that. I can read this place, rank where the city should plant, pick species, or design and cost a park — which of those?",
    "Not sure what you are after. Try **what's here**, **what should I plant**, **where should the city plant first**, or **design a 1 hectare park**.",
    "That one got past me. Say **help** for the full list, or ask about the air, canopy, soil, or a design."
  ]), actions: [] };
};

H.goto = function (msg, c) {
  const ll = (typeof window.parseLatLon === "function") ? window.parseLatLon(msg) : null;
  if (ll) {
    MEM.lastPlace = ll[0].toFixed(4) + ", " + ll[1].toFixed(4);
    return {
      reply: "Going to **" + MEM.lastPlace + "** and reading the " + Math.round(c.km2) + " km² around it.",
      actions: [{ tool: "map.goto", args: { lat: ll[0], lon: ll[1], zoom: 15 } },
                { tool: "dock.open", args: { tab: "area" } }] };
  }
  const place = GVE.extractPlace(msg);
  if (!place) return { reply: "Which place? Give me a name, or paste a coordinate pair.", actions: [], _pending: { intent: "goto", text: "go to" } };
  MEM.lastPlace = place;
  return {
    reply: pick(["Searching for **" + place + "**, then reading the " + Math.round(c.km2) + " km² around it.",
                 "On my way to **" + place + "** — I will read the " + Math.round(c.km2) + " km² there."]),
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
  if (!c.hasPoint) return NEED_POINT();
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
  // A short verdict, so the reader gets a "so what", not just a table.
  if (c.aqi != null && c.canopy != null) {
    bits.push(c.aqi > 100 && c.canopy < 18
      ? "**This is a planting case** — air above the moderate band with thin cover is exactly what the engine is built to find. Ask me what to plant."
      : c.aqi > 100
        ? "The **air is the problem more than the cover** — the gain here is buffering the source with dense-crowned, high-tolerance species along the road edge."
        : c.canopy < 12
          ? "Thin cover but tolerable air — planting here buys **shade and surface cooling** more than air quality, still worth doing in a 40 °C city."
          : "Cover and air are both in a decent band — ask me for the **priority list** to find cells that need it more.");
  }
  return { reply: bits.join("\n\n"), actions: [{ tool: "dock.open", args: { tab: "area" } }] };
};

H.air = function (msg, c) {
  if (!c.hasPoint) return NEED_POINT();
  if (c.aqi == null) return { reply: "No air-quality reading has loaded for this area yet. Click the map again to retry.", actions: [] };
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
  if (!c.hasPoint) return NEED_POINT();
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
  if (!c.hasPoint) return NEED_POINT();
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
  l.push("", "Switching to the Priority view and framing rank 1. Say a rank number to jump to another, or **why** for how the score is built.");
  MEM.cells = rows.map(r => r.rank);
  return { reply: l.join("\n"),
    actions: [{ tool: "map.view", args: { view: "priority" } },
              { tool: "priority.focus", args: { rank: rows[0].rank } }] };
};

H.empty_land = function (msg, c) {
  if (!c.hasPoint) return NEED_POINT();
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
    MEM.cells = rows.map(r => r.rank);
  }
  l.push("", "The Green view's red cells are the same signal on the map.");
  return { reply: l.join("\n"), actions: [{ tool: "map.view", args: { view: "green" } }] };
};

/* Read the ranked palette and remember what was shown, so "why those",
   "the second one" and "plant it" resolve on the next turn. */
function rankedForHere(c, goal) {
  const ctx = window.GV && GV.ctx;
  const ranked = (window.GV && GV.rankedSpecies) ? GV.rankedSpecies(ctx || {}, goal) : [];
  return ranked;
}

H.species = function (msg, c) {
  const goal = GVE.extractGoal(msg) || MEM.design.goal || c.goal;
  const ranked = rankedForHere(c, goal);
  if (!ranked.length) return { reply: "The species table has not loaded.", actions: [] };
  const top = ranked.slice(0, 6);
  MEM.species = top.map(s => ({ name: s.name, bot: s.bot }));
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
  // State the weighting, so the ranking reads as a judgement rather than a list.
  if (c.aqi != null && c.aqi >= 130)
    l.push("Pollution tolerance is weighted heavily here — AQI " + Math.round(c.aqi) + " is why the showy species drop down the list.");
  else if (c.rain != null && c.rain < 700)
    l.push("Drought tolerance is doing most of the sorting — at " + nf(Math.round(c.rain)) + " mm/yr, a thirsty species is a permanent irrigation bill.");
  l.push("Traits are indicative defaults, not verified silviculture — confirm against your state Forest Department nursery list before ordering. Say **plant the first three**, or **design a park** to lay them out.");
  return { reply: l.join("\n"),
    actions: [{ tool: "studio.suggest", args: { species: top.map(s => s.name), goal } }] };
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

function layoutDesign(area, goal, c) {
  area = Math.max(200, Math.min(250000, area));
  const ctx = (window.GV && GV.ctx) || {};
  const ranked = (window.GV && GV.rankedSpecies) ? GV.rankedSpecies(ctx, goal) : [];
  const pool = ranked.filter(s => s.type === "tree" && !s.warn).slice(0, 6);
  if (!pool.length) return null;

  const spacing = area < 2000 ? 6 : area < 20000 ? 8 : 10;
  const fits = Math.max(1, Math.floor((area * 0.55) / (spacing * spacing)));
  const cap = Math.max(1, Math.floor(fits * 0.28));
  const per = Math.max(1, Math.round(fits / pool.length));
  const counts = pool.map(() => 0);
  let left = fits;
  for (let i = 0; i < pool.length && left > 0; i++) {
    const take = Math.min(cap, per, left);
    counts[i] = take;
    left -= take;
  }
  let guard = 0;
  while (left > 0 && guard++ < 10000) {
    const room = [];
    for (let i = 0; i < counts.length; i++) if (counts[i] > 0 && counts[i] < cap) room.push(i);
    if (!room.length) break;
    for (const i of room) { if (left <= 0) break; counts[i]++; left--; }
  }
  const mix = [];
  for (let i = 0; i < pool.length; i++)
    if (counts[i] > 0) mix.push({ species: pool[i].name, count: counts[i] });
  const nTrees = mix.reduce((t, m) => t + m.count, 0);
  return { area, goal, spacing, fits, cap, mix, nTrees, pool,
           els: elementsFor(goal, area, c.rain) };
}

H.design = function (msg, c) {
  if (!c.hasPoint) return NEED_POINT();

  let area = GVE.extractArea(msg) || MEM.design.area;
  let goal = GVE.extractGoal(msg) || MEM.design.goal;

  // Vague ask — no size, no purpose — is a question, not a guess. Remember
  // it so the next message finishes the request.
  const vague = area == null && goal == null &&
    /\b(design|build|plan|make|create|something|a green|greenspace)\b/.test(normalise(msg)) &&
    !/\b(park|garden|belt|avenue|buffer|campus|school|riverfront|wetland|residential|hectare|acre|\bha\b|\bm2\b|square)\b/.test(normalise(msg));
  if (vague) {
    return { reply: "Happy to lay one out. Two things: **how big** — a quarter hectare, half, one, more — and **what is it for**: a public park, a school ground, a street buffer, a riverfront? Tell me both and I will draw it.",
      actions: [], _pending: { intent: "design", text: "design a" } };
  }

  area = area || 10000;
  goal = goal || "park";
  MEM.design = { area, goal };

  const d = layoutDesign(area, goal, c);
  if (!d) return { reply: "The species table has not loaded, so I will not guess a mix.", actions: [] };
  MEM.species = d.pool.slice(0, 6).map(s => ({ name: s.name, bot: s.bot }));

  const elNames = { path_gravel: "gravel trail", meadow: "native grass meadow",
    shrub: "shrub massing", bench: "benches", light: "solar path lights",
    tap: "drinking water point", compost: "composting bay",
    rwh: "rainwater recharge pit", drip: "drip irrigation", play: "play equipment" };

  const l = [
    "Laying out a **" + (d.area / 10000).toFixed(2) + " ha** (" + nf(Math.round(d.area)) +
      " m²) " + (GOAL_NOUN[goal] || goal) + " at " + c.where + ".",
    "",
    "**" + d.nTrees + " trees** at " + d.spacing + " m centres — " +
      d.mix.map(m => m.count + "× " + m.species).join(", ") + ".",
    "Plus " + d.els.map(e => (e.unit === "m2" ? nf(e.qty) + " m² of " : e.qty + " ") +
      (elNames[e.id] || e.id)).join(", ") + ".",
    ""
  ];
  if (d.nTrees < d.fits)
    l.push("The spacing would take " + d.fits + " trees, but only " + d.pool.length +
      (d.pool.length === 1 ? " species survives" : " species survive") +
      " this site's air and soil filters, and holding each under the " +
      d.cap + "-tree diversity cap leaves room for " + d.nTrees +
      ". A thinner stand is recoverable; a monoculture that one pest clears is not.");
  if (c.aqi != null && c.aqi >= 120)
    l.push("Species are weighted for pollution tolerance because AQI here is " + Math.round(c.aqi) + ".");
  if (c.rain != null && c.rain < 750)
    l.push("Drip irrigation is in the schedule because " + nf(Math.round(c.rain)) +
      " mm/yr will not carry this planting through its first three summers on its own.");
  if (d.area >= 300)
    l.push("A recharge pit is included because most Indian municipal codes require one above a 300 m² plot.");
  l.push("", "Drawing it now. Everything is editable — click any tree to remove it, or redraw the plot. Say **cost** for the bill of quantities, **review** to score it, or **make it bigger** / **add a pond**.");

  return {
    reply: l.join("\n"),
    actions: [
      { tool: "studio.goal", args: { goal } },
      { tool: "studio.plot", args: { area_m2: d.area } },
      { tool: "studio.autoplant", args: { mix: d.mix, spacing_m: d.spacing } },
      { tool: "studio.elements", args: { elements: d.els } },
      { tool: "dock.open", args: { tab: "studio" } }
    ]
  };
};

H.plant = function (msg, c) {
  const n = GVE.extractCount(msg) || 30;
  const goal = GVE.extractGoal(msg) || MEM.design.goal || c.goal;
  const ctx = (window.GV && GV.ctx) || {};
  const ranked = (window.GV && GV.rankedSpecies) ? GV.rankedSpecies(ctx, goal) : [];
  const named = ranked.find(s => normalise(msg).indexOf(s.name.toLowerCase().replace(/\s*\(.*?\)/, "")) >= 0);
  const pool = named ? [named] : ranked.filter(s => s.type === "tree" && !s.warn).slice(0, 4);
  if (!pool.length) return { reply: "The species table has not loaded.", actions: [] };
  const per = Math.max(1, Math.floor(n / pool.length));
  const mix = pool.map((s, i) => ({ species: s.name, count: i === pool.length - 1 ? n - per * (pool.length - 1) : per }));
  MEM.species = pool.map(s => ({ name: s.name, bot: s.bot }));
  return {
    reply: "Planting **" + n + "** — " + mix.map(m => m.count + "× " + m.species).join(", ") +
      ".\n\nIf there is no plot yet I will square one off inside the perimeter first. Click any tree to remove it, or say **cost**.",
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
  if (!m) {
    // One place named, or none — ask for the missing half.
    const one = s.match(/\bcompare\s+(.+)$/i);
    if (one && one[1].trim().length > 1)
      return { reply: "Compare **" + one[1].trim() + "** with where? Name the second place.",
        actions: [], _pending: { intent: "compare", text: "compare " + one[1].trim() + " and" } };
    return { reply: "Name two places — *compare Bopal and Vastrapur*.", actions: [] };
  }
  return { reply: "Reading **" + m[1].trim() + "** and **" + m[2].trim() +
      "** in turn and putting the two side by side.",
    actions: [{ tool: "compare.run", args: { a: m[1].trim(), b: m[2].trim() } }] };
};

/* ---------- follow-up resolution ----------------------------------
   Runs BEFORE classify(). Returns a handler result, or null to fall
   through to normal classification. This is where a conversation is:
   every branch reads MEM to give a bare "why?" or "the second one" or
   "yes" something to attach to. */

const AFFIRM = /^\s*(y|ya|yea|yeah|yep|yes|yup|sure|ok|okay|k|do it|go|go ahead|please do|sounds good|do that|make it|let'?s do it|alright|fine|proceed)\s*[.!]*\s*$/;
const DENY   = /^\s*(n|no|nope|nah|not now|no thanks?|no,? thank you|never ?mind|nvm|forget it|cancel|stop|don'?t|leave it|not really)\s*[.!]*\s*$/;
const THANKS = /\b(thanks|thank you|thankyou|cheers|appreciate it|much appreciated|thx)\b/;
const PRAISE = /\b(good (job|work|bot)|well done|nice one|you'?re (smart|clever|good|great|helpful|amazing)|impressive|love (this|it)|brilliant|perfect|great stuff)\b/;
const IDENT  = /\b(who are you|what are you|are you (an? )?(ai|bot|human|real|chatgpt|llm|model)|your name|what model|how do you work)\b/;
const SOCIAL = /\b(how are you|how's it going|hows it going|you (ok|good|alright)|bye|goodbye|see ya|see you|later|cya|sorry|my bad|oops)\b/;

const ORDINALS = { first: 1, "1st": 1, one: 1, second: 2, "2nd": 2, two: 2,
  third: 3, "3rd": 3, three: 3, fourth: 4, "4th": 4, four: 4,
  fifth: 5, "5th": 5, five: 5, last: -1 };

function ordinalRef(n) {
  // "the first three", "the top two" is a quantity, not a position — leave
  // it for the plant-a-mix branch.
  if (/\b(first|second|top)\s+(two|three|four|five|\d+)\b/.test(n)) return null;
  let m = n.match(/\b(number|no\.?|#|rank)\s*(\d+)\b/);
  if (m) return parseInt(m[2], 10);
  m = n.match(/\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last)\b/);
  if (m) return ORDINALS[m[1]];
  m = n.match(/^\s*(\d{1,3})\s*[.!]*\s*$/);           // a bare number, "3"
  if (m) return parseInt(m[1], 10);
  return null;
}

/* Tag a reply as small talk, so finish() does not let it overwrite the
   last real topic — "thanks" then "why?" must still explain the species
   list, not the thank-you. */
const chat = out => { if (out) out._intent = "chat"; return out; };

function followUp(raw, c) {
  const n = normalise(raw);
  if (!n) return null;

  // "yes" / "do it" — run whatever was proposed.
  if (AFFIRM.test(n)) {
    if (MEM.proposal && typeof MEM.proposal.run === "function") {
      const out = MEM.proposal.run();
      MEM.proposal = null;
      return out;
    }
    if (MEM.pending) return null;   // let the pending-merge path handle it
    return chat({ reply: pick(["Nothing pending on my side — tell me what to do.",
                          "I do not have anything queued. What would you like?"]), actions: [] });
  }
  if (DENY.test(n)) {
    MEM.proposal = null; MEM.pending = null;
    return chat({ reply: pick(["Alright — dropped. What would you like instead?",
                          "No problem. Tell me what you want to look at.",
                          "Cancelled. I am here when you need me."]), actions: [] });
  }

  if (IDENT.test(n)) return chat(H.identity());
  if (THANKS.test(n)) { MEM.greeted = true; return chat(H.thanks()); }
  if (PRAISE.test(n)) { MEM.greeted = true; return chat(H.smalltalk(raw, c) || H.thanks()); }
  if (SOCIAL.test(n) && !/\b(air|water|soil|plant|design|cost|canopy)\b/.test(n)) {
    const st = H.smalltalk(raw, c);
    if (st) return chat(st);
  }

  // "why?" / "why those" / "explain that" — justify the last recommendation.
  if (/^\s*(why|why (that|those|this|it|then|so)|how come|explain( that| this)?|says who|on what basis)\s*[?.!]*\s*$/.test(n)) {
    if (MEM.lastTopic === "species" || MEM.lastIntent === "design")
      return { reply: "The palette is scored against what was measured at this point, not a fixed list. Each species gets points for: pollution tolerance weighted by the AQI here, drought tolerance weighted by the rainfall, crown size against the existing cover, fit to the stated goal, and a small native bonus. The soil pH and texture filter it further when a profile covers the cell. The order you saw is that score, highest first.", actions: [] };
    if (MEM.lastTopic === "priority")
      return { reply: "The priority score is multi-criteria and numeric — AQI worsening 0.40, NDVI decline 0.35, low green cover 0.15, plantable space 0.10, traffic 0.00. Each cell is normalised per criterion, then weighted and summed. It is not simply the least-green cell today; it is where cover is falling fastest against air that is getting worse.", actions: [] };
    if (MEM.lastTopic === "empty_land")
      return { reply: "Plantable space is a proxy — 1 minus NDVI — because no bare-ground survey is wired into this build. It cannot tell open soil from a warehouse roof, and it says nothing about who owns the ground. The cells were ordered by distance from you, emptiest first as the tiebreak.", actions: [] };
    if (MEM.lastTopic === "air")
      return { reply: "AQI here is Open-Meteo's US-AQI, sampled at nine points across the perimeter and averaged. PM2.5 drives it most of the time. The forecast delta is the trained panel's 12-month projection for this H3 cell, off 42 months of history.", actions: [] };
    return { reply: "Ask that right after a specific answer — a species list, the priority ranking, an air reading — and I will show the working behind it.", actions: [] };
  }

  // "more" / "what else" / "show others" — extend the last list.
  if (/^\s*(more|show more|what else|any (more|others)|others|the rest|next (few)?|keep going)\s*[?.!]*\s*$/.test(n)) {
    if (MEM.species.length) {
      const ctx = window.GV && GV.ctx;
      const goal = MEM.design.goal || (c.goal);
      const ranked = (window.GV && GV.rankedSpecies) ? GV.rankedSpecies(ctx || {}, goal) : [];
      const more = ranked.slice(6, 12);
      if (more.length) {
        MEM.species = MEM.species.concat(more.map(s => ({ name: s.name, bot: s.bot })));
        return { reply: "The next rows down:\n\n" + more.map(s =>
          "**" + s.name + "** (*" + s.bot + "*) — " + Math.round(s.fit * 100) + "% fit" +
          (s.warn ? ". ⚠ " + s.warn : "")).join("\n") +
          "\n\nThese fit less well here — the fit percentage says how much less.", actions: [] };
      }
      return { reply: "That is the whole catalogue that suits this place. The ones I already showed are the picks.", actions: [] };
    }
    if (MEM.cells.length) return H.priority("top " + (MEM.cells.length + 5));
    return null;
  }

  // From here down are terse pointers back at what was just shown — "the
  // second one", "plant it", "make it bigger", "there". A wordy sentence
  // that already classifies as a substantive intent is not one of those.
  const strong = GVE.classify(raw);
  const wordy = n.split(/\s+/).length > 4;
  if (wordy && ["priority","species","design","air","water","canopy","soil","report",
                "compare","view","cost","review","project","empty_land","traffic"].indexOf(strong) >= 0)
    return null;

  // Ordinal / rank reference into the last list ("the second one", "#3",
  // a bare "2"). Cells win when the last answer was a ranking; otherwise
  // it points into the species list.
  const ord = ordinalRef(n);
  if (ord != null) {
    if (MEM.cells.length && (MEM.lastTopic === "priority" || MEM.lastTopic === "empty_land")) {
      const rank = ord === -1 ? MEM.cells[MEM.cells.length - 1] : MEM.cells[ord - 1];
      if (rank != null) return { reply: "Framing cell **#" + rank + "**. Click the hexagon for its full history, or say **plant here** once the map settles.",
        actions: [{ tool: "map.view", args: { view: "priority" } }, { tool: "priority.focus", args: { rank } }] };
    }
    if (MEM.species.length) {
      const s = ord === -1 ? MEM.species[MEM.species.length - 1] : MEM.species[ord - 1];
      if (s) {
        MEM.focusSpecies = s;                         // "plant it" now has a referent
        if (/\b(plant|use|pick|go with|add)\b/.test(n))
          return H.plant("plant 30 " + s.name.replace(/\s*\(.*?\)/, ""), c);
        return { reply: "**" + s.name + "** (*" + s.bot + "*) is #" + ord + " on the list for this place. Say **plant it** to place it, or **design a park** to lay the whole mix out.", actions: [] };
      }
    }
  }

  // "plant it" / "use that one" / "go with it" — the species just singled out.
  if (/^\s*(plant|use|pick|go with|add|place)\s+(it|that|that one|this one|this|them)\s*[.!]*\s*$/.test(n) && MEM.focusSpecies) {
    return H.plant("plant 30 " + MEM.focusSpecies.name.replace(/\s*\(.*?\)/, ""), c);
  }

  // "plant the first three", "plant those", "use the top two" — build the
  // mix straight from MEM.species rather than round-tripping through the
  // text parser, which would latch onto only the first name.
  if (/\b(plant|use|add|place|go with)\b.*\b(those|these|them|the (first|top) (two|three|3|2|few)|the list|the mix|all of (them|those)|that mix)\b/.test(n) && MEM.species.length) {
    const k = /\b(three|3)\b/.test(n) ? 3 : /\b(two|2)\b/.test(n) ? 2 :
             /\bfew\b/.test(n) ? 3 : Math.min(4, MEM.species.length);
    const names = MEM.species.slice(0, k).map(s => s.name);
    const numHit = n.match(/\b(\d{2,4})\b/);
    const total = numHit ? parseInt(numHit[1], 10) : 12 * k;
    const per = Math.max(1, Math.round(total / names.length));
    const mix = names.map((sp, i) => ({ species: sp, count: i === names.length - 1 ? total - per * (names.length - 1) : per }));
    return {
      reply: "Planting **" + total + " trees** — " + mix.map(m => m.count + "× " + m.species).join(", ") +
        ".\n\nIf there is no plot yet I will square one off first. Say **cost** for the bill.",
      actions: [{ tool: "studio.autoplant", args: { mix, spacing_m: 8 } },
                { tool: "dock.open", args: { tab: "studio" } }]
    };
  }

  // Editing the current design in place.
  if (c.hasDesign || MEM.lastIntent === "design") {
    if (/\b(bigger|larger|more space|expand|double it|2x)\b/.test(n)) {
      const area = (MEM.design.area || c.plotM2 || 10000) * (/(double|2x)/.test(n) ? 2 : 1.6);
      return H.design("design a " + (area / 10000).toFixed(2) + " hectare " + (MEM.design.goal || c.goal), c);
    }
    if (/\b(smaller|less space|half it|shrink|too big)\b/.test(n)) {
      const area = Math.max(200, (MEM.design.area || c.plotM2 || 10000) * (/half/.test(n) ? 0.5 : 0.6));
      return H.design("design a " + (area / 10000).toFixed(2) + " hectare " + (MEM.design.goal || c.goal), c);
    }
    if (/\b(more trees|denser|pack (it|them) in|fill it in)\b/.test(n))
      return H.plant("plant 40 trees", c);
    if (/\badd\b.*\b(pond|water|lake)\b/.test(n))
      return { reply: "Adding a pond. Watch the water line in the cost — evaporation in a dry climate is real, so it carries a top-up budget.",
        actions: [{ tool: "studio.elements", args: { elements: [{ id: "pond", qty: Math.round((c.plotM2 || 10000) * 0.06), unit: "m2" }] } }, { tool: "dock.open", args: { tab: "studio" } }] };
    if (/\badd\b.*\b(bench|seat|seating)\b/.test(n))
      return { reply: "Adding benches along the paths.", actions: [{ tool: "studio.elements", args: { elements: [{ id: "bench", qty: 4, unit: "each" }] } }, { tool: "dock.open", args: { tab: "studio" } }] };
    if (/\badd\b.*\b(path|trail|walkway)\b/.test(n))
      return { reply: "Adding a gravel trail loop.", actions: [{ tool: "studio.elements", args: { elements: [{ id: "path_gravel", qty: Math.round((c.plotM2 || 10000) * 0.05), unit: "m2" }] } }, { tool: "dock.open", args: { tab: "studio" } }] };
    if (/\badd\b.*\b(play|playground|swings?|slide)\b/.test(n))
      return { reply: "Adding a play set — it needs an impact-absorbing surface under it, which is in the rate.", actions: [{ tool: "studio.elements", args: { elements: [{ id: "play", qty: 1, unit: "each" }] } }, { tool: "dock.open", args: { tab: "studio" } }] };
    if (/\b(the cost|how much|what.*cost|price)\b/.test(n) && n.length < 30) return H.cost(raw, c);
  }

  // Bare "cost?" / "and the cost" / "how much" with no other subject.
  if (/^\s*(and )?(the )?(cost|price|bill|boq|how much)\s*[?.!]*\s*$/.test(n)) return H.cost(raw, c);
  if (/^\s*(and )?(the )?(review|score|is it any good)\s*[?.!]*\s*$/.test(n) && c.hasDesign) return H.review(raw, c);

  // "there" / "that spot" / "same place" / "again" — repeat the last query.
  if (/\b(there|that (spot|place|area|point)|same (place|spot)|do (that|it) (here|there)|again|once more|redo|repeat)\b/.test(n) && MEM.lastIntent) {
    const h = H[MEM.lastIntent];
    if (h) { const out = h(raw, c); out._repeat = true; return out; }
  }

  // "go back" / "take me back"
  if (/\b(go back|take me back|back to (where|the last)|previous (place|spot))\b/.test(n) && MEM.lastPlace)
    return H.goto("go to " + MEM.lastPlace, c);

  return null;
}

/* ---------- entry point --------------------------------------------- */

GVE.handle = async function (message, context) {
  await GVE.load();
  const c = Ctx(context);
  MEM.turns++;

  const raw = String(message || "").trim();
  const n = normalise(raw);

  // 1) A half-specified request from last turn ("design a…", "compare X
  //    and…") — fold this message into it and re-run that handler, unless
  //    the reader has clearly changed the subject or is being social.
  if (MEM.pending && raw && !AFFIRM.test(n)) {
    const bail = SOCIAL.test(n) || IDENT.test(n) || THANKS.test(n) || DENY.test(n) ||
      /^\s*(help|hi|hello|hey)\b/.test(n);
    if (!bail) {
      const pend = MEM.pending; MEM.pending = null;
      const merged = (pend.text + " " + raw).trim();
      let out;
      try { out = (H[pend.intent] || H.report)(merged, c); } catch (e) { out = null; }
      if (out) return finish(out, pend.intent, c);
    }
  }

  // 2) Follow-up resolution against memory.
  let out = null, intent = null;
  try { out = followUp(raw, c); } catch (e) { out = null; }
  if (out) {
    intent = out._intent || MEM.lastIntent || "report";
  } else {
    // 3) Normal path: classify + handle, with compound "go there, then ask X".
    intent = GVE.classify(message);
    let lead = [];
    if (["goto", "compare", "greet", "help", "unknown"].indexOf(intent) < 0) {
      const place = GVE.extractPlace(message);
      const ll = (typeof window.parseLatLon === "function") ? window.parseLatLon(message) : null;
      if (place && !ll) {
        lead = [{ tool: "map.search", args: { query: place } },
                { tool: "dock.open", args: { tab: "area" } }];
        if (!c.place) c.place = place;
        MEM.lastPlace = place;
      }
    }
    try {
      out = (H[intent] || H.report)(message, c);
    } catch (e) {
      out = { reply: "That question hit an error in the offline planner: " + e.message +
        "\n\nThe full engine handles more than this build does — start it with " +
        "`python -m greenplan.server --config config/city.yaml`.", actions: [] };
    }
    out.actions = out.actions || [];
    if (lead.length) {
      out.actions = lead.concat(out.actions);
      out.reply = "Moving to **" + GVE.extractPlace(message) + "** first, then answering that.\n\n" + out.reply;
    }
  }

  return finish(out, intent, c);
};

/* Common tail: record what happened into memory, set a proposal if the
   handler offered one, stamp the honest source line. */
const CHATTY = { chat: 1 };                 // replies that must not disturb topic memory
const SUPERSEDES = { design: 1, plant: 1, goto: 1, priority: 1, empty_land: 1, compare: 1, view: 1 };

function finish(out, intent, c) {
  out = out || { reply: "", actions: [] };
  out.actions = out.actions || [];

  if (out._pending) { MEM.pending = out._pending; delete out._pending; }

  // Small talk and "why?/more" answers keep the conversation pointed at the
  // last real subject, so a follow-up after them still resolves.
  if (!out._repeat && !CHATTY[intent]) {
    MEM.lastIntent = intent;
    MEM.lastTopic = TOPIC_OF[intent] || MEM.lastTopic;
  }
  delete out._repeat; delete out._intent;

  // A pending "yes" survives small talk and clarifying chatter, but a new
  // substantive command replaces it.
  if (SUPERSEDES[intent]) MEM.proposal = null;
  if (intent === "species" && out.actions.some(a => a.tool === "studio.suggest")) {
    const goal = MEM.design.goal || c.goal;
    MEM.proposal = { run: () => H.design("design a 1 hectare " + goal, c) };
  }

  out.intent = intent;
  out.lang = "en";
  out.dir = "ltr";
  out.source = "offline planner (static build) · " + intent;
  return out;
}

})();
