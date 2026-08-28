# Green Vision

**Where should this city plant trees next, and what should go in the ground?**

Green Vision answers that in two parts that share one subject, one grid and
one process:

| Part | What it is | Runs |
|---|---|---|
| `greenplan/` | **The priority engine.** Ranks every hexagonal cell in the city by where green cover is declining fastest against worsening air quality, picks species against soil and pollution, and answers the assistant. This is the AI. | Python, offline, one command |
| `index.html` | **The design studio.** Click a place, read live conditions across 100 km², draw a plot, place trees, cost it, project 25 years. | Browser |

### Run it

```bash
start.bat        # Windows
./start.sh       # macOS / Linux
```

Then open **http://127.0.0.1:8000**.

That is the whole thing. The script makes the virtual environment, installs
five dependencies, trains on the 42-month panel and serves the studio and the
engine from **one origin** — so the page calls `/api/...` with no CORS, no
hard-coded host and no key, and the ranking you see is what this machine
computed a moment ago rather than a stale export someone remembered to commit.

**No API key is needed, ever.** Nothing in the default path calls a paid
service, and with the engine running locally no figure about your city leaves
your machine. See [Privacy, stated accurately](#privacy-stated-accurately) for
what *does* go out, because the map layer is not offline and saying otherwise
would be false.

### Put it on the internet

```bash
.venv/Scripts/python scripts/build_static.py --config config/city.yaml
npx wrangler pages deploy dist --project-name green-vision
```

That bakes the trained engine down to ~293 KB of static JSON and publishes it
to Cloudflare Pages, free, on a public `*.pages.dev` URL. Everything works
there: the map, the search, the green forecast, all 146 priority cells,
soil-aware species, the studio, and an in-page assistant that drives the same
tools the Python one does. See **[DEPLOY.md](DEPLOY.md)** for what is baked,
what is narrower on the static build, and how the two are kept in step.

#### Opening `index.html` directly still works, partially

The studio degrades honestly on `file://`: live weather, air quality, the
OpenStreetMap census and the canopy heatmap all work, because those are public
keyless APIs. What needs the engine — the **assistant**, the **priority
ranking**, the **forecast layer** and **soil-aware species matching** — says so
rather than inventing an answer. Run `start.bat` to get all of it.

---

## Part 1 — the priority engine

### Run it right now (no API key, no cost)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m greenplan run --config config/city.yaml --mock --recommend
```

`--mock` swaps in synthetic adapters **and** an offline stand-in model, so the
whole pipeline runs with zero network calls. To run on the **real** Ahmedabad
data with the stand-in model, set `model.provider: mock` in the config and drop
`--mock` — the CSV adapters then load and you get a genuine 146-cell result.

### Run it with a real model — locally, on Intel OpenVINO

Optional. No API key, no network, no per-token cost. Without it the engine
uses its deterministic writer, which — see the table below — is the *more
accurate* of the two at the numeric half of the job.

```bash
.venv/Scripts/pip install -r requirements-openvino.txt   # runtime, one time
python scripts/fetch_openvino_model.py                   # ~1 GB, one time
.venv/Scripts/python -m greenplan run --config config/city.yaml --recommend
```

`config/city.yaml` ships with `provider: openvino`, and that is safe with or
without the two steps above: if the runtime or the weights are missing the
engine logs a warning naming exactly what it could not load and continues on
the offline writer. It used to raise and refuse to start, which took the whole
server down over an optional dependency.

Intel publishes instruct models already converted to OpenVINO IR and
weight-compressed to **INT4**, which is what makes this practical: ~1 GB on
disk instead of ~3 GB at FP16, **2.5 s to load and ~4 s per reply on a plain
CPU**, no GPU anywhere. `model.device` passes straight through to OpenVINO, so
the same build targets `CPU`, an integrated `GPU`, or an `NPU` unchanged.

`scripts/fetch_openvino_model.py --list` shows the catalogue (TinyLlama 1.1B →
Phi-3.5-mini 3.8B).

### Or with a hosted model

```bash
$env:NVIDIA_API_KEY="..."        # Linux/macOS: export NVIDIA_API_KEY=...
.venv/Scripts/python -m greenplan run --config config/city.yaml --recommend
```

Set `model.provider` to `nvidia` (NIM, `meta/llama-3.1-70b-instruct`) or
`openrouter`. Keys are read from the environment and never hard-coded.

### Measured: local model vs. the offline writer

**These are two different runs, and the table says so rather than blending
them.** Only the right-hand column is reproducible from this repository. The
left needs the ~1 GB OpenVINO model, which is not in the repo, and its training
artifact was not kept.

| | Qwen2.5-1.5B INT4 (local)<br>*separate run, artifact not in this repo* | Offline stand-in (Theil–Sen + seasonality)<br>*the run this repo ships* |
|---|---|---|
| Backtest iterations | 30 | 60 |
| Horizon | 12 months | 12 months |
| MAE — AQI | 33.3 | **26.49643289681824** |
| MAE — NDVI | 0.103 | **0.028172118623493957** |
| MAE — traffic | n/a | 3.372718854765962 |
| Skill vs baseline | −0.385 → −0.333 | **+0.03219564546512241 → +0.05292599349570087** |
| `memory_helped` | false | true |
| Malformed JSON replies | 0 / 30 | n/a |

The right-hand column is read straight out of the artifact in this repo,
`models/ahmedabad/memory.meta.json` (`trained_at`
`2026-08-27T11:19:56.074699+00:00`, `memory_records` 60, matching the 60 lines
in `models/ahmedabad/memory.jsonl`), and quoted at the precision that file
stores rather than tidied. Its 60 iterations are (zone, cutoff) backtest
samples over the real 146-cell × 42-month panel. Traffic is the inert
placeholder with MCDA weight 0, so its MAE scores nothing that reaches a
ranking. Re-run `--recommend` and you get this column back. You cannot get the
left-hand column back from what is here.

**Read this before quoting any accuracy number.** A 1.5B model is *worse than
a trend line* at numeric extrapolation, and no amount of prompting fixes that —
it is the wrong tool for the regression half of the task. What it does do
reliably is produce well-formed output (0 repairs needed in 30 iterations) and
write the per-cell justification and species picks, which is the half a
language model is actually good at. Its memory loop does work — skill improves
+0.052 from the first half of training to the second — it simply starts from
too far behind to overtake the baseline. Every figure in that sentence and in
the left-hand column comes from the un-shipped 30-iteration run, so quote them
as that and nothing more. The shipped run's memory loop also helps, by a
smaller margin: `skill_gain_first_to_second_half` 0.0207.

If you need the ranking to be as accurate as possible, run a hosted model. If
you need it to run anywhere, for free, with nothing leaving the machine, run
OpenVINO. That trade is real and this table is the honest version of it.

### What a run does

1. **Adapters** load three per-cell monthly streams:
   - `data/ahmedabad_ndvi.csv` — **real** NASA MOD13Q1 vegetation index (250 m,
     16-day) via ORNL DAAC, keyless
   - `data/ahmedabad_aqi.csv` — **real** US AQI from the Open-Meteo Air-Quality
     archive, keyless
   - `data/ahmedabad_traffic_placeholder.csv` — an **inert placeholder**. No
     free historical traffic source exists for arbitrary coordinates. Its MCDA
     weight is `0.0`, so it never influences a ranking. Disclosed, not real.
2. **Grid** merges all three onto H3 resolution 7 (~5 km² cells).
   Measured panel: **146 cells × 42 months, 2023-01 → 2026-06.**
3. **Training** backtests: predict a past month from history strictly before a
   cutoff, score against what really happened, ask the model for a one-line
   lesson, append to `models/{city}/memory.jsonl`. Later prompts retrieve the
   most relevant records. **In-context learning — no weight updates, no
   fine-tuning, no retraining.**
4. **`--recommend`** ranks **all** cells by a numeric MCDA score, has the model
   justify the ranking and pick species for the top `mcda.top_n` of them — 10
   in the shipped config, not all 146 — and writes `recommendations.geojson`,
   `recommendations.csv`, and `planting_brief.txt`. All 146 cells reach the
   GeoJSON (`mcda.geojson_n: 0`); each one carries a `reasoned` flag saying
   which kind it is.

**Which model writes the justifications depends on how you started it.** The
command line above is one path. The other — the one `start.bat` / `start.sh`
run, and therefore the one nearly every reader actually experiences — is
`greenplan.server`, and its startup pass calls `build_model(cfg.model, True)`.
That `True` forces the offline deterministic writer for the whole 146-cell
batch regardless of `model.provider`. It is deliberate, not a bug: the batch is
one forecast per cell plus the training loop, ~270 requests at startup, and
this account answers that with HTTP 429. The configured provider is used for
per-click reasoning only — `POST /api/recommend`, the handful of cells someone
actually clicks. So with `provider: openvino` set and the weights installed,
the zone justifications in the served Priority view still come from the offline
writer; the OpenVINO model answers your clicks.

### MCDA weights (`config/city.yaml`)

| Criterion | Weight |
|---|---|
| AQI worsening | 0.40 |
| NDVI decline | 0.35 |
| Low green cover | 0.15 |
| Plantable space | 0.10 |
| Traffic worsening | **0.0** (inert placeholder) |

### Validation honesty

A horizon can only be *checked* if it fits inside the data:

```bash
.venv/Scripts/python -m greenplan horizon --config config/city.yaml
# 42 months of history, min_history=18 -> largest honestly validatable horizon: 23 months.
```

Requesting a longer `--horizon` fails loudly. Anything beyond the ceiling must
go through `--project N`, which is labeled **UNVALIDATED** in the output file
itself.

### Soil

Real, and wired in. `data/ahmedabad_soilgrids.csv` holds pH, sand/silt/clay,
organic carbon and nitrogen per H3 cell from **ISRIC SoilGrids v2.0** (250 m,
free, no key). Species selection respects soil pH and texture alongside
pollution load. Refresh with:

```bash
python scripts/soilgrids_export.py --config config/city.yaml   --out data/ahmedabad_soilgrids.csv
```

SoilGrids masks built-up land, so cells in the dense core come back empty at
their centre; the exporter re-samples four offsets around each miss and
recovers most of them. Whatever is still empty falls through to pollution-only
matching rather than inventing a value.

### Still not wired up

- **Soil moisture.** `soil.moisture_csv` stays commented out: NASA SMAP needs
  an Earthdata login, which breaks the no-key property.
- **Bare-ground site finder.** `sites.enabled: true` but `candidates_csv` is
  commented out, so it falls back to the `1 - NDVI` plantable proxy and emits no
  `planting_sites.geojson`.
- **10 years of history.** Not possible from free keyless sources. Open-Meteo's
  air-quality archive returns nothing before **2023-01** (verified back to
  2013), which caps the panel at 42 months regardless of how far MODIS goes
  back. Any claim of a decade of data is wrong.

---

## Part 2 — the design studio

- **The assistant.** Type what you want; it does it. "Design a 1-hectare park
  for a school here" sets the goal, draws the plot, plants a mixed 79-tree
  scheme chosen for this cell's air quality, rainfall and soil, adds paths,
  meadow, benches, lighting and a recharge pit, and opens the studio — then
  tells you what it did, in a visible action log, so the map never moves for a
  reason you cannot see. "Where are the top 3 cells to plant in" switches to
  the Priority view and focuses rank 1. "Take me to Vastrapur" searches, flies
  and reads the 100 km² around it.

  **How it works, precisely** — because "AI" does a lot of unearned work in
  most product copy. The browser posts your message plus a snapshot of what it
  has *measured*. The engine classifies the request, answers it from the
  trained 42-month panel and those live readings, and returns prose plus a list
  of **tool calls**. The page executes those calls against the real map and the
  real studio state — the same code paths the buttons use, so everything the
  assistant builds stays editable by hand, and costs, reviews and projects
  identically.

  The model never produces a figure. Every number in an assistant reply comes
  from the panel or from the page's own measurements; the language layer only
  chooses words and picks species from a fixed table. With no local model
  installed the prose comes from the engine's deterministic writer, which is
  the default and needs nothing. Install the OpenVINO extras and it is a local
  INT4 model instead. Either way nothing leaves the machine.

- **100 km² area of interest.** Clicking the map draws a circle of
  r = √(100/π) km ≈ 5.64 km (`GV.CFG.AOI_KM2`) and scopes every reading to it,
  sampling air quality at 9 points spread across the area. **Any click more
  than 150 m from the last one re-reads everything** (`GV.CFG.AOI_REFRESH_M`).
  That threshold used to be a quarter of the AOI radius — 1,410 m — which
  silently reused the previous area's air quality, canopy, census, cost and
  review for any click inside a 1.4 km circle. Different neighbourhood, same
  numbers, new title. It is fixed, and 150 m now exists only to swallow
  double-clicks.

- **Live, keyless data.** Open-Meteo weather + air quality, Open-Meteo archive
  (2020–2024) for rainfall, OpenStreetMap via Overpass for the feature census,
  Nominatim for place names and search, Esri World Imagery for canopy.

- **Search.** Type a place, an address, or paste `23.0225, 72.5714` in any
  common coordinate format. Debounced, keyboard-navigable (↑/↓/Enter/Esc),
  results biased toward the current view but never restricted to it, with
  recent searches remembered on the device.

- **Green view — two layers, and they answer different questions.**
  - *Canopy now*: `vegScore()` computes a greenness index from the RGB of Esri
    tiles and renders a red→green heatmap, with a live readout of the canopy
    percentage in view and how much of it reads as bare. Tile zoom is chosen
    to fit a budget, so the view works at city scale and street scale instead
    of refusing to render.
    This is **current** imagery — the studio has no historical satellite stack.
  - *Engine forecast*: the 146 ranked H3 cells, tagged **green** (vegetated and
    holding), **amber** (green today, forecast to decline) and **red** (already
    bare). Amber is the whole point, and it is the one thing a snapshot
    physically cannot show — it comes from the NDVI trend, not from today's
    pixels. Click a hexagon for its history.

- **Species matched to the place, not a fixed list.** The palette is **ranked**
  for wherever you are, against measured AQI, the five-year rainfall normal,
  days over 40 °C, existing canopy and your stated goal — and, when the engine
  is running, against that cell's **SoilGrids pH and texture** too. Each chip
  carries a fit bar and its reason on hover; a poor fit is shown as a poor fit
  rather than hidden, because "don't plant this here" is useful. Toxic species
  are flagged and demoted wherever children are the point (schools, community
  gardens, residential streets).

  This was previously the same sixteen species in the same order everywhere on
  Earth, which made the product's central question — *what should go in the
  ground here* — unanswerable.

- **Cost, priced for the site.** Indicative 2026 Indian rates, and the
  three-year establishment water is scaled against the site's own rainfall and
  heat (`waterFactor()`), which is the largest single line in that phase. The
  same 1-hectare design costs about **₹70 L** where 210 mm falls and **₹56 L**
  where 2,600 mm does, and the cost panel shows the multiplier and why. Before
  this, `computeCost()` accepted the climate context and never read it.

- **Review, against the site's own water budget.** Twelve checks encoding
  published practice (Santamour 10/20/30, mature-crown spacing, shade over
  walking routes). The water-balance check runs demand against what actually
  falls on the plot, so the identical design scores **74 with a critical water
  flaw** in a desert and **83 with that flaw gone** in a wet climate.

- **25-year projection.** Explicitly labeled `PROJECTED, not forecast` and
  `UNVALIDATED` in the source — logistic canopy growth, survival curves,
  species lifespan, saturating cooling. Treat it as a defensible shape, not a
  prediction.

- **Priority view.** All **146** ranked H3 cells as colour-coded hexagons —
  warm where planting is most urgent, deliberately *not* the green ramp used
  for current canopy. Click a cell for its priority score and the predicted AQI
  and canopy change, then hand off to the Studio to design on it.

  **Ten of the 146 carry model reasoning; the rest carry the ranking only.**
  `mcda.top_n` in `config/city.yaml` is 10, and that is how many cells the
  model is asked to justify and pick species for. In the shipped
  `outputs/ahmedabad/recommendations.geojson` exactly 10 features have
  `reasoned: true`; the other 136 have an empty `species` array and an empty
  `justification`. Every feature carries a per-feature `reasoned` flag so a
  reader of the file can tell the two apart, and the map degrades honestly on
  the other 136: it prints "the engine recorded no justification for this cell"
  rather than an empty box, and drops the "Plant here" species section
  altogether rather than showing an empty one. (It decides that from the empty
  fields themselves; nothing in `index.html` reads the `reasoned` flag yet.)
  Raise `mcda.top_n` and re-run to cover more — the top rows go to the model in
  one call, so `model.max_new_tokens` is what bounds how far you can push it.

  Served by `greenplan.server`, it loads from `/api/zones` — live from this
  machine. On a plain file server it falls back to
  `outputs/<city>/recommendations.geojson`. Opened as `file://` the browser
  forbids that read, so the page asks you to pick the file — parsed in-page,
  never uploaded.

- **Seven languages.** English, Hindi, Gujarati, Marathi, Bengali, Spanish and
  French — the whole interface, including the assistant's replies and the
  species reasoning, not just the labels. The picker switches the live page in
  place; it does not reload.

  `data/i18n/index.json` used to register thirteen, but a language is listed
  only when `data/i18n/<code>.json` exists — the registered-but-fileless codes
  fell back to English silently, so picking Tamil got you English with no
  notice, and Urdu was registered `"dir": "rtl"` while never resolving to Urdu,
  so right-to-left layout never switched on. `/api/health` and the baked
  `meta.json` count what is in the registry, so they now report seven.

- **Storage.** Designs are saved to browser LocalStorage. Nothing is uploaded
  unless you set `COMMUNITY_URL` and publish deliberately.

### Configuration

There are **two** config objects in `index.html`:

| Object | Line | Holds |
|---|---|---|
| `CFG` | ~355 | map start, tile URLs, Overpass mirrors |
| `GV.CFG` | ~1968 | `AOI_KM2`, `AOI_REFRESH_M`, `RATES`, `TOMTOM_KEY`, `AUTH_URL`, `COMMUNITY_URL`, `GOOGLE_CLIENT_ID` |

```javascript
GV.CFG = {
  AOI_KM2: 100,
  AOI_REFRESH_M: 150,     // how far a click must move before everything re-reads
  AUTH_URL: "",           // blank => local-only sign-in
  COMMUNITY_URL: "",      // blank => gallery shows only your own designs
  GOOGLE_CLIENT_ID: "",   // for "Continue with Google"
  TOMTOM_KEY: "",         // blank => traffic is MODELLED from OSM topology, not measured
  RATES: { labour_day: 650, mali_month: 14000, water_kl: 45,
           contingency: 0.12, design_fee: 0.08, gst: 0.18 }
};
```

Editing `RATES` re-bases every number in the Cost tab, so put your own tender
rates in and the whole estimate follows.

### The HTTP surface

`greenplan.server` serves the studio and the engine from one origin.

| Endpoint | Returns |
|---|---|
| `GET /api/health` | engine status, city, model actually in use, zone and memory counts, greenloss tally, languages |
| `GET /api/zones` | `recommendations.geojson` — all ranked cells, live from this machine |
| `GET /api/greenloss` | every ranked cell as an H3 polygon tagged green / amber / red from the NDVI forecast |
| `GET /api/cells` | the ranked panel as rows |
| `GET /api/languages` | available interface languages |
| `POST /api/recommend` | species + justification for one clicked point, using that cell's soil |
| `POST /api/species` | the species table ranked for one location, with per-species reasons |
| `POST /api/assistant` | prose + tool calls for one assistant message |

Every one of them works with no key. `/api/health` names the model that is
*actually* answering — `offline-engine` when no local weights are installed,
`openvino-local` when they are — rather than whatever the config asked for.

### Privacy, stated accurately

Designs never leave the browser, and the engine's reasoning never leaves the
machine — no key, no endpoint, no per-cell figures in anyone's logs. That holds
for the default offline engine and for `provider: openvino` alike.

What still goes out is the map itself: every click sends coordinates to
Open-Meteo, Overpass, Nominatim and Esri, plus unpkg/cdnjs for libraries.
Switching `model.provider` to a hosted option also sends per-cell aggregates to
that provider.

So: **local-first, and fully local for inference — but the map layer is not
offline.** Say that, rather than "runs entirely on the host computer".

---

## Repo notes

- `start.bat` / `start.sh` are the intended entry point: venv, deps, train,
  serve, one command.
- `requirements.txt` is the core five. The OpenVINO extras are split into
  `requirements-openvino.txt` **on purpose** — the server degrades to the
  offline engine with a logged warning when they are absent, so nobody has to
  install a gigabyte of runtime to see the product work. It used to raise and
  refuse to start.
- `Front_End.html` is the earlier committed UI. `index.html` supersedes it and
  is the one the server serves.
