# Deploying Green Vision to Cloudflare Pages

Two commands, once you have a Cloudflare account.

```bash
.venv/Scripts/python scripts/build_static.py --config config/city.yaml
npx wrangler pages deploy dist --project-name green-vision
```

The first prints the bundle it wrote. The second asks you to log in the first
time — it opens a browser, you approve, and it remembers. When it finishes it
prints your public URL, of the form:

```
https://green-vision.pages.dev
```

That link is live immediately and free on Cloudflare's free tier.

---

## What you are deploying

`dist/` is ~293 KB and contains no server code.

| | |
|---|---|
| `index.html` | the studio |
| `engine/zones.geojson` | 146 ranked cells — score for all, species and justification for the top `mcda.top_n` (10) |
| `engine/greenloss.json` | the same cells as polygons, green / amber / red |
| `engine/cells.json` | the ranked panel as rows |
| `engine/soil.json` | SoilGrids pH and texture, 120 cells |
| `engine/species.json` | the 30-species knowledge base |
| `engine/meta.json` | city, counts, thresholds, build timestamp |
| `gv-engine.js` | the in-page engine and assistant |
| `data/i18n/*.json` | interface and assistant strings, 5 languages |

## What works on the deployed site

Everything the local build does, with two honest exceptions below.

- The map, search, and the 100 km² area read — live Open-Meteo weather and air
  quality, Open-Meteo archive rainfall, OpenStreetMap census, Esri canopy.
  These are public keyless APIs and are called from the visitor's browser, so
  they work exactly as they do locally.
- **Green view** — live canopy heatmap **and** the engine's 146-cell forecast
  (25 holding, 34 losing cover, 73 already bare), from `engine/greenloss.json`.
- **Priority view** — all 146 ranked cells, from `engine/zones.geojson`.
- **Species matched to the place** — ranked against measured AQI, rainfall,
  days over 40 °C, canopy and goal, and against that cell's soil from
  `engine/soil.json`.
- **The studio** — draw, plant, cost, review, 25-year projection. All of this
  was always client-side.
- **The assistant** — the in-page planner. It reads the baked engine output and
  drives the same tools the Python assistant drives.

## The two honest exceptions

**1. The assistant is narrower than the Python one.**

`web/gv-engine.js` implements the intents people actually type: species,
design, plant, priority, empty land, air, canopy, water, soil, cost, review,
project, goto, view, compare, report. The Python additionally handles a
glossary ("what is NDVI"), data-provenance questions, planting season,
survival rates, carbon, budget-constrained design, maintenance and population
questions.

`scripts/parity_check.py` runs both over one corpus and fails the build if they
disagree on anything in the core set. It currently reports:

```
PASS: 30 core messages agree on intent and place.
9 intent(s) differ in the KNOWN-NARROWER set
```

Every reply carries a `source` field, and the UI prints it — `offline-planner
(static build)` versus `offline-engine`. Which brain answered is never hidden.

**2. The ranking is baked, not recomputed.**

For a fixed city it is a constant: it changes when you re-run training on new
data, not when a visitor clicks. Re-run `build_static.py` and redeploy to
refresh it. `engine/meta.json` carries `built_utc` so you can always tell how
old a deployed ranking is.

## Updating

```bash
.venv/Scripts/python scripts/build_static.py --config config/city.yaml
npx wrangler pages deploy dist --project-name green-vision
```

Same project name redeploys to the same URL. `dist/_headers` sets the engine
payloads to a one-hour cache and `index.html` to `no-cache`, so a redeploy is
visible immediately rather than after a stale-cache wait.

## Custom domain

In the Cloudflare dashboard: **Workers & Pages → green-vision → Custom
domains**. If the domain is already on Cloudflare, DNS is written for you.

## Checking a deploy

```bash
curl -s https://green-vision.pages.dev/engine/meta.json | python -m json.tool
```

`zones` should read 146 and `built_utc` should be recent. If the site loads but
the Priority view is empty, that file is what to look at first.

## Deploying somewhere else

`dist/` is plain static files — GitHub Pages, Netlify, S3 or any web server
will serve it. Only `_headers` and `_redirects` are Cloudflare-specific, and
both are optional.

## Running the full engine instead

If you want the complete Python assistant — the wider intent set, live
retraining, per-point soil lookup at arbitrary coordinates — run it locally:

```bash
start.bat
```

and open http://127.0.0.1:8000. The page prefers a reachable engine over the
in-page planner automatically, including when you open `index.html` straight
off the filesystem.
