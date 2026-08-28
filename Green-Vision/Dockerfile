FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY greenplan/ greenplan/
COPY config/ config/
COPY data/ data/
COPY index.html .
# index.html loads /gv-engine.js unconditionally, and greenplan/server.py routes
# that path to web/gv-engine.js. Without this COPY the route 404s in the image,
# which silently removes the in-page planner the page falls back to when the
# engine dies mid-session — the reason that route exists at all.
COPY web/ web/

# outputs/ is NOT copied: it is in .dockerignore, so it is not in the build
# context and `COPY outputs/ outputs/` failed the build outright. It does not
# need copying — Engine._build_and_train calls recommend(), which writes
# outputs/<city>/ at startup. It only has to exist and be writable.
RUN mkdir -p models outputs

ENV PYTHONUNBUFFERED=1

# The API server. Serves the ranking, per-cell reasoning and species picks over
# HTTP so the studio can call a live engine instead of reading a static export.
#
# PORT is injected by most hosts (Render, Railway, Fly); 8000 locally.
# Bind 0.0.0.0 or the host's health check never reaches us.
#
# MODEL_PROVIDER, read by greenplan/server.py at startup: when it is set and
# non-empty it overrides `model.provider` from config/city.yaml, and the
# override is logged at INFO so the container log says which provider actually
# won. The value is not validated beyond non-emptiness — an unknown name
# degrades to the offline engine through the same graceful path a missing
# OpenVINO runtime takes, rather than crashing the server.
# This variable was set and documented here for a long time while nothing in
# the codebase read it, so setting it was a silent no-op.
#   mock      - no model, no download, ~200 MB RAM. The default here, because
#               free tiers cannot hold a 1 GB model in memory.
#   openvino  - local INT4 inference. Needs ~2 GB RAM and the model baked in
#               or fetched on boot; use a paid tier or a Hugging Face Space.
#   nvidia    - hosted model, needs NVIDIA_API_KEY. Lightest on RAM.
ENV MODEL_PROVIDER=mock

CMD ["sh", "-c", "python -m greenplan.server --config config/city.yaml --host 0.0.0.0 --port ${PORT:-8000}"]
