"""The numeric model bake-off, and the hybrid that deploys its winner.

Three families have been scored on the same held-out task (predict a cell's
AQI/NDVI 12 months out, history strictly before the cutoff, skill measured
against a Theil-Sen + seasonality baseline):

    statistical forecaster (trend + season + memory correction)   skill +0.08
    Qwen2.5-1.5B INT4 LLM, in-context                             skill -0.33
    trained MLP on the panel (this package, `train`)              skill -1.41
    ridge on zone-relative residuals (this package's diagnostics) skill -0.03

With 42 months of history, city-wide shocks dominate the residual and no
richer model finds signal the robust baseline misses. So the `hybrid`
provider deploys the statistical forecaster for NUMBERS and the local LLM for
WORDS — and `build_model` re-checks the evidence on every run: if a
challenger trained on a longer panel ever reports positive held-out skill,
it is deployed automatically. The challenger harness (ONNX export, OpenVINO
inference on CPU/GPU/NPU) is production-ready and waiting for more data.

Train / re-score the challenger (writes models/{city}/forecaster/):
    python -m greenplan.forecast.train --config config/city.yaml
"""

from .features import FEATURE_NAMES, feature_vector
from .ovmodel import HybridModel, OVForecaster

__all__ = ["FEATURE_NAMES", "feature_vector", "OVForecaster", "HybridModel"]
