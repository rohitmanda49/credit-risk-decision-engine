"""Load model artifacts once at application startup."""
import joblib
import lightgbm as lgb
from pathlib import Path
from functools import lru_cache

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class ModelBundle:
    """Container for all model artifacts."""
    def __init__(self):
        self.model: lgb.Booster = None
        self.calibrator = None
        self.metadata: dict = None
        self.decision_policy: dict = None

    def load(self):
        self.model = lgb.Booster(model_file=str(MODELS_DIR / "lgb_tuned.txt"))
        self.calibrator = joblib.load(MODELS_DIR / "isotonic_calibrator.pkl")
        self.metadata = joblib.load(MODELS_DIR / "metadata.pkl")
        self.decision_policy = joblib.load(MODELS_DIR / "decision_policy.pkl")


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """Returns the singleton model bundle (loaded once)."""
    bundle = ModelBundle()
    bundle.load()
    return bundle