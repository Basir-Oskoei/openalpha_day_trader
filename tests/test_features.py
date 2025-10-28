
import pandas as pd, numpy as np
from app.features import compute_features

def test_compute_features_minimal():
    idx = pd.date_range("2023-01-01", periods=300, freq="T")
    df = pd.DataFrame({
        "Open": np.random.rand(300)+100,
        "High": np.random.rand(300)+101,
        "Low": np.random.rand(300)+99,
        "Close": np.random.rand(300)+100,
        "Volume": np.random.randint(100, 10000, size=300)
    }, index=idx)
    feats = compute_features(df)
    assert not feats.empty
    for c in ["sma20","sma50","sma200","atr14"]:
        assert c in feats.columns
