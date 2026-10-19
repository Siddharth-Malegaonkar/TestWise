import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}. Run scripts/train_model.py first.')

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError('This model does not expose feature_importances_.')

    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None:
        raise AttributeError('Model was trained without feature names; cannot print important features.')

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(10)

    print('Top 10 most important features:')
    for name, value in top.items():
        print(f'{name}: {value:.6f}')


if __name__ == '__main__':
    main()
