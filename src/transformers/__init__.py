"""
src.transformers — reusable sklearn-compatible transformers.
All transformers implement TransformerMixin and BaseEstimator and can be used
in ColumnTransformer or sklearn Pipeline.

Pipeline order
--------------
1. ColumnTransformer (per-feature-group encoding)
   ├── CyclicalEncoder        — hour_created, day_of_week_created
   ├── SafeTargetEncoder      — project_code
   ├── OrdinalEncoder         — project_category (Personal/Work → 0/1)
   ├── ScalerSelector         — numeric features (log + scale)
   └── TextVectorizerTransformer — desc_clean (TF-IDF/BoW + optional LSA)
2. FeatureSelectorTransformer — SelectKBest (f_classif / mutual_info) or none
3. DimensionalityReducerTransformer — PCA / LDA / none
4. Classifier

Exports
-------
CyclicalEncoder                — sin/cos encoding for periodic features
SafeTargetEncoder              — target encoding with unseen-category fallback
TextVectorizerTransformer      — BoW/TF-IDF + optional LSA
ScalerSelector                 — configurable scaler (standard/robust/minmax/none)
ResamplerTransformer           — SMOTE/ADASYN/none (used OUTSIDE pipeline)
FeatureSelectorTransformer     — SelectKBest or passthrough (inside pipeline)
DimensionalityReducerTransformer — PCA/LDA/none (inside pipeline)
"""
from .cyclical_encoder import CyclicalEncoder
from .dimensionality_reducer import DimensionalityReducerTransformer
from .feature_selector import FeatureSelectorTransformer
from .resampler import ResamplerTransformer
from .scaler_selector import ScalerSelector
from .target_encoder import SafeTargetEncoder
from .text_vectorizer import TextVectorizerTransformer

__all__ = [
    "CyclicalEncoder",
    "SafeTargetEncoder",
    "TextVectorizerTransformer",
    "ScalerSelector",
    "ResamplerTransformer",
    "FeatureSelectorTransformer",
    "DimensionalityReducerTransformer",
]
