"""
This module serves as a backward-compatibility wrapper.
The classes and functions have been refactored into:
- src.preprocessing
- src.feature_selection
- src.evaluation
"""

from src.preprocessing import (
    ColumnDropper,
    OneHotFeatureEncoder,
    StandardFeatureScaler,
    OrdinalFeatureEncoder,
    TargetFeatureEncoder,
    FeatureEngineer
)

from src.feature_selection import (
    RecursiveFeatureEliminator
)

from src.evaluation import (
    classification_kfold_cv,
    plot_classification_kfold_cv,
    evaluate_classifier,
    plot_feature_importances,
    precision_vs_recall_curve,
    get_threshold_metrics,
    plot_probability_distributions,
    probability_scores_ordering
)
