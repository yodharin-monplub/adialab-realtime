"""Model 5 access to the shared leakage-safe feature-selection engine.

The implementation is kept in Model 4 so both experiments use exactly the
same ranking, cross-validation, winner-freezing, and holdout logic. Passing a
``Model5Config`` gives every checkpoint a Model 5-specific version and path.
"""

from models.model4.feature_selection import (
    FeatureSelectionConfig,
    build_feature_rankings,
    choose_feature_winner,
    evaluate_feature_selected_holdout,
    feature_search_table,
    fit_feature_selected_model,
    freeze_feature_winner,
    search_feature_subsets,
)


__all__ = [
    "FeatureSelectionConfig",
    "build_feature_rankings",
    "choose_feature_winner",
    "evaluate_feature_selected_holdout",
    "feature_search_table",
    "fit_feature_selected_model",
    "freeze_feature_winner",
    "search_feature_subsets",
]
