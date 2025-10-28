"""
Inference script to generate predictions for the test set.
"""

import lightgbm as lgb
import numpy as np

from . import config, constants
from .data_processing import load_and_merge_data
from .features import create_features


def predict() -> None:
    """Generates and saves predictions for the test set.

    This script loads the complete dataset, applies the same feature engineering
    pipeline as in training, then loads the 5 trained models. It averages the
    predictions from all models and saves the result to a submission file.
    """
    # Load and process data
    merged_df, book_genres_df, _ = load_and_merge_data()
    featured_df = create_features(merged_df, book_genres_df)

    # Separate test set for prediction
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    # Define features
    features = [
        col for col in test_set.columns if col not in [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION]
    ]
    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = test_set[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_test = test_set[features]

    # Generate predictions from all fold models
    test_preds = []
    print(f"Loading {config.N_SPLITS} models and generating predictions...")

    for fold in range(config.N_SPLITS):
        model_path = config.MODEL_DIR / config.MODEL_FILENAME_PATTERN.format(fold=fold)
        print(f"Loading model from {model_path}")
        model = lgb.Booster(model_file=str(model_path))
        fold_preds = model.predict(X_test)
        test_preds.append(fold_preds)

    # Average the predictions
    avg_preds = np.mean(test_preds, axis=0)

    # Clip predictions to be within the valid rating range [0, 10]
    clipped_preds = np.clip(avg_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    # Create submission file
    submission_df = test_set[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission_df[constants.COL_PREDICTION] = clipped_preds

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created at: {submission_path}")


if __name__ == "__main__":
    predict()
