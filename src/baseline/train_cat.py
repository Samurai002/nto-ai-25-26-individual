"""
Main training script for the model (CatBoost instead of LightGBM).

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import optuna
from catboost import CatBoostRegressor

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def train() -> None:
    """Runs the model training pipeline with temporal split.

    Loads prepared data from data/processed/, performs temporal split based on
    absolute date threshold, computes aggregate features on train split only,
    and trains a single CatBoost model (hyperparameters tuned with Optuna).

    Note: Data must be prepared first using prepare_data.py
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train and test sets
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Check for timestamp column
    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set. "
            "Make sure data was prepared with timestamp preserved."
        )

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Perform temporal split
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    # Split data
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Verify temporal correctness
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"Max train timestamp: {max_train_timestamp}")
    print(f"Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError(
            f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
            f"is not greater than max train timestamp ({max_train_timestamp})."
        )
    print("✅ Temporal split validation passed: all validation timestamps are after train timestamps")

    # Compute aggregate features on train split only (to prevent data leakage)
    print("\nComputing aggregate features on train split only...")
    train_split_with_agg = add_aggregate_features(train_split.copy(), train_split)
    val_split_with_agg = add_aggregate_features(val_split.copy(), train_split)  # Use train_split for aggregates!

    # Handle missing values (use train_split for fill values)
    print("Handling missing values...")
    train_split_final = handle_missing_values(train_split_with_agg, train_split)
    val_split_final = handle_missing_values(val_split_with_agg, train_split)

    # Define features (X) and target (y)
    # Exclude timestamp, source, target, prediction columns
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in train_split_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_train = train_split_final[features]
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features]
    y_val = val_split_final[config.TARGET]

    print(f"Training features: {len(features)}")

    # --- Определяем категориальные фичи для CatBoost ---
    cat_feature_names = [
        col for col in X_train.columns
        if str(train_split_final[col].dtype) == "category"
    ]
    cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_feature_names]
    if cat_feature_indices:
        print(f"CatBoost categorical features: {cat_feature_names}")
    else:
        print("No categorical features detected for CatBoost.")

    # --- Сэмпл 30% данных для подбора гиперпараметров Optuna ---
    sample_frac = 0.3
    if len(X_train) > 0:
        sample_indices = X_train.sample(frac=sample_frac, random_state=config.RANDOM_STATE).index
        X_train_sample = X_train.loc[sample_indices]
        y_train_sample = y_train.loc[sample_indices]
        print(f"Using {len(X_train_sample):,} rows ({sample_frac*100:.0f}%) for Optuna hyperparameter tuning.")
    else:
        X_train_sample = X_train
        y_train_sample = y_train
        print("Train set is empty, skipping sampling for Optuna.")

    # --- Optuna: подбор гиперпараметров для CatBoost ---
    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_state": config.RANDOM_STATE,
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "task_type": "CPU",
            "thread_count": -1,
        }

        model = CatBoostRegressor(**params)
        model.fit(
            X_train_sample,
            y_train_sample,
            eval_set=(X_val, y_val),
            verbose=500,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            cat_features=cat_feature_indices if cat_feature_indices else None,
        )

        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        return rmse

    print("\nStarting Optuna hyperparameter search for CatBoost...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print(f"\nBest trial RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # --- Обучение финальной модели ---
    best_params = study.best_params.copy()
    best_params.update(
        {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_state": config.RANDOM_STATE,
            "task_type": "CPU",
            "thread_count": -1,
        }
    )

    print("\nTraining final CatBoost model on full training data...")
    final_model = CatBoostRegressor(**best_params)

    final_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=100,
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        cat_features=cat_feature_indices if cat_feature_indices else None,
    )

    # Evaluate the model
    val_preds = final_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    print(f"\nValidation RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Save the trained model in .cbm format
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    final_model.save_model(str(model_path), format="cbm")
    print(f"Model saved to {model_path} (CatBoost .cbm format)")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
