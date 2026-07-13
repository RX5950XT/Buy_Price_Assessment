"""逐月 expanding-window 機率與價格門檻模型。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from buy_price_assessment.modeling import expanding_month_splits

REQUIRED_COLUMNS = {
    "month",
    "date",
    "adjusted_open",
    "previous_adjusted_close",
    "remaining_min_log_ratio",
    "near_optimal",
}


@dataclass(frozen=True)
class WalkForwardConfig:
    """固定而可重現的第一版模型設定。"""

    initial_months: int = 60
    classifier_c: float = 0.1
    quantile_estimators: int = 60
    quantile_learning_rate: float = 0.04
    quantile_max_depth: int = 2
    quantile_min_samples_leaf: int = 20
    reservation_buffer: float = 0.005
    random_state: int = 42


def _validate_input(
    frame: pd.DataFrame, features: Sequence[str], config: WalkForwardConfig
) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"walk-forward 資料缺少欄位：{sorted(missing)}")
    missing_features = set(features).difference(frame.columns)
    if missing_features:
        raise ValueError(f"walk-forward 缺少特徵：{sorted(missing_features)}")
    if not features:
        raise ValueError("feature_columns 不可為空")
    if config.initial_months < 1:
        raise ValueError("initial_months 必須大於零")
    if config.quantile_estimators < 1:
        raise ValueError("quantile_estimators 必須大於零")
    if not 0.0 <= config.reservation_buffer <= 0.1:
        raise ValueError("reservation_buffer 必須介於 0 與 0.1")


def _month_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("month")["month"].transform("size").to_numpy(dtype=float)
    return 1.0 / counts


def _classifier(config: WalkForwardConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.classifier_c,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _quantile_regressor(config: WalkForwardConfig) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "regressor",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=0.5,
                    n_estimators=config.quantile_estimators,
                    learning_rate=config.quantile_learning_rate,
                    max_depth=config.quantile_max_depth,
                    min_samples_leaf=config.quantile_min_samples_leaf,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _predict_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    config: WalkForwardConfig,
) -> np.ndarray:
    target = train["near_optimal"].astype(bool)
    if target.nunique() == 1:
        return np.full(len(test), float(target.iloc[0]))
    model = _classifier(config)
    model.fit(
        train.loc[:, list(features)],
        target,
        classifier__sample_weight=_month_weights(train),
    )
    probabilities = np.asarray(model.predict_proba(test.loc[:, list(features)]), dtype=float)
    return probabilities[:, 1]


def _predict_remaining_log_ratio(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    config: WalkForwardConfig,
) -> np.ndarray:
    valid = train["remaining_min_log_ratio"].notna()
    regression_train = train.loc[valid]
    if regression_train.empty:
        raise ValueError("沒有可用的剩餘月份低價訓練標籤")
    model = _quantile_regressor(config)
    model.fit(
        regression_train.loc[:, list(features)],
        regression_train["remaining_min_log_ratio"],
        regressor__sample_weight=_month_weights(regression_train),
    )
    predicted = np.asarray(model.predict(test.loc[:, list(features)]), dtype=float)
    return np.clip(predicted, -0.5, 0.5)


def _add_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    result = test.copy()
    result["near_probability"] = _predict_probability(train, test, features, config)
    result["predicted_remaining_log_ratio"] = _predict_remaining_log_ratio(
        train, test, features, config
    )
    result["reservation_adjusted"] = (
        result["previous_adjusted_close"]
        * np.exp(result["predicted_remaining_log_ratio"])
        * (1.0 + config.reservation_buffer)
    )
    if "adjustment_factor" in result.columns:
        result["reservation_raw"] = result["reservation_adjusted"] / result["adjustment_factor"]
    result["training_through"] = str(pd.PeriodIndex(train["month"], freq="M").max())
    return result


def walk_forward_predictions(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """逐月重訓，回傳完全樣本外的每日機率與 reservation price。"""

    settings = config or WalkForwardConfig()
    _validate_input(frame, feature_columns, settings)
    source = frame.copy()
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    source = source.sort_values("date", ignore_index=True)
    outputs: list[pd.DataFrame] = []
    for train_index, test_index in expanding_month_splits(
        source, initial_months=settings.initial_months
    ):
        train = source.loc[train_index]
        test = source.loc[test_index]
        outputs.append(_add_predictions(train, test, feature_columns, settings))
    return pd.concat(outputs, ignore_index=True)
