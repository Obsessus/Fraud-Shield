"""Tests for feature engineering, focused on leakage-safety and correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudintel.features.build import (
    add_frequency_features,
    add_identity_features,
    add_temporal_features,
    add_test_entity_features,
    add_train_entity_features,
    build_temporal_splits,
    build_test_features,
    build_train_features,
)


def _train_card1():
    # card1 'A' appears at times 1,2,3 with fraud 0,1,0 -> prior-only rates: t1=NaN, t2=0.0, t3=0.5
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [1, 2, 3, 10, 20],
            "TransactionAmt": [10.0, 20.0, 30.0, 5.0, 7.0],
            "isFraud": [0, 1, 0, 0, 0],
            "card1": ["A", "A", "A", "B", "B"],
            "addr1": [1, 1, 1, 2, 2],
            "P_emaildomain": ["a", "a", "a", "b", "b"],
            "DeviceInfo": ["x", "x", "x", "y", "y"],
        }
    )


def test_time_safe_excludes_current_row():
    df, _ = add_train_entity_features(_train_card1(), keys=["card1"])
    a = df[df["card1"] == "A"].sort_values("TransactionDT")
    rates = a["card1_hist_fraud_rate"].tolist()
    # t1: no prior -> NaN ; t2: prior=[0] -> 0.0 ; t3: prior=[0,1] -> 0.5
    assert np.isnan(rates[0])
    assert rates[1] == 0.0
    assert rates[2] == 0.5
    # if the current row were included, t2 would be 0.5 and t3 would be 0.333
    assert rates[2] != 1 / 3


def test_test_uses_train_global_mapping_only():
    train, artifacts = add_train_entity_features(_train_card1(), keys=["card1"])
    # global fraud rate for card1 'A' over all train rows = 1/3
    expected = 1 / 3
    test = pd.DataFrame(
        {"TransactionID": [9], "TransactionDT": [100], "TransactionAmt": [1.0], "card1": ["A"]}
    )
    out = add_test_entity_features(test, artifacts, keys=["card1"])
    assert out["card1_hist_fraud_rate"].iloc[0] == pytest.approx(expected)
    # test must not see its own (absent) label; value comes purely from train history
    assert out["card1_hist_count"].iloc[0] == 3


def test_frequency_encoding_consistent_across_splits():
    train = pd.DataFrame(
        {"card1": ["X", "X", "Y"], "TransactionDT": [1, 2, 3], "isFraud": [0, 0, 1],
         "addr1": [1, 1, 2], "P_emaildomain": ["a", "a", "b"], "DeviceInfo": ["x", "x", "y"]}
    )
    df, maps = add_frequency_features(train, ["card1"])
    assert df.loc[df["card1"] == "X", "card1_freq"].iloc[0] == 2
    test = pd.DataFrame(
        {"card1": ["X", "Z"], "TransactionDT": [4, 5], "isFraud": [0, 0],
         "addr1": [1, 3], "P_emaildomain": ["a", "c"], "DeviceInfo": ["x", "z"]}
    )
    te, _ = add_frequency_features(test, ["card1"], mappings=maps)
    assert te.loc[te["card1"] == "X", "card1_freq"].iloc[0] == 2  # mapped from train
    assert te.loc[te["card1"] == "Z", "card1_freq"].iloc[0] == 0  # unseen -> 0


def test_temporal_features_derive_correctly():
    df = pd.DataFrame({"TransactionDT": [0, 86400]})  # 2017-11-30 00:00, 2017-12-01 00:00
    out = add_temporal_features(df)
    assert out.loc[0, "hour"] == 0 and out.loc[0, "DT_M"] == 11
    assert out.loc[1, "hour"] == 0 and out.loc[1, "DT_M"] == 12


def test_identity_features_parse():
    df = pd.DataFrame(
        {
            "id_30": ["iOS 11.1", "Windows 10"],
            "id_31": ["Chrome Mobile 71.0", "Samsung Browser 6.4"],
            "id_12": [1, None],
        }
    )
    out = add_identity_features(df)
    assert out["os_family"].tolist() == ["iOS", "Windows"]
    assert out["browser_family"].tolist() == ["Chrome", "Samsung"]
    assert out["n_identity_present"].tolist() == [3, 2]


def test_build_train_and_test_shapes():
    train = _train_card1()
    test = pd.DataFrame(
        {
            "TransactionID": [9], "TransactionDT": [100], "TransactionAmt": [1.0], "card1": ["A"],
            "addr1": [1], "P_emaildomain": ["a"], "DeviceInfo": ["x"],
        }
    )
    tr, art = build_train_features(train)
    te = build_test_features(test, art)
    assert "card1_hist_fraud_rate" in tr.columns
    assert "card1_hist_fraud_rate" in te.columns
    assert te["card1_hist_fraud_rate"].iloc[0] == pytest.approx(1 / 3)


def test_build_temporal_splits_leakage_safe():
    # card 'A' fraud: train [0,1] (mean 0.5), val [1], holdout [0]
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [1, 2, 10, 20],
            "TransactionAmt": [1.0, 2.0, 3.0, 4.0],
            "isFraud": [0, 1, 1, 0],
            "card1": ["A", "A", "A", "A"],
            "addr1": [1, 1, 1, 1],
            "P_emaildomain": ["a", "a", "a", "a"],
            "DeviceInfo": ["x", "x", "x", "x"],
        }
    )
    tr, va, ho = build_temporal_splits(df, t1=5, t2=15)
    # validation maps TRAIN-ONLY global mean (0.5), not train+val (2/3) -> no val leakage
    assert va["card1_hist_fraud_rate"].iloc[0] == pytest.approx(0.5)
    # hold-out maps TRAIN+VAL global mean (2/3), not train+val+holdout (0.5) -> no holdout leakage
    assert ho["card1_hist_fraud_rate"].iloc[0] == pytest.approx(2 / 3)
    assert "isFraud" in tr.columns and "isFraud" in va.columns and "isFraud" in ho.columns
