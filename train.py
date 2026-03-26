"""
Training script for Pakistan AQI category prediction.
Trains Random Forest, XGBoost, LSTM, Temporal CNN; saves best model by F1.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.utils import to_categorical
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

from preprocessing import run_pipeline, AQI_CATEGORIES, DATA_DIR

SEQ_LEN = 7  # days for LSTM / Temporal CNN


def train_and_evaluate(
    data_dir=None,
    test_size=0.2,
    random_state=42,
    save_dir=".",
):
    data_dir = data_dir or str(DATA_DIR)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    print("Running preprocessing pipeline...")
    X, y, meta, feature_cols, daily = run_pipeline(
        data_dir=data_dir,
        save_daily_path=save_dir / "daily_aggregated.csv",
    )
    joblib.dump(feature_cols, save_dir / "feature_columns.joblib")

    # Scale features (tree models don't require it but helps consistency for future LSTM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, save_dir / "scaler.joblib")

    # Split by time: last N days as test to simulate forecast
    n_test = max(int(len(X) * test_size), 1)
    X_train, X_test = X_scaled[:-n_test], X_scaled[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    # Ensure classes are 1..5 for sklearn
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    classes = sorted(y_train.unique())

    results = {}

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
    results["Random Forest"] = {"accuracy": acc_rf, "f1": f1_rf, "model": rf}
    print(f"Accuracy: {acc_rf:.4f}, F1 (weighted): {f1_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=[AQI_CATEGORIES.get(c, str(c)) for c in classes]))

    # XGBoost if available
    if HAS_XGB:
        print("\n--- XGBoost ---")
        # Map labels to 0..4 for XGBoost
        label_map = {c: i for i, c in enumerate(classes)}
        y_train_xgb = np.array([label_map[int(v)] for v in y_train])
        y_test_xgb = np.array([label_map[int(v)] for v in y_test])
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        xgb_model.fit(X_train, y_train_xgb)
        y_pred_xgb = xgb_model.predict(X_test)
        inv_map = {i: c for c, i in label_map.items()}
        y_pred_xgb_labels = np.array([inv_map[int(p)] for p in y_pred_xgb])
        acc_xgb = accuracy_score(y_test, y_pred_xgb_labels)
        f1_xgb = f1_score(y_test, y_pred_xgb_labels, average="weighted")
        results["XGBoost"] = {
            "accuracy": acc_xgb,
            "f1": f1_xgb,
            "model": xgb_model,
            "label_map": label_map,
            "inv_map": inv_map,
        }
        print(f"Accuracy: {acc_xgb:.4f}, F1 (weighted): {f1_xgb:.4f}")
        print(classification_report(y_test, y_pred_xgb_labels, target_names=[AQI_CATEGORIES.get(c, str(c)) for c in classes]))

    # Build 7-day sequences per city for LSTM / Temporal CNN
    if HAS_KERAS:
        X_seq_list, y_seq_list = [], []
        for city in meta["city"].unique():
            idx = meta["city"] == city
            X_c = X_scaled[idx]
            y_c = y.iloc[np.where(idx)[0]]
            order = np.argsort(meta.loc[idx, "date"].values)
            X_c = X_c[order]
            y_c = y_c.iloc[order].values.astype(int)
            for i in range(SEQ_LEN, len(X_c)):
                X_seq_list.append(X_c[i - SEQ_LEN : i])
                y_seq_list.append(y_c[i])
        if X_seq_list:
            X_seq = np.stack(X_seq_list)
            y_seq = np.array(y_seq_list)
            label_map_nn = {c: i for i, c in enumerate(classes)}
            y_seq_01 = np.array([label_map_nn[v] for v in y_seq])
            y_cat = to_categorical(y_seq_01, num_classes=len(classes))
            n_seq = len(X_seq)
            n_test_seq = max(int(n_seq * test_size), 1)
            X_train_seq, X_test_seq = X_seq[:-n_test_seq], X_seq[-n_test_seq:]
            y_train_seq, y_test_seq = y_cat[:-n_test_seq], y_seq[-n_test_seq:]

            # LSTM
            print("\n--- LSTM ---")
            lstm = Sequential([
                LSTM(64, input_shape=(SEQ_LEN, X_seq.shape[2]), return_sequences=False),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(len(classes), activation="softmax"),
            ])
            lstm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            lstm.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32, verbose=0)
            y_pred_lstm = np.argmax(lstm.predict(X_test_seq, verbose=0), axis=1)
            y_pred_lstm_labels = np.array([classes[i] for i in y_pred_lstm])
            acc_lstm = accuracy_score(y_test_seq, y_pred_lstm_labels)
            f1_lstm = f1_score(y_test_seq, y_pred_lstm_labels, average="weighted")
            results["LSTM"] = {"accuracy": acc_lstm, "f1": f1_lstm, "model": lstm, "is_seq": True}
            print(f"Accuracy: {acc_lstm:.4f}, F1 (weighted): {f1_lstm:.4f}")

            # Temporal CNN (1D CNN)
            print("\n--- Temporal CNN (1D CNN) ---")
            tcn = Sequential([
                Conv1D(32, 3, activation="relu", input_shape=(SEQ_LEN, X_seq.shape[2])),
                MaxPooling1D(2),
                Conv1D(64, 2, activation="relu"),
                Flatten(),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(len(classes), activation="softmax"),
            ])
            tcn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            tcn.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32, verbose=0)
            y_pred_tcn = np.argmax(tcn.predict(X_test_seq, verbose=0), axis=1)
            y_pred_tcn_labels = np.array([classes[i] for i in y_pred_tcn])
            acc_tcn = accuracy_score(y_test_seq, y_pred_tcn_labels)
            f1_tcn = f1_score(y_test_seq, y_pred_tcn_labels, average="weighted")
            results["Temporal CNN"] = {"accuracy": acc_tcn, "f1": f1_tcn, "model": tcn, "is_seq": True}
            print(f"Accuracy: {acc_tcn:.4f}, F1 (weighted): {f1_tcn:.4f}")

    # Pick best by F1 and save as main model
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    model_obj = best["model"]
    is_seq = best.get("is_seq", False)

    if is_seq and HAS_KERAS:
        keras_path = save_dir / "model.keras"
        model_obj.save(keras_path)
        bundle = {
            "model": None,
            "model_path": str(keras_path),
            "model_name": best_name,
            "is_xgb": False,
            "is_seq": True,
            "label_map": None,
            "inv_map": None,
            "classes": classes,
        }
    else:
        bundle = {
            "model": model_obj,
            "model_path": None,
            "model_name": best_name,
            "is_xgb": best_name == "XGBoost",
            "is_seq": False,
            "label_map": best.get("label_map"),
            "inv_map": best.get("inv_map"),
            "classes": classes,
        }
    joblib.dump(bundle, save_dir / "model.joblib")
    print(f"\nSaved best model: {best_name} -> {save_dir / 'model.joblib'}")

    # Save evaluation summary (all models)
    eval_df = pd.DataFrame([
        {"model": k, "accuracy": v["accuracy"], "f1_weighted": v["f1"]}
        for k, v in results.items()
    ])
    eval_df.to_csv(save_dir / "evaluation.csv", index=False)
    print("\nAll models comparison:")
    print(eval_df.to_string(index=False))

    return results, X, y, meta, feature_cols, daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Path to folder with city CSVs")
    parser.add_argument("--save-dir", default=".", help="Directory to save model and artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_evaluate(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.seed,
        save_dir=args.save_dir,
    )
