from spatialmt.config.paths import Dirs

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from tqdm import tqdm


class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, n_estimators):
        self.pbar = tqdm(total=n_estimators, desc="Training", unit="tree")

    def after_iteration(self, model, epoch, evals_log):
        train_loss = list(evals_log.get("train", {}).values())
        val_loss   = list(evals_log.get("eval",  {}).values())
        postfix = {}
        if train_loss:
            postfix["train_loss"] = f"{train_loss[0][-1]:.4f}"
        if val_loss:
            postfix["val_loss"] = f"{val_loss[0][-1]:.4f}"
        self.pbar.set_postfix(postfix)
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

if __name__ == "__main__":
    data_dir = Dirs.model_data_ml

    cell_labels = pd.read_csv(data_dir / "cell_labels.csv").squeeze()
    gene_labels = pd.read_csv(data_dir / "gene_labels.csv").squeeze()
    y_raw = pd.read_csv(data_dir / "cell_type_labels.csv").squeeze()

    X = pd.DataFrame(
        np.load(data_dir / "expression_matrix.npy"),
        index=cell_labels,
        columns=gene_labels,
    )

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        callbacks=[TqdmCallback(n_estimators=300)],
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    recall   = recall_score(y_test, y_pred, average="weighted")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1       : {f1:.4f}  (weighted)")
    print(f"Recall   : {recall:.4f}  (weighted)")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    proba = clf.predict_proba(X_test)
    proba_df = pd.DataFrame(proba, index=X_test.index, columns=le.classes_)
    proba_df.to_csv(Dirs.results / "xgb_classifier_proba.csv")
    print(f"Probability distributions saved to {Dirs.results / 'xgb_classifier_proba.csv'}")

    out_dir = Dirs.trained_models
    clf.save_model(out_dir / "xgb_classifier.ubj")
    joblib.dump(le, out_dir / "xgb_label_encoder.joblib")
    print(f"Model saved to {out_dir / 'xgb_classifier.ubj'}")
    print(f"Label encoder saved to {out_dir / 'xgb_label_encoder.joblib'}")
