from spatialmt.config.paths import Dirs

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestCentroid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

if __name__ == "__main__":
    data_dir = Dirs.model_data_ml

    cell_labels = pd.read_csv(data_dir / "cell_labels.csv").squeeze()
    gene_labels = pd.read_csv(data_dir / "gene_labels.csv").squeeze()
    y = pd.read_csv(data_dir / "cell_type_labels.csv").squeeze()

    X = pd.DataFrame(
        np.load(data_dir / "expression_matrix.npy"),
        index=cell_labels,
        columns=gene_labels,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = CalibratedClassifierCV(NearestCentroid(), cv=5, method="sigmoid")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    recall   = recall_score(y_test, y_pred, average="weighted")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1       : {f1:.4f}  (weighted)")
    print(f"Recall   : {recall:.4f}  (weighted)")
    print()
    print(classification_report(y_test, y_pred))

    proba = clf.predict_proba(X_test)
    proba_df = pd.DataFrame(proba, index=X_test.index, columns=clf.classes_)
    proba_df.to_csv(Dirs.results / "nearest_centroid_proba.csv")
    print(f"Probability distributions saved to {Dirs.results / 'nearest_centroid_proba.csv'}")

    joblib.dump(clf, Dirs.trained_models / "nearest_centroid.joblib")
    print(f"Model saved to {Dirs.trained_models / 'nearest_centroid.joblib'}")
