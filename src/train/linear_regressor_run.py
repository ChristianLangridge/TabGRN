from spatialmt.config.paths import Dirs

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

if __name__ == "__main__":
    data_dir = Dirs.model_data_ml

    cell_labels = pd.read_csv(data_dir / "cell_labels.csv").squeeze()
    gene_labels = pd.read_csv(data_dir / "gene_labels.csv").squeeze()
    y = pd.read_csv(data_dir / "pseudotime_labels.csv").squeeze()

    X = pd.DataFrame(
        np.load(data_dir / "expression_matrix.npz")["X"],
        index=cell_labels,
        columns=gene_labels,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    n, p = X_test.shape
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    MAE  = mean_absolute_error(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)

    print(f"R2     : {r2:.4f}")
    print(f"R2 adj : {r2_adj:.4f}")
    print(f"MAE    : {MAE:.4f}")
    print(f"RMSE   : {RMSE:.4f}")

    pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}, index=X_test.index)
    pred_df.to_csv(Dirs.linear_regressor / "linear_regressor_predictions.csv")
    print(f"Predictions saved to {Dirs.linear_regressor / 'linear_regressor_predictions.csv'}")

    joblib.dump(clf, Dirs.linear_regressor / "linear_regressor.joblib")
    print(f"Model saved to {Dirs.linear_regressor / 'linear_regressor.joblib'}")
