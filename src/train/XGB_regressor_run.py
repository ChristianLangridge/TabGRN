from spatialmt.config.paths import Dirs

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
from tqdm import tqdm

# callback to track progress
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
    
# grabbing data inputs 
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
    
    # test train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    
    # initializing XGBR out-the-box
    clf = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        callbacks=[TqdmCallback(n_estimators=300)],
    )
    
    # training model
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    
    # running on held-out data
    y_pred = clf.predict(X_test)
    
    # metrics of interest
    r2 = r2_score(y_test, y_pred)
    n, p = X_test.shape
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)

    print(f"R2     : {r2:.4f}")
    print(f"R2 adj : {r2_adj:.4f}")
    print(f"MAE    : {MAE:.4f}")
    print(f"RMSE   : {RMSE:.4f}")
    print()

    # saving all outputs 
    pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}, index=X_test.index)
    pred_df.to_csv(Dirs.xgb_regressor / "xgb_regressor_predictions.csv")
    print(f"Predictions saved to {Dirs.xgb_regressor / 'xgb_regressor_predictions.csv'}")

    clf.save_model(Dirs.xgb_regressor / "xgb_regressor.ubj")
    print(f"Model saved to {Dirs.xgb_regressor / 'xgb_regressor.ubj'}")