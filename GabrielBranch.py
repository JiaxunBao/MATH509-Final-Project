#importation of required libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            features,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            ["cma"],
        ),
    ]
)

# 6. Candidate models
models = {
    "ridge": Pipeline([
        ("pre", preprocessor),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ]),
    "elastic_net": Pipeline([
        ("pre", preprocessor),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            alphas=np.logspace(-3, 1, 30),
            cv=5,
            max_iter=20000,
            random_state=123,
        )),
    ]),
    "random_forest": Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=700,
            max_depth=8,
            min_samples_leaf=2,
            random_state=123,
            n_jobs=-1,
        )),
    ]),
    "gradient_boosting": Pipeline([
        ("pre", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            random_state=123,
        )),
    ]),
}
