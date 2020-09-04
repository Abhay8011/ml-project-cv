from sklearn import ensemble
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

MODELS = {
    'randomforest':ensemble.RandomForestClassifier(n_jobs = -1, verbose=2),
    'xgboost':XGBClassifier(),
    'catboost':CatBoostClassifier(iterations=150, learning_rate=0.01)
}