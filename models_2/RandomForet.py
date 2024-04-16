from sklearn.ensemble import RandomForestClassifier
def rf_param_selector(criterion,n_estimators,max_depth,min_samples_split,max_features):
    params = {
        "criterion": criterion,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    return model
