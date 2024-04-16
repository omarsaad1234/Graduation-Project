from sklearn.ensemble import GradientBoostingClassifier
def gb_param_selector(learning_rate,n_estimators,max_depth):
    params = {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    model = GradientBoostingClassifier(**params)
    return model
