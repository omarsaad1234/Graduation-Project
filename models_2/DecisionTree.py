from sklearn.tree import DecisionTreeClassifier
def dt_param_selector(criterion,max_depth,min_samples_split,max_features):
    params = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }
    model = DecisionTreeClassifier(**params)
    return model
