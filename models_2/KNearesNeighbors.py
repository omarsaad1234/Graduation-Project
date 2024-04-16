from sklearn.neighbors import KNeighborsClassifier
def knn_param_selector(n_neighbors,metric):
    params = {"n_neighbors": n_neighbors, "metric": metric}
    model = KNeighborsClassifier(**params)
    return model
