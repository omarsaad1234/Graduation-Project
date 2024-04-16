import numpy as np
from sklearn.linear_model import LogisticRegression
def lr_param_selector(solver,penalty,C,max_iter):
    C = np.round(C, 3)
    params = {"solver": solver, "penalty": penalty, "C": C, "max_iter": max_iter}
    model = LogisticRegression(**params)
    return model
