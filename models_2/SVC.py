from sklearn.svm import SVC
def svc_param_selector(C,kernel):
    params = {"C": C, "kernel": kernel}
    model = SVC(**params)
    return model
