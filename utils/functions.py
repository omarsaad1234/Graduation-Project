from pathlib import Path
import base64
import time
import streamlit as st
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from models.utils import model_infos, model_urls
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models_2.NaiveBayes import nb_param_selector
from models_2.RandomForet import rf_param_selector
from models_2.DecisionTree import dt_param_selector
from models_2.LogisticRegression import lr_param_selector
from models_2.KNearesNeighbors import knn_param_selector
from models_2.SVC import svc_param_selector
from models_2.GradientBoosting import gb_param_selector
from multiprocessing import Process


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)


def generate_data(dataset, n_samples, train_noise, test_noise, n_classes):
    if dataset == "moons":
        x_train, y_train = make_moons(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_moons(n_samples=n_samples, noise=test_noise)
    elif dataset == "circles":
        x_train, y_train = make_circles(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_circles(n_samples=n_samples, noise=test_noise)
    elif dataset == "blobs":
        x_train, y_train = make_blobs(
            n_features=2,
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=train_noise * 47 + 0.57,
            random_state=42,
        )
        x_test, y_test = make_blobs(
            n_features=2,
            n_samples=n_samples // 2,
            centers=2,
            cluster_std=test_noise * 47 + 0.57,
            random_state=42,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def plot_decision_boundary_and_metrics(
    model, x_train, y_train, x_test, y_test, metrics
):
    d = x_train.shape[1]

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    model_input = [(xx.ravel() ** p, yy.ravel() ** p)
                   for p in range(1, d // 2 + 1)]
    aux = []
    for c in model_input:
        aux.append(c[0])
        aux.append(c[1])

    Z = model.predict(np.concatenate([v.reshape(-1, 1) for v in aux], axis=1))

    Z = Z.reshape(xx.shape)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [
            {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Decision Boundary", None, None),
        row_heights=[0.7, 0.30],
    )

    heatmap = go.Heatmap(
        x=xx[0],
        y=y_,
        z=Z,
        colorscale=["tomato", "rgb(27,158,119)"],
        showscale=False,
    )

    train_data = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        name="train data",
        mode="markers",
        showlegend=True,
        marker=dict(
            size=10,
            color=y_train,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    test_data = go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        name="test data",
        mode="markers",
        showlegend=True,
        marker_symbol="cross",
        visible="legendonly",
        marker=dict(
            size=10,
            color=y_test,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(heatmap, row=1, col=1,).add_trace(train_data).add_trace(
        test_data
    ).update_xaxes(range=[x_min, x_max], title="x1").update_yaxes(
        range=[y_min, y_max], title="x2"
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_accuracy"],
            title={"text": f"Accuracy (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_accuracy"]},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_f1"],
            title={"text": f"F1 score (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_f1"]},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=700,
    )

    return fig


def train_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()

    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)

    return model, train_accuracy, train_f1, test_accuracy, test_f1, duration


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def get_model_tips(model_type):
    model_tips = model_infos[model_type]
    return model_tips


def get_model_url(model_type):
    model_url = model_urls[model_type]
    text = f"**Link to scikit-learn official documentation [here]({model_url}) 💻 **"
    return text


def add_polynomial_features(x_train, x_test, degree):
    for d in range(2, degree + 1):
        x_train = np.concatenate(
            (
                x_train,
                x_train[:, 0].reshape(-1, 1) ** d,
                x_train[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
        x_test = np.concatenate(
            (
                x_test,
                x_test[:, 0].reshape(-1, 1) ** d,
                x_test[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
    return x_train, x_test


def build_model(df, model):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider(
            'Data split ratio (% for Training Set)', 10, 90, 80, 5)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)

    st.markdown('** Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('** Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    model.fit(X_train, Y_train)
    st.write('Model Info')
    st.info(model)

    st.markdown('** Training set**')
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = np.round(accuracy_score(Y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(Y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(Y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(Y_test, y_test_pred, average="weighted"), 3)
    st.write('train_accuracy')
    st.info(train_accuracy)
    st.write('train_f1')
    st.info(train_f1)
    st.write('test_accuracy')
    st.info(test_accuracy)
    st.write('test_f1')
    st.info(test_f1)

    st.write('Model Parameters')
    st.write(model.get_params())


def split_data(df, split_size):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)

    return X_train, X_test, Y_train, Y_test


def encodeing_df(df):
    col_name = []
    label_encoder = LabelEncoder()
    for (colname, colval) in df.items():
        if colval.dtype == 'object' or colval.dtype == 'bool':
            col_name.append(colname)
    for col in col_name:
        df[col] = label_encoder.fit_transform(df[col])
    return df


def replace_null(df):
    col_nan = []
    for (colname, colval) in df.items():
        if df[colname].isnull().values.any() == True:
            col_nan.append(colname)

    for col in col_nan:
        mean_value = df[col].mean()
        df[col].fillna(value=mean_value, inplace=True)
    return df


def scaling(df):
    x = df.iloc[:, :-1]  # Using all column except for the last column as X
    y = df.iloc[:, -1]  # Selecting the last
    m = x.max() - x.min()
    m = m.to_dict()
    l = []
    for key, val in m.items():
        if val == 0:
            l.append(key)
    for i in l:
        x = x.drop([i], axis=1)

    df_norm = (x-x.min())/(x.max()-x.min())
    df_norm = pd.concat((df_norm, y),1)

    return df_norm


def cal_mean(df):
    df_mean = df.mean(axis=0).mean(axis=0)
    return df_mean


def cal_std(df):
    col_std = df.std()
    col = col_std.mean(axis=0)
    return col


def data_shape(df):
    # n_class	n_rows	n_coloumn	data_size
    shape = df.shape
    n_rows = shape[0]
    n_coloumn = shape[1]
    index_last_col = df.columns[-1]
    n_class = len(df[index_last_col].unique())
    return n_rows, n_coloumn, n_class


def pre_proses_data(df):
    df = encodeing_df(df)
    df = replace_null(df)

    # df = scaling(df)
    return df, std, mean


def sim_id(n_rows, n_coloumn, n_class, df_size, std, mean):
    dataf = pd.read_csv('datasets.csv', header=None)
    datasets = dataf.to_numpy()
    max_id = []
    max_score = [0, ]
    n = -1
    max_id2 = []
    for dataset in datasets:
        score = 0
        (id, num_class, num_row, nm_col, dataSiz, data_mean, data_std) = dataset
        nclass = (n_class/num_class)
        nrows = (n_rows/num_row)
        ncol = (n_coloumn/nm_col)
        siz = (df_size/dataSiz)
        meannn = (mean/data_mean)
        stddd = (std/data_std)
        if (nclass > 0.6 and nclass < 1.4):
            score += 1
        if (nrows > 0.6 and nrows < 1.4):
            score += 1
        if (ncol > 0.6 and ncol < 1.4):
            score += 1
        if (siz > 0.6 and siz < 1.4):
            score += 1
        if (meannn > 0.6 and meannn < 1.4):
            score += 1
        if (stddd > 0.6 and stddd < 1.4):
            score += 1
        if max_score[-1] > score:
            max_score = max_score
            max_id = max_id
        else:
            max_score .append(score)
            max_id .append(int(id))
    for i in max_score:
        if i == max(max_score):
            max_id2.append(max_id[n])
        n += 1

    return (max_id2)


def best_data(models, new_data):
    a = []
    for model in models:
        (model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
         max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, duration, data_id, n_class, n_rows, n_coloumn, data_size, mean, std) = model
        if model_name == 'Random Forest ' or model_name == 'Random Forest':
            if min_samples_split == "1":
                min_samples_split = "2"
            model = rf_param_selector(
                criterion, int(n_estimators), int(max_depth), int(min_samples_split), max_features)
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1,
             duration) = train_model(model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)

        elif model_name == 'Decision Tree':
            if max_features != 'sqrt' or max_features != 'log2':
                max_features = None
            if min_samples_split == "1":
                min_samples_split = "2"

            model = dt_param_selector(criterion, int(
                max_depth), int(min_samples_split), max_features)
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)
        elif model_name == 'SVC':
            model = svc_param_selector(float(C), kernel)
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)

        elif model_name == 'K Nearest Neighbors':
            model = knn_param_selector(int(n_neighbors), metric)
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)

        elif model_name == 'Gradient Boosting':
            model = gb_param_selector(
                float(learning_rate), int(n_estimators), int(max_depth))
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)

        elif model_name == 'Logistic Regression':
            model = lr_param_selector(solver, penalty, float(C), int(max_iter))
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)
        elif model_name == 'NaiveBayes':
            model = nb_param_selector()
            (X_train, X_test, Y_train, Y_test) = split_data(new_data, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            b = [model_name, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors, metric, solver, penalty, C,
                 max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, np.round(duration, 3)]
            a.append(b)
    a = sorted(a, key=lambda x: x[16], reverse=True)
    return a


def two_arr(arr):
    a1 = []
    for i in arr:
        for j in arr[arr.index(i)]:
            a1.append(j)
    a1 = sorted(a1, key=lambda x: x[16], reverse=True)
    return a1


def k_nearst(df):
    metrics = ["minkowski", "euclidean", "manhattan", "chebyshev"]
    for niebour in range(5, 21):
        for metric in metrics:
            model = knn_param_selector(niebour, metric)
            (X_train, X_test, Y_train, Y_test) = split_data(df, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            write_file("test2.csv", 'K Nearest Neighbors', 0, 0, 0, 0, 0, 0, niebour, metric, 0, 0, 0, 0, 0, train_accuracy, train_f1,
                       test_accuracy, test_f1, np.round((duration), 5))


def SVC(df):
    kernel = ["rbf",  "poly", "sigmoid", "linear"]
    for C in np.arange(0.1, 2.0, 0.1):
        for metric in kernel:
            model = svc_param_selector(C, metric)
            (X_train, X_test, Y_train, Y_test) = split_data(df, 80)
            (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                model, X_train, Y_train, X_test, Y_test)
            write_file('test2.csv', 'SVC', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.round(C, 5), 0, metric, train_accuracy, train_f1, test_accuracy,
                       test_f1, np.round((duration), 5))


def NaiveBayes(df):
    model = nb_param_selector()
    (X_train, X_test, Y_train, Y_test) = split_data(df, 80)
    (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
        model, X_train, Y_train, X_test, Y_test)
    write_file('test2.csv', 'NaiveBayes', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, train_accuracy, train_f1, test_accuracy,
               test_f1, np.round((duration), 5))


def Decision_Tree(df):
    criterion = ["gini", "entropy"]
    max_features = [None,  "sqrt", "log2"]
    for max_depth in range(1, 51):
        for metric in criterion:
            for min_samples_split in range(2, 21):
                for feature in max_features:

                    model = dt_param_selector(
                        metric, max_depth, min_samples_split, feature)
                    (X_train, X_test, Y_train, Y_test) = split_data(df, 80)
                    (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                        model, X_train, Y_train, X_test, Y_test)
                    if feature == None:
                        write_file('test2.csv', 'Decision Tree', metric, max_depth, min_samples_split, np.nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, train_accuracy,
                                   train_f1, test_accuracy, test_f1, np.round((duration), 5))
                    else:
                        write_file('test2.csv', 'Decision Tree', metric, max_depth, min_samples_split, feature, 0, 0, 0, 0, 0, 0, 0, 0, 0, train_accuracy,
                                   train_f1, test_accuracy, test_f1, np.round((duration), 5))


def RandomForest(df):
    criterion = ["gini", "entropy"]
    max_features = ["sqrt", "log2"]
    for metric in criterion:
        for n_estimators in range(50, 310, 20):
            for max_depth in range(1, 26, 1):
                for min_samples_split in range(2, 21, 2):
                    for feature in max_features:
                        model = rf_param_selector(
                            metric, n_estimators, max_depth, min_samples_split, feature)
                        (X_train, X_test, Y_train, Y_test) = split_data(df, 80)
                        (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                            model, X_train, Y_train, X_test, Y_test)
                        write_file('test2.csv', 'Random Forest ', metric, max_depth, min_samples_split, feature, 0, n_estimators, 0, 0, 0, 0, 0, 0, 0, train_accuracy,
                                   train_f1, test_accuracy, test_f1, np.round((duration), 5))


def Gradient_Boosting(df):
    for learning_rate in np.arange(0.1, 0.4, 0.1):
        for n_estimators in range(10, 510, 20):
            for max_depth in range(3, 31, 2):
                model = gb_param_selector(
                    learning_rate, n_estimators, max_depth)
                (X_train, X_test, Y_train, Y_test) = split_data(df, 80)
                (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                    model, X_train, Y_train, X_test, Y_test)
                write_file('test2.csv', 'Gradient Boosting', 0, max_depth, 0, 0, np.round(learning_rate, 5), n_estimators, 0, 0, 0, 0, 0, 0, 0,  train_accuracy,
                           train_f1, test_accuracy, test_f1, np.round((duration), 5))


def LogisticRegression(df):
    solvers = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
    for solver in solvers:
        if solver in ["newton-cg", "lbfgs", "sag"]:
            penalties = ["l2"]
        elif solver == "saga":
            penalties = ["l1", "l2"]
        elif solver == "liblinear":
            penalties = ["l1"]
        for p in penalties:
            for C in np.arange(0.1, 2.0, 0.1):
                for max_iter in range(100, 2050, 50):
                    model = lr_param_selector(solver, p, C, max_iter)
                    (X_train, X_test, Y_train, Y_test) = split_data(df, 80)
                    (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
                        model, X_train, Y_train, X_test, Y_test)
                    write_file('test2.csv', 'Logistic Regression', 0, 0, 0, 0, 0, 0, 0, 0, solver, p, np.round(C, 5), max_iter, 0, train_accuracy,
                               train_f1, test_accuracy, test_f1, np.round((duration), 5))


def write_file(path, *args):
    with open(path, "a") as file:
        writer = csv.writer(file)
        writer.writerow([*args])
    file.close()


def convert_df(df):
    return df.to_csv().encode('utf-8')


def display_best_model(df):
    temp = {}
    names = ['model', 'criterion', 'max_depth', 'min_samples_split', 'max_features', 'learning_rate', 'n_estimators',
             'n_neighbors', 'metric', 'solver', 'penalty', ' C', 'max_iter', 'kernel']
    best_one = df.iloc[0]
    (model, criterion, max_depth, min_samples_split, max_features, learning_rate, n_estimators, n_neighbors,
     metric, solver, penalty, C, max_iter, kernel, train_accuracy, train_f1, test_accuracy, test_f1, duration) = best_one
    data = dict(zip(names, best_one))
    for x, y in data.items():
        if y != '0' and y != '0.0' and y != 0:
            temp[x] = y
    new = pd.DataFrame(temp, index=[0])
    st.table(new)
    boot_body(model, np.round(train_accuracy*100, 2), np.round(test_accuracy*100, 2),
              np.round(train_f1*100, 2), np.round(test_f1*100, 2), duration)
    return model


def boot_body(model_type, train_accuracy, test_accuracy, train_f1, test_f1, duration):
    col1, col2, col3 = st.columns((1, 1, 2))
    with col1:
        plot_placeholder = st.empty()
        st.write('train_accuracy')
        train_placeholder = st.empty()
        st.write('test_accuracy')
        test_placeholder = st.empty()

    with col2:
        st.write('train_f1')
        train_f1_placeholder = st.empty()
        st.write('test_f1')
        test_f1_placeholder = st.empty()
    with col3:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    if model_type == 'Random Forest ':
        model_type = 'Random Forest'
    model_url = get_model_url(model_type)
    train_placeholder.info(train_accuracy)
    test_placeholder.info(test_accuracy)
    train_f1_placeholder.info(train_f1)
    test_f1_placeholder.info(test_f1)
    model_tips = get_model_tips(model_type)

    duration_placeholder.warning(
        f"Training took {float (duration):.3f} seconds")
    model_url_placeholder.markdown(model_url)
    # snippet_placeholder.code(snippet)
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)


def run_all_model(df):
    p1 = Process(target=NaiveBayes(df))
    p2 = Process(target=k_nearst(df))
    p3 = Process(target=SVC(df))
    p4 = Process(target=Decision_Tree(df))
    p5 = Process(target=RandomForest(df))
    p6 = Process(target=Gradient_Boosting(df))
    p7 = Process(target=LogisticRegression(df))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p7.start()
    p7.join()


def empty_datafreame(data_name):
    df1 = pd.read_csv(data_name)
    df1.drop(df1.index, inplace=True)
    df1.to_csv(data_name, index=False)


def pre_Selector(df):
    Pre_Selector = st.sidebar.radio(
        "Do you want to Preprocess Your Dataset?", ['Yes', 'No'], index=1)
    encodeing = 'NO'
    replaceNull = 'NO'
    scale = 'NO'
    if Pre_Selector == "Yes":
        encodeing = st.sidebar.radio(
            "Do you want to Encodeing Your Dataset?", ['Yes', 'No'])
        replaceNull = st.sidebar.radio(
            "Do you want to replace Null for Your Dataset?", ['Yes', 'No'])
        scale = st.sidebar.radio("Do you want to scaling Your Dataset?", [
                                 'Yes', 'No'], index=1)
        if encodeing == "Yes":
            df = encodeing_df(df)
        if replaceNull == "Yes":
            df = replace_null(df)
        if scale == "Yes":
            df = scaling(df)
        st.write("Data Set After preprossing ")
        st.write(df.head())
    std = cal_std(df)
    std = np.round(std, 5)
    mean = cal_mean(df)
    mean = np.round(mean, 5)
    return df, std, mean, Pre_Selector, encodeing, replaceNull, scale

def footer():
    ft = """
    <style>
    a:link , a:visited{
    color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
    background-color: transparent;
    text-decoration: none;
    }

    a:hover,  a:active {
    color: #0283C3; /* theme's primary color*/
    background-color: transparent;
    text-decoration: underline;
    }

    #page-container {
    position: relative;
    min-height: 10vh;
    }

    footer{
        visibility:hidden;
    }

    .footer {
    position: relative;
    left: 0;
    top:230px;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #808080; /* theme's text color hex code at 50 percent brightness*/
    text-align: center; /* you can replace 'left' with 'center' or 'right' if you want*/
    }
    </style>

    <div id="page-container">

    <div class="footer">
    <span style='font-size: 0.875em;'>Made with <a style='display: inline; text-align: left;' href="https://streamlit.io/" target="_blank">Streamlit</a><br 'style= top:3px;'>by
     <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="mailto:mohamed01017715716@email.com" target="_blank">  Mohamed Mokhtar</a></span>
    <span style='font-size: 0.875em;'>
     <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="mailto:Alaay9003@gmail.com" target="_blank">  Youssef Alaa</a></span>
    <span style='font-size: 0.875em;'>
     <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="mailto:amr.atallahh147@gmail.com" target="_blank">  Amr Atallah</a></span>
    <span style='font-size: 0.875em;'>
     <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="mailto:" target="_blank">  Mohamed Khaled</a></span>
    <span style='font-size: 0.875em;'>
     <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="mailto:Eidk61857@gmail.com" target="_blank">  Eid Khaled</a></span>
    <span style='font-size: 0.875em;'>
     <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="mailto:omarsaad698@gmail.com" target="_blank"> Omar Saad</a></span>
  
    </div>

    </div>
    """
    st.write(ft, unsafe_allow_html=True)