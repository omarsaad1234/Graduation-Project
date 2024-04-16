import numpy as np
import streamlit as st


from models.NaiveBayes import nb_param_selector
from models.NeuralNetwork import nn_param_selector
from models.RandomForet import rf_param_selector
from models.DecisionTree import dt_param_selector
from models.LogisticRegression import lr_param_selector
from models.KNearesNeighbors import knn_param_selector
from models.SVC import svc_param_selector
from models.GradientBoosting import gb_param_selector

from models.utils import model_imports
from utils.functions import img_to_bytes


def introduction():
    st.title("**Welcome to playground 🧪**")
    st.subheader(
        """
        This is a place where you can get familiar with machine learning models directly from your browser
        """
    )

    st.markdown(
        """
    - 🗂️ Choose a dataset
    - ⚙️ Pick a model and set its hyper-parameters
    - 📉 Train it and check its performance metrics and decision boundary on train and test data
    - 🩺 Diagnose possible overitting and experiment with other settings
    -----
    """
    )


def dataset_selector():
    dataset_container = st.sidebar.expander("Configure a dataset", True)
    with dataset_container:
        dataset = st.selectbox(
            "Choose a dataset", ("moons", "circles", "blobs"))
        n_samples = st.number_input(
            "Number of samples",
            min_value=50,
            max_value=1000,
            step=10,
            value=300,
        )

        train_noise = st.slider(
            "Set the noise (train data)",
            min_value=0.01,
            max_value=0.2,
            step=0.005,
            value=0.06,
        )
        test_noise = st.slider(
            "Set the noise (test data)",
            min_value=0.01,
            max_value=1.0,
            step=0.005,
            value=train_noise,
        )

        if dataset == "blobs":
            n_classes = st.number_input("centers", 2, 5, 2, 1)
        else:
            n_classes = None

    return dataset, n_samples, train_noise, test_noise, n_classes


def model_selector():
    model_training_container = st.sidebar.expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                # "Neural Network",
                "K Nearest Neighbors",
                "Gaussian Naive Bayes",
                "SVC",
            ),
        )

        if model_type == "Logistic Regression":
            model = lr_param_selector()

        elif model_type == "Decision Tree":
            model = dt_param_selector()

        elif model_type == "Random Forest":
            model = rf_param_selector()

        elif model_type == "Neural Network":
            model = nn_param_selector()

        elif model_type == "K Nearest Neighbors":
            model = knn_param_selector()

        elif model_type == "Gaussian Naive Bayes":
            model = nb_param_selector()

        elif model_type == "SVC":
            model = svc_param_selector()

        elif model_type == "Gradient Boosting":
            model = gb_param_selector()

    return model_type, model


def generate_snippet(
    model, model_type, n_samples, train_noise, test_noise, dataset, degree
):
    train_noise = np.round(train_noise, 3)
    test_noise = np.round(test_noise, 3)

    model_text_rep = repr(model)
    model_import = model_imports[model_type]

    if degree > 1:
        feature_engineering = f"""
    >>> for d in range(2, {degree+1}):
    >>>     x_train = np.concatenate((x_train, x_train[:, 0] ** d, x_train[:, 1] ** d))
    >>>     x_test= np.concatenate((x_test, x_test[:, 0] ** d, x_test[:, 1] ** d))
    """

    if dataset == "moons":
        dataset_import = "from sklearn.datasets import make_moons"
        train_data_def = (
            f"x_train, y_train = make_moons(n_samples={n_samples}, noise={train_noise})"
        )
        test_data_def = f"x_test, y_test = make_moons(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "circles":
        dataset_import = "from sklearn.datasets import make_circles"
        train_data_def = f"x_train, y_train = make_circles(n_samples={n_samples}, noise={train_noise})"
        test_data_def = f"x_test, y_test = make_circles(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "blobs":
        dataset_import = "from sklearn.datasets import make_blobs"
        train_data_def = f"x_train, y_train = make_blobs(n_samples={n_samples}, clusters=2, noise={train_noise* 47 + 0.57})"
        test_data_def = f"x_test, y_test = make_blobs(n_samples={n_samples // 2}, clusters=2, noise={test_noise* 47 + 0.57})"

    snippet = f"""
    >>> {dataset_import}
    >>> {model_import}
    >>> from sklearn.metrics import accuracy_score, f1_score

    >>> {train_data_def}
    >>> {test_data_def}
    {feature_engineering if degree > 1 else ''}    
    >>> model = {model_text_rep}
    >>> model.fit(x_train, y_train)
    
    >>> y_train_pred = model.predict(x_train)
    >>> y_test_pred = model.predict(x_test)
    >>> train_accuracy = accuracy_score(y_train, y_train_pred)
    >>> test_accuracy = accuracy_score(y_test, y_test_pred)
    """
    return snippet


def generate_data_snippet(
        model, model_type,df, Pre_Selector,encodeing,replaceNull,scale
):
    encod=''
    replaceNan=''
    scaleing=''
    if Pre_Selector=="Yes":
        
        if encodeing=="Yes":
            
            encod=f""">>>def encodeing_df(df):
        >>>col_name = []
        >>>label_encoder = LabelEncoder()
        >>>for (colname, colval) in df.items():
        >>>    if colval.dtype == 'object':
        >>>        col_name.append(colname)
        >>>for col in col_name:
        >>>    df[col] = label_encoder.fit_transform(df[col])
        >>>return df
    >>>df = encodeing_df(df)
        """
        if replaceNull=="Yes":
            replaceNan=f""">>>def replace_null(df):
        >>>col_nan = []
        >>>for (colname, colval) in df.items():
        >>>    if df[colname].isnull().values.any() == True:
        >>>        col_nan.append(colname)
        >>>for col in col_nan:
        >>>    mean_value = df[col].mean()
        >>>    df[col].fillna(value=mean_value, inplace=True)
        >>>return df
    >>>df=replace_null(df)
        """
        if scale=='Yes':
            scaleing=f""">>>def scaling(df):
        >>>x = df.iloc[:, :-1]  # Using all column except for the last column as X
        >>>y = df.iloc[:, -1]  # Selecting the last
        >>>m = x.max() - x.min()
        >>>m = m.to_dict()
        >>>l = []
        >>>for key, val in m.items():
        >>>    if val == 0:
        >>>        l.append(key)
        >>>for i in l:
        >>>    x = x.drop([i], axis=1)
        >>>df_norm = (x-x.min())/(x.max()-x.min())
        >>>df_norm = pd.concat((df_norm, y), 1)
        >>>return df_norm
    >>>df=scaling(df)
            """

    model_text_rep = repr(model)
    if model_type =='Random Forest ':
        model_import = model_imports["Random Forest"]
    else:   
        model_import = model_imports[model_type]
        

    snippet = f"""
    >>> {model_import}
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from sklearn.tree import {model_type}
    >>> from sklearn.metrics import accuracy_score, f1_score
    >>> df = pd.read_csv(Data Frame Path)
    { encod}
    { replaceNan}
    { scaleing}
    >>>>>> X = df.iloc[:, :-1]
    >>>>>> Y = df.iloc[:, -1]
    >>>>>> X_train, X_test, Y_train, Y_test = train_test_split(
    >>>    X, Y, test_size=(100-split_size)/100)
    >>> model = {model_text_rep}
    >>> model.fit(x_train, y_train)
    >>> y_train_pred = model.predict(x_train)
    >>> y_test_pred = model.predict(x_test)
    >>> train_accuracy = accuracy_score(y_train, y_train_pred)
    >>> test_accuracy = accuracy_score(y_test, y_test_pred)
    """
    return snippet


def polynomial_degree_selector():
    return st.sidebar.number_input("Highest polynomial degree", 1, 10, 1, 1)


def upload_data():
    with st.sidebar.header('1. Upload your CSV data'):
        st.sidebar.write('Only classification Dataset Allowed')
        uploaded_file = st.sidebar.file_uploader(
            "Upload your input CSV file", type=["csv"])
    return uploaded_file
