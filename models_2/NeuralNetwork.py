from sklearn.neural_network import MLPClassifier


def nn_param_selector(*n_neurons):
    hidden_layer_sizes = []
    for neuron in n_neurons:
        hidden_layer_sizes.append(neuron)
    hidden_layer_sizes = tuple(hidden_layer_sizes)
    params = {"hidden_layer_sizes": hidden_layer_sizes}

    model = MLPClassifier(**params)
    return model