import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_recall_fscore_support, confusion_matrix
from models_and_plotting import eta_and_lambda_grid, PreRec, plotGrid, plotMultiConfusion, plotConfusion
from PreprosessingAstroids import get_haz_data, get_classi_data


def save_data(y_pred_test, y_pred_train, y_test, y_train, X_test, X_train, pre, rec, data, filename):
    np.save(f"y_pred_test_{filename}", y_pred_test)
    np.save(f"y_pred_train_{filename}", y_pred_train)
    np.save(f"y_test_{filename}", y_test)
    np.save(f"y_train_{filename}", y_train)
    np.save(f"X_test_{filename}", X_test)
    np.save(f"X_train_{filename}", X_train)
    np.save(f"pre_{filename}", pre)
    np.save(f"rec_{filename}", rec)
    np.save(f"data_{filename}", data)


def load_data(filename):
    y_pred_test = np.load(f"y_pred_test_{filename}.npy")
    y_pred_train = np.load(f"y_pred_train_{filename}.npy")
    y_test = np.load(f"y_test_{filename}.npy")
    y_train = np.load(f"y_train_{filename}.npy")
    X_test = np.load(f"X_test_{filename}.npy")
    X_train = np.load(f"X_train_{filename}.npy")
    pre = np.load(f"pre_{filename}.npy")
    rec = np.load(f"rec_{filename}.npy")
    data = np.load(f"data_{filename}.npy")
    return y_pred_test, y_pred_train, y_test, y_train, X_test, X_train, pre, rec, data


def get_parameters(filename):
    with open(filename, "r") as infile:
        infile.readline()
        infile.readline()
        infile.readline()
        text = infile.read()
        text = text.replace("\'", "\"")
        parameters = json.loads(text)

    activation = parameters["activation"]
    hidden_layers = parameters["layer_1"]
    solver = parameters["optimizer"]
    lmbda_value = parameters["lmbda_value"]
    learning_rate = parameters["learning_rate"]
    return activation, hidden_layers, solver, lmbda_value, learning_rate


def create_grid(lmbda_value, eta_value):
    d_eta = 0.05
    d_lmbda = 0.5

    # Generer verdiene rundt sentrum for x og y aksen
    eta_vals = np.linspace(eta_value - 2 * d_eta, eta_value + 2 * d_eta, 5)
    lmbda_vals = np.linspace(lmbda_value - 2 * d_lmbda, lmbda_value + 2 * d_lmbda, 5)
    return np.abs(eta_vals), np.abs(lmbda_vals)


def binary(filename, data_imbalance):
    activation, hidden_layers, solver, lmbda_value, learning_rate = get_parameters(filename)
    eta_vals, lmbda_vals = create_grid(lmbda_value, learning_rate)

    features = "all"
    X_train, X_test, y_train, y_test = get_haz_data(features, data_imbalance, random_state=None, no_plotting=True)

    model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=lmbda_value, learning_rate_init=learning_rate)
    clf = model.fit(X_train, y_train)
    
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    pre, rec, _ = precision_recall_curve(y_test, y_pred_test)
    data = eta_and_lambda_grid(X_train, X_test, y_train, y_test, eta_vals, lmbda_vals, hidden_layers, activation, solver)

    save_data(y_pred_test, y_pred_train, y_test, y_train, X_test, X_train, pre, rec, data, f"{data_imbalance}_binary")
    binary_analysis(data_imbalance)


def binary_analysis(data_imbalance):
    y_pred_test, y_pred_train, y_test, y_train, X_test, X_train, pre, rec, data = load_data(f"{data_imbalance}_binary")
    PreRec(pre, rec, "Precision/Recall", filename="PreRec_RUS")
    plotConfusion(y_test, y_pred_test, "Confusion Matrix", filename="Confusion_RUS")



    # Printing different scores
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)

    
    # precision_recall_fscore_support()
    tmp = precision_recall_fscore_support(y_test, y_pred_test)
    precision = tmp[0][0]
    recall = tmp[1][0]
    f1_score = tmp[2][0]
    import ipdb;ipdb.set_trace()
    tn, _, _, tp = confusion_matrix(y_test, y_pred_test).ravel()
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"f1_score = {f1_score}")
    print(f"TN = {tn}")
    print(f"TP = {tp}")

def mulitclass(filename):
    activation, hidden_layers, solver, lmbda_value, learning_rate = get_parameters(filename)
    eta_vals, lmbda_vals = create_grid(lmbda_value, learning_rate)

    features = "all"
    X_train, X_test, y_train, y_test = get_classi_data(random_state=None, no_plotting=True)

    model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=lmbda_value, learning_rate_init=learning_rate)
    clf = model.fit(X_train, y_train)
    
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    pre, rec, _ = precision_recall_curve(y_test, y_pred_test)
    data = eta_and_lambda_grid(X_train, X_test, y_train, y_test, eta_vals, lmbda_vals, hidden_layers, activation, solver)

    save_data(y_pred_test, y_pred_train, y_test, y_train, X_test, X_train, pre, rec, data, "multiclass")
    # multiclass_analysis()


def multiclass_analysis():
    # Was not able to finish this
    y_pred_test, y_pred_train, y_test, y_train, X_test, X_train, pre, rec, data = load_data("multiclass")
    title = "multiclass"
    PreRec(pre, rec, title) # title kommer
    plotConfusion(y_test, y_pred, title)
    plotGrid(data, "eta", "lambda", title)

    # # Create tabel:
    # y_pred_test
    # y_test

    # y_pred_train
    # y_train


def main(saved=False):
    if not(saved): 
        binary("best_parameters_binary_RUS_5372s_03-54-06_18-12-2023.txt", "RUS")
    else:
        binary_analysis("RUS")

if __name__ == "__main__":
    main()

