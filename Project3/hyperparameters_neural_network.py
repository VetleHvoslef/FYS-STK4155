import optuna
import time
import pprint
import datetime
from PreprosessingAstroids import get_haz_data, get_classi_data
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


def define_architecture(trial, layers=2):
    n_layers = trial.suggest_int("n_layers", 1, layers)

    if n_layers == 1:
        layer_1 = trial.suggest_int("layer_1", 10, 100)
        return layer_1
    elif n_layers == 2:
        layer_1 = trial.suggest_int("layer_1", 10, 100)
        layer_2 = trial.suggest_int("layer_2", 0, 100)
        return layer_1, layer_2
    else:
        layer_1 = trial.suggest_int("layer_1", 10, 100)
        layer_2 = trial.suggest_int("layer_2", 0, 100)
        layer_3 = trial.suggest_int("layer_3", 0, 100)
        return layer_1, layer_2, layer_3



def objective_binary(trial, data_imbalance, X, y):
    # Setting the parameters:
    activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("optimizer", ["lbfgs", "sgd", "adam"])
    hidden_layers = define_architecture(trial)
    learning_rate = trial.suggest_float("learning_rate", 1E-3, 1E5, log = True)
    lmbda_value = trial.suggest_float("lmbda_value", 1E-6, 1, log = True)

    features = "all" # 'all', 'select', 'one'

    if data_imbalance == "unbalanced" or data_imbalance == "OS":
        model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=lmbda_value, learning_rate_init=learning_rate)
        X_train, X_test, y_train, y_test = get_haz_data(features, data_imbalance, random_state=None, no_plotting=True)

        try:
            clf = model.fit(X_train, y_train)
            accuracy_score = clf.score(X_test, y_test)
        except Exception:
            accuracy_score = 0
        return accuracy_score

    # Getting the data:


    # Defining the model and training the model:
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=lmbda_value, learning_rate_init=learning_rate)
    clf = make_pipeline(preprocessing.StandardScaler(), model)

    try:
        accuracy_scores = cross_val_score(clf, X, y)
        accuracy_score = accuracy_scores.mean()
    except Exception:
        accuracy_score = 0
    return accuracy_score


def objective_multi_class(trial):
    # Setting the parameters:
    activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("optimizer", ["lbfgs", "sgd", "adam"])
    hidden_layers = define_architecture(trial, layers=3)
    learning_rate = trial.suggest_float("learning_rate", 1E-3, 1E5, log=True)
    lmbda_value = trial.suggest_float("lmbda_value", 1E-6, 1, log=True)

    # Getting the data:
    X_train, X_test, y_train, y_test, _ = get_classi_data(random_state=None, no_plotting=True)
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=lmbda_value, learning_rate_init=learning_rate)

    # Defining the model and training the model:
    try:
        clf = model.fit(X_train, y_train)
        accuracy_score = clf.score(X_test, y_test)
    except:
        accuracy_score = 0
    return accuracy_score


def best_parameters_binary(n_trials=100, X=None, y=None, data_imbalance="RUS"):
    now = datetime.datetime.now()

    start = time.perf_counter()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_binary(trial, data_imbalance, X, y), n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
    end = time.perf_counter()

    time_and_date = now.strftime("%H-%M-%S_%d-%m-%Y")
    total_runtime = int(abs(end - start))
    filename = f"best_parameters_binary_{data_imbalance}_{total_runtime}s_{time_and_date}.txt"
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Score: {study.best_trial.value}")
    print(f"Saving best parameters to file: {filename}")

    with open(filename, "w") as outfile:
        outfile.write(f"Total runtime of the optimization: {total_runtime}\n")
        outfile.write(f"Number of trials: {n_trials}\n")
        outfile.write(f"Score: {study.best_trial.value}\n")
        text = pprint.pformat(study.best_params)
        outfile.write(text)
        outfile.write("\n")


def best_parameters_multi_class(n_trials=100):
    now = datetime.datetime.now()

    start = time.perf_counter()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_multi_class, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
    end = time.perf_counter()

    time_and_date = now.strftime("%H-%M-%S_%d-%m-%Y")
    total_runtime = int(abs(end - start))
    filename = f"best_parameters_mulit_class_{total_runtime}s_{time_and_date}.txt"
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Score: {study.best_trial.value}")
    print(f"Saving best parameters to file: {filename}")

    with open(filename, "w") as outfile:
        outfile.write(f"Score: {study.best_trial.value}\n")
        outfile.write(f"Total runtime of the optimization: {total_runtime}")
        outfile.write(f"Number of trials: {n_trials}")
        text = pprint.pformat(study.best_params)
        outfile.write(text)
        outfile.write("\n")

    

def main():
    X, y = get_haz_data("all", "RUS", random_state=None, return_X_y=True, no_plotting=True)
    best_parameters_binary(1000, X, y)
    # best_parameters_binary(10, data_imbalance="unbalanced")
    # best_parameters_multi_class(10)

## Optimaliser hyperparameterne med optuna:
if __name__ == "__main__":
    main()



