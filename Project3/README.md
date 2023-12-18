# FYS-STK4155: Project 2:


## Project overview:
This is the code for reproducing our results in _Project 3_ of **FYS-STK4155** for the Autumn 2023 semester at UiO. The graphs from the plotting of the neural network are stored in the _plots_ folder. 

## Installation instructions:
To install all the necessary packages, run this code:

```Python
pip install -r requirements.txt
```

where **requirements.txt** contains all the required packages to run the code for this repository.


## Datasets: Classification problem
The dataset is from JPL Small-Body Database Search Engine. It can be loaded from Kaggle. More information on the dataset can be found on:

[https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset/data](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset/data)


## Usage guide:
The file **decision_trees.py** contains the code for our analysis of the Decision Trees (and its derivatives such as Random Forests, etc). 
The `lines 7 - 18` shows how you can configure this script to run different analysis.

```Python
python3 decision_trees.py
```

To generate the result for the Neural Network we must first run the script **hyperparameters_neural_network.py**:

```Python
python3 hyperparameters_neural_network.py
```

This script generates the best hyperparameters that Optuna manages to find after a certain time. The files:

- **best_parameters_binary_RUS_5372s_03-54-06_18-12-2023.txt**
- **best_parameters_binary_unbalanced_1481s_02-32-23_18-12-2023.txt**
- **best_parameters_mulit_class_2406s_02-57-05_18-12-2023.txt**

are examples of this.

To run the analysis of the Neural Network hyperparameters run **neural_network.py**:

```Python
python3 neural_network.py
```

this will generate the folder _plots_. Disregard the multiclass functions. I did not have time (more like energy) to complete them, but 
the framework is in place for that analysis, also my work PC could not run the entire dataset (at least in a reasonable time).