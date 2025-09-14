# Decision Tree

A from-scratch implementation of a CART (Classification and Regression Tree) algorithm in Rust. This repository provides a library and a command-line script, eval_tree, for training, evaluating, and visualizing models on CSV datasets.

## Features

* Builds a decision tree from a CSV dataset.
* Supports both **Gini Impurity** and **Information Gain (Entropy)** as splitting criteria.
* Implements two types of pruning to prevent overfitting:
  * Pre-pruning: `max_depth` and `min_samples_split`.
  * post-pruning: Based on a minimum gain threshold.
* Classifies new data samples, including robust handling of missing feature values (e.g., `?`).
* Provides three robust evaluation methods:
  * Simple train/test split.
  * K-Fold Cross-Validation for a stable performance estimate.
  * Evaluation on a separate, pre-defined test file.
* Visualises the trained decision tree into an image file (.png, .svg, etc.) using Graphviz.


## Prerequisites

This project requires **Graphviz** to be installed on your system to render the decision tree images.

-   **macOS (using Homebrew):**
    ```bash
    brew install graphviz
    ```
-   **Debian/Ubuntu (using APT):**
    ```bash
    sudo apt-get install graphviz
    ```
-   **Windows (using Chocolatey):**
    ```bash
    choco install graphviz
   

## Installation

Create a virtual environment and install the required Python packages from `requirements.txt`:

    ```bash
    cargo build --release
    ```

## Usage

The primary script for training and evaluation is ```eval_tree```.
Run the script from your terminal, specifying the dataset and the desired options:

``` text
Usage: eval_tree [OPTIONS] <FILE_PATH>

Arguments:
  <FILE_PATH>  Path to the training CSV dataset file

Options:
      --test-file <TEST_FILE>
          Optional path to a separate CSV test dataset file. If provided, overrides split/k-fold
  -s, --split-ratio <SPLIT_RATIO>
          Proportion for training in a simple split [default: 0.8]
  -c, --criterion <CRITERION>
          The splitting criterion to use [default: gini] [possible values: entropy, gini]
  -p, --prune <PRUNE>
          Minimum gain to keep a branch (0.0 = no pruning) [default: 0.0]
      --plot <PLOT>
          Export the decision tree as a Graphviz image
  -k, --k-folds <K_FOLDS>
          Number of folds for k-fold cross-validation. If > 1, this overrides --split_ratio [default: 0]
      --max-depth <MAX_DEPTH>
          Maximum depth of the tree (None = unlimited)
      --min-samples-split <MIN_SAMPLES_SPLIT>
          Minimum number of samples required to split a node [default: 2]
  -h, --help
          Print help
  -V, --version
          Print version
```

## Examples

In the data folder there are some commonly used datasets in .cvs format:

| Name              | #Samples   | #Classes   | #Features |
| :-----            | ---:       | ---:       | ---:      |
| Tbc               |     10     |      2     |  5        |
| Iris              |    150     |      3     |  4        |
| Winequality red   |   1607     | 10 (6)     | 11        |
| Winequality white |   4906     | 10 (7)     | 11        |
| Adult train       |  32561     |      2     | 14        |
| Adult test        |  16282     |      2     | 14        |


### 1. Pruning and Visualisation

The ```demo``` cli app has two built-in examples (Tbc and Iris). The other datasets can be exploted with ```eval_tree```.

### 1. Demo: Train on Iris with Pruning and Visualisation

``` bash
cargo run --bin demo 2 --plot Assets/tree_iris.svg
```

Example output:
``` text
data/iris.csv: 150 data rows
header: ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
target column: 4 ('Species')
class labels: ["setosa", "versicolor", "virginica"]

PetalLength >= 3?
yes -> PetalWidth >= 1.8?
    yes -> PetalLength >= 4.9?
        yes -> 2: 43
        no  -> SepalLength >= 6?
            yes -> 2: 2
            no  -> 1: 1
    no  -> PetalLength >= 5?
        yes -> PetalWidth >= 1.6?
            yes -> SepalLength >= 7.2?
                yes -> 2: 1
                no  -> 1: 2
            no  -> 2: 3
        no  -> PetalWidth >= 1.7?
            yes -> 2: 1
            no  -> 1: 47
no  -> 0: 50

A branch was pruned: gain = 0.1461

PetalLength >= 3?
yes -> PetalWidth >= 1.8?
    yes -> PetalLength >= 4.9?
        yes -> 2: 43
        no  -> SepalLength >= 6?
            yes -> 2: 2
            no  -> 1: 1
    no  -> PetalLength >= 5?
        yes -> PetalWidth >= 1.6?
            yes -> SepalLength >= 7.2?
                yes -> 2: 1
                no  -> 1: 2
            no  -> 2: 3
        no  -> 1: 47, 2: 1
no  -> 0: 50

--- Classification Examples ---
----------------------------------------
Input Sample: [Numeric(6.0), Numeric(2.2), Numeric(5.0), Numeric(1.5)]
--> Predicted Class: 'virginica'

Detailed Scores (Leaf Node Counts / Weights):
    - virginica   : 3
----------------------------------------
----------------------------------------
Input Sample: [None, None, None, Numeric(1.5)]
--> Predicted Class: 'versicolor'

Detailed Scores (Leaf Node Counts / Weights):
    - versicolor  : 28
    - setosa      : 17
    - virginica   : 1
----------------------------------------
Generated temporary file: assets/tree_iris.dot
Decision tree exported to assets/tree_iris.svg

| ![Iris Decision Tree](assets/tree_iris.svg) |
| --- |


### 2. Eval: Simple Split with Pruning 

Train on the Winequality-red dataset [3] using an 80/20 split, apply post-pruning (--prune):

``` bash
cargo run --bin eval_tree --release -- data/winequality-red.csv --split-ratio 0.8 --prune 0.4
```
Example Output:
``` text
data/winequality-red.csv: 1599 data rows
header: ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
target column: 11 ('quality')
class labels: ["3", "4", "5", "6", "7", "8"]
Dataset loaded successfully with 1599 rows.

Performing a simple train/test split...
Data split into 1279 training samples and 320 test samples.
------------------------------
Training the decision tree model (criterion: gini)...
Trained a model with 645 nodes
Pruning the tree with min_gain = 0.4...
A branch was pruned: gain = 0.3200
...snip
A branch was pruned: gain = 0.3750
Pruned down to 523 nodes

Evaluating model accuracy on the test set...

--- Evaluation Result ---
Model Accuracy: 65.00%
------------------------------
```

### 2. 10-Fold Cross Validation on the Wine Quality Dataset

Perform a robust evaluation on the Wine Quality dataset [3], replicating the methodology from the original paper.
The original paper reported an accuracy of 62% for red wine and 65% for white wine (SVM classifier).

``` bash
cargo run --bin eval_tree --release -- data/winequality-red.csv -k 10 --prune 0.3
```

Example Output:
```
data/winequality-red.csv: 1599 data rows
header: ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
target column: 11 ('quality')
class labels: ["3", "4", "5", "6", "7", "8"]
Dataset loaded successfully with 1599 rows.

Performing 10-fold cross-validation...

--- Fold 1/10 ---
Training on 1440 samples, testing on 159 samples.
Trained a model with 723 nodes
Pruned model down to 643 nodes
Fold 1 Accuracy: 62.26%

...snip

--- Fold 10/10 ---
Training on 1431 samples, testing on 168 samples.
Trained a model with 699 nodes
Pruned model down to 617 nodes
Fold 10 Accuracy: 66.67%

==============================
Cross-Validation Summary
==============================
Average Accuracy: 62.26%
Standard Deviation: 4.35%
==============================
```

### 3. Pre-Pruning with a separate Test Set (Adult Dataset)

Train on the Adult (US Census) dataset, which has a pre-defined train/test split and contains missing values. Use
pre-pruning (`--max_depth` and `--min_samples_split`) to control tree size and prevent overfitting.

``` bash
cargo run --bin eval_tree --release -- data/adult_train.csv --test-file data/adult_test.csv --max-depth 10 --min-samples-split 10
```
Example Output: 
``` text
data/adult_train.csv: 32561 data rows
header: ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
target column: 14 ('income')
class labels: ["<=50K", ">50K"]

data/adult_test.csv: 16281 data rows
header: ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
target column: 14 ('income')
class labels: ["<=50K", ">50K"]

Training decision tree (criterion: gini)...
Trained a model with 553 nodes

Evaluating model accuracy on the external test set...

--- Evaluation Result ---
Model Accuracy: 86.03%
------------------------------
```

## References

1. [TBC Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
2. [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)
3. [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
4. [Modeling wine preferences by data mining from physicochemical properties, Paulo Cortez et al.](https://repositorium.sdum.uminho.pt/bitstream/1822/10029/1/wine5.pdf)
5. [Adult (1994 Census) Dataset](https://archive.ics.uci.edu/dataset/2/adult)

