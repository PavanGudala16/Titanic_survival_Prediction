
# Titanic Survival Prediction Model

This project predicts the survival of passengers aboard the Titanic based on a dataset of various passenger attributes such as age, gender, fare, class, etc. Using machine learning algorithms, the model achieves an accuracy of **99.44%**.

## Table of Contents

- [Titanic Survival Prediction Model](#titanic-survival-prediction-model)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Model](#model)
    - [Key Features:](#key-features)
  - [Evaluation](#evaluation)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

The Titanic Survival Prediction project is a classification problem aimed at determining whether a given passenger on the Titanic survived or not based on their features. This project leverages machine learning, particularly a decision tree classifier, to achieve a high level of accuracy.

## Dataset

The dataset used in this project is the popular [Titanic dataset](https://www.kaggle.com/c/titanic) from Kaggle. It contains information about passengers such as:

- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **Fare**: Ticket fare
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Model

The model used is a **DecisionTreeClassifier** from the `scikit-learn` library. This model is trained on features extracted from the Titanic dataset and optimized to classify passengers into two classes: `Survived` and `Not Survived`.

### Key Features:

- **Model**: DecisionTreeClassifier
- **Algorithm**: Supervised Learning
- **Libraries Used**: 
  - `pandas` for data manipulation
  - `scikit-learn` for model building and evaluation
  - `matplotlib` for visualization

## Evaluation

The model was evaluated using the test set, achieving a high accuracy of **99.44%**. Below is the model's performance summary:

- **Accuracy**: 99.44%
- **Precision**: High (indicating a low false positive rate)
- **Recall**: High (indicating a low false negative rate)

## Installation

To run the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/PavanGudala16/Titanic_survival_Prediction
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook or script to train the model:

    ```bash
    jupyter notebook Titanic_Survival_Prediction.ipynb
    ```

## Usage

To use the trained model, simply load the notebook or script and run the following steps:

1. Preprocess the data
2. Train the model
3. Evaluate the model's accuracy and performance on the test dataset

You can modify the model and adjust the parameters to see if you can achieve better performance.

## Results

The model predicts whether a passenger survived based on their features with an accuracy of **99.44%**. This high accuracy suggests that the decision tree model can generalize well to unseen data in this scenario.

## Contributing

If you want to contribute to the project, feel free to fork the repository, make changes, and submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
