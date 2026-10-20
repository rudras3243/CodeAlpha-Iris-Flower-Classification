# Iris Flower Classification Using Machine Learning

## 1. Introduction
The objective of this project is to build a machine learning classification model that identifies the species of an Iris flower from its physical measurements.

The three target species are:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## 2. Dataset
The supplied `Iris.csv` contains 150 records and six columns:
- Id
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species

There are 50 records for each species and no missing values.

The `Id` column is excluded from model training because it is only an identifier. The four flower measurements are used as input features.

## 3. Methodology
1. Load the CSV dataset using Pandas.
2. Check the dataset for missing values and class distribution.
3. Select the four measurement columns as features.
4. Select Species as the target.
5. Split the data into 80% training and 20% testing data using stratification.
6. Standardize the feature values.
7. Train a Support Vector Machine (SVM) classifier with an RBF kernel.
8. Predict the species for the test data.
9. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## 4. Model Comparison
| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 93.33% |
| K-Nearest Neighbors | 93.33% |
| Decision Tree | 93.33% |
| Random Forest | 90.00% |
| SVM (RBF) | 96.67% |

For this particular train/test split, SVM produced the highest test accuracy and was therefore used as the main model.

## 5. Results
The SVM model achieved **96.67% test accuracy** on 30 test samples.

Classification report:
- Iris-setosa: Precision 1.00, Recall 1.00, F1-score 1.00
- Iris-versicolor: Precision 1.00, Recall 0.90, F1-score 0.95
- Iris-virginica: Precision 0.91, Recall 1.00, F1-score 0.95

Confusion matrix:
[[10, 0, 0],
 [0, 9, 1],
 [0, 0, 10]]

This means 29 out of 30 test flowers were classified correctly. One versicolor flower was classified as virginica.

## 6. Conclusion
The project demonstrates a complete basic machine learning classification workflow: data loading, preprocessing, train/test splitting, model training, prediction, and evaluation.

The four flower measurements contain enough information to classify the Iris species with high accuracy. The SVM model achieved 96.67% accuracy on the selected test set.

## 7. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 8. How to Run
Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Keep `Iris.csv` in the same folder as `iris_classification.py`, then run:

```bash
python iris_classification.py
```

## 9. Viva Questions
1. What is classification?
   Classification is supervised learning where a model predicts a category or class.

2. Why is this a classification problem?
   Because the output is one of three species categories.

3. Why was Id removed?
   Id is an identifier and does not represent a flower measurement.

4. What is train-test split?
   It separates data used to learn the model from data used to evaluate it.

5. Why is feature scaling used?
   Scaling puts features on comparable numerical scales and is especially useful for distance/margin-based models such as SVM.

6. What is SVM?
   Support Vector Machine is a supervised learning algorithm that finds a decision boundary separating classes; kernels can model nonlinear boundaries.

7. What is accuracy?
   Accuracy is the proportion of test predictions that are correct.

8. What is a confusion matrix?
   It shows the actual classes against the predicted classes and helps identify which classes are being confused.
