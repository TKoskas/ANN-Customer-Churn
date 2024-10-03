# Artificial Neural Network for Customer Churn Prediction

This repository contains a Jupyter Notebook that demonstrates the creation and training of an artificial neural network (ANN) for predicting customer churn. The notebook is well-structured with Markdown cells for explanations and code cells for implementation.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Building the ANN](#building-the-ann)
4. [Training the ANN](#training-the-ann)
5. [Making Predictions and Evaluating the Model](#making-predictions-and-evaluating-the-model)
6. [Practical Applications](#practical-applications)

## Introduction

This notebook presents the creation and training of an artificial neural network for predicting customer churn. The dataset used is `Churn_Modelling.csv`, which contains information about bank customers.

## Data Preprocessing

### Importing the Dataset

The dataset is imported using the `pandas` library. The features (`X`) and the target variable (`y`) are extracted from the dataset.

```python
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
```

### Encoding Categorical Data

#### Label Encoding the "Gender" Column

The "Gender" column is label encoded using the `LabelEncoder` from `sklearn`.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
```

#### One Hot Encoding the "Geography" Column

The "Geography" column is one-hot encoded using the `ColumnTransformer` and `OneHotEncoder` from `sklearn`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

### Splitting the Dataset

The dataset is split into training and test sets using the `train_test_split` function from `sklearn`.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

### Feature Scaling

The features are scaled using the `StandardScaler` from `sklearn`.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Building the ANN

The ANN is built using the `Sequential` model from `tensorflow.keras`. The model consists of an input layer, two hidden layers, and an output layer.

```python
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

## Training the ANN

The ANN is compiled and trained using the `adam` optimizer and the `binary_crossentropy` loss function.

```python
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
```

## Making Predictions and Evaluating the Model

### Predicting the Result for a Specific Customer

In this section, we will utilize our Artificial Neural Network (ANN) model to predict whether a specific customer is likely to leave the bank based on their provided information. Here are the details of the customer in question:

- **Geography**: France
- **Credit Score**: 600
- **Gender**: Male
- **Age**: 40 years old
- **Tenure**: 3 years
- **Balance**: $ 60000
- **Number of Products**: 2
- **Has a Credit Card**: Yes
- **Is an Active Member**: Yes
- **Estimated Salary**: $ 50000

### Making the Prediction

The model is used to predict whether the specific customer will leave the bank based on the provided information.

```python
# Predicting for a specific customer
prediction = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5
print(prediction)
```

### Interpretation of the Result

According to our model, **this customer is expected to stay with the bank**. The model outputs a probability, which indicates that the customer is unlikely to leave if this probability is below 0.5.

### Important Notes

1. **Input Format**:
   - The feature values are input in a double pair of square brackets `[[ ]]`. This is crucial as the `predict` method expects the input to be in a 2D array format. Using double brackets ensures the input is properly formatted.

2. **One-Hot Encoding for Geography**:
   - Notice that the geography value "France" is represented as `1, 0, 0` in the first three columns instead of as a string. This is because the `predict` method requires one-hot encoded values for categorical features.
   - It is important to include these values in the correct order since dummy variables are typically created in the first columns.

### Predicting on the Test Set

Next, we will use the model to make predictions on the entire test set and compare the predicted results with the actual outcomes.

```python
# Predicting results on the test set
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Displaying predicted results alongside actual values
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
```

### Making the Confusion Matrix

The confusion matrix and accuracy score are calculated to evaluate the model's performance.

```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
```
## Model Result

### Practical Applications
This ANN model can be applied in various scenarios in the banking sector, including:

- **Customer Retention**: Identifying customers at risk of leaving the bank.
- **Targeted Marketing**: Developing personalized offers for customers based on churn predictions.

### Model Results and Interpretation
**Conclusion**: The model predicts that the customer will stay with the bank (prediction: **False**). By default, the model outputs True if the probability that the customer will leave the bank is greater than 50%, and False otherwise.

To obtain the exact probability that the customer will leave the bank, simply print the model's prediction output without applying the `> 0.5` comparison.

**Probability that the customer will leave the bank**: **4.54%** (This indicates a low likelihood of the customer churning.)

```python
# Predicting the result of a single observation
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
**Output**:
```python
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step
[[0.04542754]]  # This means the probability of leaving the bank is 4.54%.


Therefore, our ANN model predicts that this customer stays in the bank!
