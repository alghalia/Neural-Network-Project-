
# Employee Attrition and Department Prediction Model- Neural-Network-Project

This project aims to predict employee attrition (whether an employee is likely to leave) and the department best suited for an employee. The model uses a branched neural network with shared layers for feature processing and separate branches for each prediction target.

## Project Overview

1. **Attrition Prediction**: This branch predicts if an employee will stay or leave (binary classification).
2. **Department Prediction**: This branch predicts the department an employee is best suited for (multi-class classification).

## Data Preprocessing

- **Encoding**: Categorical variables such as `BusinessTravel` and `OverTime` were encoded. The `Department` and `Attrition` columns were one-hot encoded for compatibility with the model’s output structure.
- **Scaling**: All features were standardized using `StandardScaler` to improve model performance.

## Model Architecture

- **Input Layer**: Accepts standardized features of each employee.
- **Shared Layers**: Two hidden layers shared by both branches, allowing the model to learn general features about employees.
- **Attrition Branch**: A hidden layer followed by an output layer with a sigmoid activation function for binary classification.
- **Department Branch**: A hidden layer followed by an output layer with a softmax activation function for multi-class classification.

## Activation Functions

- **Sigmoid** for `attrition_output`: Provides probabilities for binary classification (staying or leaving).
- **Softmax** for `department_output`: Provides a probability distribution across department classes.

## Evaluation Metrics

- **Accuracy**: Primary metric for both branches, though additional metrics like precision, recall, and F1-score can be useful, especially for imbalanced classes in attrition prediction.

## Usage

1. **Preprocess the Data**:
   - Ensure all categorical data is encoded.
   - Scale the features using `StandardScaler`.

2. **Train the Model**:
   - The model is compiled with `binary_crossentropy` for attrition and `categorical_crossentropy` for department prediction.
   - Use model fitting with the training dataset to train both branches simultaneously.

3. **Evaluate the Model**:
   - Evaluate on test data using accuracy as a metric, along with other potential metrics for better insights.

## Potential Improvements

1. **Feature Engineering**: Adding derived features, such as tenure or satisfaction score, could enhance predictive power.
2. **Hyperparameter Tuning**: Experiment with learning rate, batch size, and layer configuration.
3. **Ensemble Methods**: Combining predictions from multiple models could yield better accuracy.
4. **Regularization**: Techniques like dropout or L2 regularization can reduce overfitting.

## Requirements

- Python
- TensorFlow
- Pandas
- Scikit-learn

Install dependencies with:

```bash
pip install tensorflow pandas scikit-learn
```

## Conclusion

This model provides a foundation for predicting employee attrition and department suitability. Improvements through feature engineering, tuning, and regularization could further enhance its effectiveness.
