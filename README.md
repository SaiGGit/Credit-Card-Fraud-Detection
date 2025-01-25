# Fraud Detection System

This repository contains a **Fraud Detection System** built with a deep learning model to predict fraudulent transactions based on transaction data. The application is built using Flask for the web interface and PyTorch for training the machine learning model. The system allows users to manually input transaction data, upload CSV files for batch processing, and download prediction results.

## Features

- **Manual Prediction**: Users can input transaction details (e.g., distance from home, repeat retailer, etc.) to predict if the transaction is fraudulent.
- **Batch Prediction (CSV Upload)**: Upload a CSV file containing multiple transactions, and the system will predict fraud for each transaction.
- **View Transaction Data**: View all transactions stored in the database, including predictions and confidence levels.
- **Download Data**: Download all transaction data (including predictions) as a CSV file.

## Technologies

- **Flask**: Web framework for creating the web interface and API endpoints.
- **PyTorch**: Deep learning library used to train and deploy the fraud detection model.
- **SQLAlchemy**: ORM for managing the SQLite database.
- **SMOTE**: Used for handling class imbalance during training by oversampling the minority class.
- **Joblib**: For saving the scaler data used in data preprocessing.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: For model evaluation metrics and data splitting.

### Prerequisites

1. **Python 3.7+**
2. **Pip** (Python package manager)


## Usage

### Manual Prediction

1. Go to the **/manual** page.
2. Input the transaction details (e.g., distance from home, distance from last transaction, etc.).
3. Click the **Predict** button to get the fraud prediction along with the confidence level.

### Batch Prediction (CSV Upload)

1. Go to the **/upload_csv** page.
2. Upload a CSV file containing transaction data (the file should have the same columns as the training data).
3. The system will process the file, make predictions, and show the results.

### View Transaction Data

1. Go to the **/show_data** page.
2. View the list of all transactions stored in the database, including prediction results and confidence.

### Download Transaction Data

1. Go to the **/show_data** page.
2. Click the **Download CSV** button to download the transaction data (including predictions) as a CSV file.



## Machine Learning Model

The fraud detection model is built using PyTorch, and it leverages a neural network to classify transactions as either fraudulent or not. Below is an overview of the key components and steps involved in the model training and usage:

### Data Preparation

1. **Loading the Data**:
   - The model uses transaction data from the `transaction_data.csv` file. This dataset contains various features such as `distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price`, and others.
   
2. **Feature Scaling**:
   - The features are normalized using min-max scaling to ensure that all input features are on a similar scale. This helps improve the model's performance and training stability.
   - The scaling parameters (minimum and maximum values) are saved into a `scaler.pkl` file for reuse during inference.

3. **Handling Imbalanced Data**:
   - Since fraud detection often involves highly imbalanced datasets (with fewer fraudulent transactions), **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance the dataset by generating synthetic examples of the minority class (fraudulent transactions).

4. **Train/Test Split**:
   - The dataset is split into training and testing sets (80% for training, 20% for testing) using `train_test_split` from `sklearn`.

### Model Architecture

The model is a feed-forward neural network with the following architecture:

- **Input Layer**: Accepts the features of the transactions.
- **Hidden Layers**:
  - Layer 1: Fully connected layer with 128 units, followed by batch normalization and ReLU activation.
  - Layer 2: Fully connected layer with 64 units, followed by batch normalization and ReLU activation.
  - Layer 3: Fully connected layer with 32 units, followed by batch normalization and ReLU activation.
  - Dropout is applied after each hidden layer to prevent overfitting.
- **Output Layer**: A single unit that outputs the probability of a transaction being fraudulent, using a sigmoid activation function.

### Model Training

- The model is trained using **Binary Cross-Entropy Loss** (`BCEWithLogitsLoss`) since it's a binary classification problem.
- **Adam Optimizer** is used to minimize the loss during training.
- The model is trained over 5 epochs with a batch size of 128.
- During each epoch, the model calculates the **AUC-ROC** score to evaluate its performance on the training set.

### Model Evaluation

After training, the model is evaluated on the test set. The following metrics are computed:

- **Test Loss**: The binary cross-entropy loss on the test data.
- **AUC-ROC**: The Area Under the Receiver Operating Characteristic Curve, indicating the model's ability to distinguish between fraudulent and non-fraudulent transactions.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual fraudulent transactions.

### Model Saving

- After training and evaluation, the trained model is saved to the file `fraud_detection_model.pth`.
- The scaling parameters are also saved to `scaler.pkl`, which are necessary for preprocessing during inference.

### Model Inference

The model can be used for inference through the following steps:

1. Preprocess the input data using the same scaling technique that was applied during training (using the saved `scaler.pkl`).
2. Pass the preprocessed data to the trained model to get the predicted probability of fraud.
3. If the probability is greater than a predefined threshold (0.5), classify the transaction as fraudulent.
