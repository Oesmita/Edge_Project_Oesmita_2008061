# Logistic-Regression-SVM-Analysis

This project compares two machine learning algorithms—**Logistic Regression** and **Support Vector Machine (SVM)**—to classify the **Iris dataset**.

## Project Overview

In this project, we utilize two popular classification algorithms to classify the Iris dataset and compare their performance. The goal is to explore how both **Logistic Regression** and **Support Vector Machines** (SVM) perform on the same dataset, using various evaluation metrics such as accuracy, classification report, and confusion matrix.

## Dataset

The dataset used is the famous **Iris dataset** which consists of 150 samples of iris flowers from three different species (Setosa, Versicolour, and Virginica). Each sample has four features representing the length and width of the sepals and petals:

- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

This dataset is used to predict the species of the iris flowers based on the four feature measurements.

## Files and Directories

- **iris.csv**: The Iris dataset in CSV format containing 150 samples with 4 features and a target variable (species).
- **logistic_regression_svm_analysis.ipynb**: The Jupyter notebook containing the entire analysis. It includes the code to load, preprocess, and analyze the Iris dataset using both Logistic Regression and SVM.
  
## Steps Involved

### 1. **Data Loading and Preprocessing**
The project begins by loading the **iris.csv** file. After loading the dataset, the following steps are performed:
- **Target Encoding**: The species (target variable) is encoded into numeric values to be used in machine learning models.
  
### 2. **Data Splitting**
The dataset is split into training and testing sets (70% training, 30% testing) using **train_test_split** from `sklearn`. This ensures that the model is trained on one subset of the data and evaluated on a separate subset to prevent overfitting.

### 3. **Feature Scaling**
Both the training and testing features are scaled using **StandardScaler** to normalize the features so that all variables contribute equally to the model.

### 4. **Model Training**
Two machine learning algorithms are trained:
- **Logistic Regression**: A classification algorithm that works well with linearly separable data.
- **Support Vector Machine (SVM)**: A powerful classification method that works by finding the optimal hyperplane to separate classes in the feature space.

### 5. **Model Evaluation**
The models are evaluated on the test set using several metrics:
- **Accuracy**: The proportion of correctly predicted samples.
- **Classification Report**: A detailed report including precision, recall, and F1-score.
- **Confusion Matrix**: A matrix that shows the true positive, true negative, false positive, and false negative values for both models.

### 6. **Comparison**
The results of both models are compared based on the evaluation metrics, and confusion matrices are visualized to better understand the performance of the classifiers.

## How to Run the Project

1. **Upload the `iris.csv` file and the Jupyter notebook**:
   - Ensure that the `iris.csv` file and the `logistic_regression_svm_analysis.ipynb` notebook are in the same directory.
  
2. **Run the Jupyter notebook**:
   - Open the `logistic_regression_svm_analysis.ipynb` notebook in a Jupyter notebook environment (e.g., [Google Colab](https://colab.research.google.com/)).
   - Run all cells in the notebook to load, preprocess, train, and evaluate the models.

3. **Viewing Results**:
   - The notebook will display the accuracy of each model, along with classification reports and confusion matrices for further analysis.

## Dependencies

To run the project locally, you need to install the required Python libraries. You can do this by running the following command:

```bash
pip install -r requirements.txt

Required libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
Running the Project:

Upload the iris.csv file and the logistic_regression_svm_analysis.ipynb notebook to your Jupyter notebook environment (e.g., Google Colab or Jupyter Lab).
Run the notebook cells in order to:
Load the dataset
Preprocess the data
Train the Logistic Regression and SVM models
Evaluate the models' performance
Results:
The notebook will output accuracy scores, classification reports, and confusion matrices for both models, along with visualizations for better understanding.

Project Structure
iris.csv: The dataset.
logistic_regression_svm_analysis.ipynb: The Jupyter notebook with the full analysis.
requirements.txt: Lists the Python dependencies.
Conclusion
This project demonstrates how to apply Logistic Regression and SVM to a classic dataset and compare their performance using various metrics. Both algorithms are trained and tested on the same dataset, and the results are analyzed using classification reports and confusion matrices.
