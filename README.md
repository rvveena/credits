# Credit Card Fraud Detection using Machine Learning Algorithms
----

**Project Overview: Credit Card Fraud Detection using Machine Learning**

The goal of this project, titled *"Credit Card Fraud Detection using Machine Learning,"* is to develop an effective system that identifies fraudulent credit card transactions. By leveraging several machine learning algorithms, we aim to detect fraudulent activities and minimize financial losses for both credit card companies and their customers. The models implemented for this task include Logistic Regression, Support Vector Machine (SVM), Naive Bayes, K-Nearest Neighbors (KNN), Random Forest, AdaBoost, and XGBoost.

----

 **Data Description**

The dataset used for this project consists of credit card transaction records, where each entry includes features such as transaction amount, timestamp, and anonymized customer information. Each transaction is labeled as either fraudulent or non-fraudulent. This labeled data allows us to train supervised machine learning models that can learn patterns from these features and predict whether a new transaction is fraudulent.

The dataset can be accessed here: [Credit Card Fraud Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

-----

 **Machine Learning Models**

- **Logistic Regression**: A simple yet effective classification model, Logistic Regression predicts the probability of a transaction being fraudulent based on a logistic function. It assigns a binary label (fraud or non-fraud) based on a threshold value, making it easy to interpret the results.

- **Support Vector Machine (SVM)**: SVM is a powerful algorithm for classification tasks, where it constructs a hyperplane (or a set of hyperplanes) that separates fraudulent and non-fraudulent transactions. The choice of kernel and hyperparameter tuning allows SVM to adapt to complex, high-dimensional data.

- **Naive Bayes**: This probabilistic classifier uses Bayes' theorem to estimate the probability of fraud given the features of a transaction. Despite its assumption that features are independent, Naive Bayes often performs well in real-world scenarios due to its simplicity and efficiency.

- **K-Nearest Neighbors (KNN)**: A non-parametric algorithm, KNN classifies a transaction by comparing it to the k nearest transactions in the feature space. By adjusting the number of neighbors (k) and the distance metric, KNN can effectively identify patterns of fraud.

- **Random Forest**: Random Forest is an ensemble learning method that constructs multiple decision trees, each trained on random subsets of the data. It combines the predictions of individual trees to produce a more accurate and robust model, making it particularly useful for imbalanced datasets like credit card fraud.

- **AdaBoost**: AdaBoost (Adaptive Boosting) is an ensemble method that combines weak classifiers by giving more weight to misclassified samples in subsequent iterations. This helps the model focus on the harder-to-classify fraudulent transactions, improving its overall accuracy.

- **XGBoost**: An optimized gradient boosting algorithm, XGBoost is known for its high performance in machine learning tasks. It builds an ensemble of decision trees, but with more advanced regularization and optimization techniques, making it particularly effective at preventing overfitting.

----
**Evaluation Metrics**

The performance of these models is evaluated using both initial and main metrics, with the aim of achieving a well-balanced and effective fraud detection system.

- **Initial Metrics**:
  - **Accuracy**: Measures the overall correctness of the model's predictions (the proportion of correct predictions out of all predictions). While useful, accuracy alone can be misleading in imbalanced datasets like fraud detection.
  - **Precision**: Precision focuses on the proportion of actual fraudulent transactions among those the model classified as fraudulent. It helps assess the model’s ability to avoid false positives (incorrectly classifying legitimate transactions as fraud).
  - **F1-Score**: The harmonic mean of precision and recall, F1-Score provides a balanced evaluation by considering both false positives and false negatives. It’s especially useful when dealing with imbalanced datasets.

------
- **Main Metrics**:
  - **Recall**: Recall, or sensitivity, measures the proportion of actual fraudulent transactions that are correctly identified by the model. It is crucial for minimizing false negatives (failing to detect fraud).
  - **AUC/ROC Curve**: The Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve helps evaluate the trade-off between the true positive rate (recall) and the false positive rate. A higher AUC indicates a better model, as it means the model is better at distinguishing between fraudulent and non-fraudulent transactions across different thresholds.

By carefully considering these metrics, the project aims to build a fraud detection system that minimizes financial losses while providing a secure environment for credit card transactions.

---

