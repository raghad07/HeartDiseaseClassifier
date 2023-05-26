# HeartDiseaseClassifier

# Heart Disease Classification using Machine Learning Models
This repository contains a Python notebook that demonstrates the implementation of various machine learning models for heart disease classification. The models covered in this notebook include Logistic Regression, K-Nearest Neighbor, Support Vector Machines, Decision Trees, and Deep Neural Networks.

# Dataset
The dataset used in this project contains different characteristics of patients and their corresponding heart disease status. The features include age, sex, chest pain type, resting blood pressure, cholesterol level, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise relative to rest, ST slope, and the presence of heart disease.

# Preprocessing
Before applying the machine learning models, the dataset is preprocessed to handle categorical variables and perform feature normalization. The text values in columns such as RestingECG, ChestPainType, ExerciseAngina, ST_Slope, and Sex are converted to numerical values. Numerical features are then normalized using the StandardScaler from scikit-learn.

# Modelling
Three machine learning models are implemented and evaluated:

K-Nearest Neighbor (KNN): A KNN classifier is trained using the KNeighborsClassifier from scikit-learn. The optimal hyperparameters are selected using grid search.

Support Vector Machines (SVM): An SVM classifier is trained using the SVC from scikit-learn. The hyperparameters are tuned using grid search to achieve the best performance.

Decision Trees: A decision tree classifier is trained using the DecisionTreeClassifier from scikit-learn. The hyperparameters, such as maximum depth and minimum samples leaf, are optimized using grid search.

# Evaluation
The performance of each model is evaluated using various metrics, including accuracy, precision, recall, and the confusion matrix. The best hyperparameters for each model are selected based on their respective evaluation metrics.

# Results
The results obtained from the trained models are as follows:

K-Nearest Neighbor:

Accuracy: 0.8913
Precision: 0.8942
Recall: 0.8913
Support Vector Machines:

Accuracy: 0.9130
Precision: 0.9130
Recall: 0.9130
Decision Trees:

Accuracy: 0.8261
Precision: 0.8489
Recall: 0.8261
Conclusion
In this project, we demonstrated the implementation of various machine learning models for heart disease classification. The results show that the Support Vector Machines model achieved the highest accuracy, precision, and recall among the three models. However, further experimentation and fine-tuning of hyperparameters may be necessary to improve the performance of the Decision Trees model.
