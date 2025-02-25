# **Assignment 2**

This assignment is designed to provide hands-on experience with implementing machine learning concepts and algorithms. You will work with the provided dataset to complete the tasks outlined below.


## **Objectives**
- Gain a deeper understanding of core machine learning concepts
- Implement Logistic Regression, Linear Regression, and other ML models
- Evaluate model performance using accuracy, MSE, and visualization techniques



## **Dataset**
Use the Pokémon dataset (CSV format) available in the same folder as this assignment, or download it from: [Kaggle - Pokémon Dataset](https://www.kaggle.com/abcsds/pokemon)


## **Learning Resources**
- [Machine Learning - YouTube (Statquest)](https://www.youtube.com/c/joshstarmer)
- [Learn Intro to Machine Learning | Kaggle](https://www.kaggle.com/learn/intro-to-machine-learning) (Implementation)
- [Start Here with Machine Learning](https://machinelearningmastery.com/start-here/) (Comprehensive overview)
- [Introduction | Machine Learning Notebook](https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/mlcc/intro_to_ml.ipynb)
- [Stanford CS229: Machine Learning Full Course by Andrew Ng](https://www.youtube.com/watch?v=jGwO_UgTS7I) (Mathematical foundations)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- Additional resources: Google, ChatGPT, DeepSeek, etc.
- Refer to **D11 Timeline.md** for further recommended resources.



## **Task 1: Predict a Pokémon’s Type Based on Its Stats**

Can you predict a Pokémon’s primary type using only its stats?

### **1. Data Preprocessing and Visualization**
- Select relevant features for predicting the Pokémon type
- Use the primary type as the target variable
- Perform feature normalization
- One-hot encode categorical labels
- Split the dataset into training and testing sets (80:20 split)

### **2. Baseline Model: Logistic Regression**
- Train a Logistic Regression model on the training data
- Evaluate its accuracy

### **3. Explore Advanced Models**
- **Random Forest Classifier** – Captures feature importance effectively
- **XGBoost** – High-performance model for structured data
- **SVM (RBF Kernel)** – Captures complex relationships in data
- **kNN Classifier** – A simple and intuitive approach based on nearest neighbors

### **4. Model Evaluation**
- Plot training accuracy and loss versus epochs
- Plot validation accuracy and training accuracy versus epochs
- Generate a confusion matrix for the predicted labels


## **Task 2: Predict a Pokémon's Offensive Power (CAS) Using Regression**

Can we estimate a Pokémon’s attack ability without using its Attack or Special Attack stats?

### **1. Create a New Feature: Combined Attack Score (CAS)**
Define **CAS** as:
\[
CAS = Attack + Special Attack + Speed*0.5
\]
- Add CAS as a new column in the dataset

### **2. Train a Linear Regression Model to Predict CAS**
- Exclude **Attack, Special Attack, and Speed** from the feature set
- Set **CAS** as the target variable
- Train a basic Linear Regression model

### **3. Model Evaluation**
- Use **Mean Squared Error (MSE)** as the performance metric
- Plot **Predicted CAS vs. Actual CAS**

---

## **Task 3: Identifying Hidden Legendary Pokémon Using CAS**

Some non-legendary Pokémon have CAS values similar to legendary Pokémon. Can we identify them?

### **1. Compare CAS Distributions**
- Plot a **CAS histogram** for **Legendary vs. Non-Legendary Pokémon**
- Identify non-legendary Pokémon with CAS values close to legendary Pokémon

### **2. Train a Binary Classifier to Predict Legendary Status**
- Build a classification model to determine whether a Pokémon is legendary based on its stats. Take care of the data imbalance as most of the data has legendary as False.
- Experiment with different models to find the most effective one
- Evaluate models using **accuracy and F1-score**

### **3. Model Evaluation**
- Plot **training accuracy vs. epochs**
- Plot **validation accuracy vs. epochs**
- Generate a **confusion matrix** for the model’s predictions

### **4. Final Analysis**
- Evaluate the trained model on the test dataset
- Compare its predictions with the CAS histogram analysis from Task 3.1



## **Submission Guidelines**

### **Notebook/Scripts**
- Submit your work in a **Jupyter Notebook (.ipynb) or a Python script**
- Ensure that all code is well-structured and outputs (plots, tables, etc.) are included

### **Documentation**
- Add **inline comments** to explain key steps and logic in your code

### **Report**
- Submit a brief report (2-4 pages) summarizing:
  - **Evaluation Reports** (F1-score, Accuracy, Loss etc.)
  - **Challenges faced**
  - **Insights derived from the data**

### **Deadline**
- **Submit your assignment by Tuesday, March 11**

### **Bonus**
Explore advanced ML models such as XGBoost, Decision Trees and Random Forests. Whoever has a good understanding and is able to implement as well as explain, gets a treat ;)

---
