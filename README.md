# Data Profiling, Preparation, Classification and Explanation of a Dataset

This project aims to create a machine learning model around a health realated database, and then explaning the predictions it obtains through two diferent explanability methods, LIME (Local Intrepertable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations).  
_The models will be created using the python libraries **scikit-learn**, **matplotlib** and **imblearn**_

## Folder Organization

As it stands the project folders are **PIC** (the main folder), **Csvs** (folder with the transformed data bases) and **Images** (folder with the images and graphs related to the data transformation procesess).  
In the main folder there are four _jupyter notebooks_: 
 - **DataProfiling** - Has the code that outputs not only the graphic representation of the data base and correspondent variables, but also _missing values_ and _outliers_.
 - **DataPreparation** - Has the code that prepares the data base for the classification step. In this step there were tested four transformations (**Dummification**, **Scaling**, **Balancing**, **Discretization**).
 - **DataClassification** - Has the separation of the dataset in _train_ and _test_ (70/30), and then the application of diferent classification models, such as **Naive Bayes**, **KNN**, **Decision Trees**, **Random Forest**, **Neural Networks** and **Gradient Boosting**. 
 - **Prediction** - Has the functions created to transform the _input_ so that it can be correctly evaluated by the machine learning model and the correspondent class prediction of the transformed _input_ data.

 ## Results

The initial evaluation of the dada preparation techniques were done by evalutating the variation of the accuracy of a **Neural Network** model.
#### Missing value Imputation:

|  Techniques | Deleting Lines | Na -> 0 | Na -> Most Frequent |  
|-------------|----------------|---------|---------------------|
| Accuracy    | 0.714          | 0.675   | 0.675               |  

#### Scaling:

|  Techniques | Minmax | z-score |
|-------------|--------|---------|
| Accuracy    | 0.771  | 0.771   |

#### Balancing:

| Techniques | UnderSampling | OverSampling | SMOTE |
|-----------|---------------|--------------|-------|
| Accuracy  | 0.667         | 0.796        | 0.760 |

#### Discretization: 

| Techniques | Equal-frequency | Equal-width|
|-----------|---------------|--------------|
| Accuracy  | 0.771         | 0.742        |

#### Classification

| Techniques | NaiveBayes    | KNN|Random Forest|Neural Networks|Gradient Boosting       |NeuralNetworks - BalancedBaggingClassifier|
|----------- |---------------|--------------|--------------|--------------|--------------|--------------|
| Accuracy   | 0.69          | 0.82          |0.90         |0.67          |0.77         |0.69|
