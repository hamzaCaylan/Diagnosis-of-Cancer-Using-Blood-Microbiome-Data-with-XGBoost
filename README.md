Midterm Project 

Project Title: Diagnosis of Cancer Using Blood Microbiome Data 

Students:  1030510638 Hamza ÇAYLAN 

 

Data Set : https://drive.google.com/file/d/15evTOZTYuopoBnolYWOPy2P_VF6wnlFm/view?usp=sharing 

Source Code: https://github.com/hamzaCaylan/Diagnosis-of-Cancer-Using-Blood-Microbiome-Data-with-XGBoost 

 

1. Introduction 

This study explores the potential of using blood microbiome data to diagnose different types of cancer through machine learning techniques. The main objective is to develop classifiers capable of distinguishing four types of cancer—colon, breast, lung, and prostate—from other conditions. Random Forest (RF) and Gradient Boosted Trees (XGBoost) models will be employed for this task. 

2. Dataset 

Samples: Data collected from a total of 355 individuals, each providing a unique biological profile. 

Features: The dataset includes DNA fragment counts across 1,836 distinct microorganism types, capturing microbiome variations linked to cancer. 

Label File: Contains the corresponding cancer type for each individual sample, used for supervised learning in model training. 

Target Classes: The cancer types of interest include Colon Cancer, Breast Cancer, Lung Cancer, and Prostate Cancer, each serving as a distinct classification category. 

 
 

3. Methods 

3.1 Data Preprocessing 

- Label Cleaning: Corrected for typos and normalized case format. 
 - Feature Normalization: Converted raw counts to relative abundances per sample. 

3.2 Model Training 

- Cross-validation: 5-fold StratifiedKFold 
Algorithms Used: 

Random Forest: The model utilized 500 trees with balanced class weights to handle any imbalances in the dataset, ensuring fair representation of each class during training. 

XGBoost: The XGBoost model was set up with 400 estimators, a learning rate of 0.05, and a maximum depth of 6, chosen to balance model complexity and prevent overfitting.  
Each cancer type was treated as a binary classification task: cancer vs. other types. 

3.3 Evaluation Metrics 

- Sensitivity (Recall): True Positive Rate 
 - Specificity: True Negative Rate 

4. Results 

Cancer Type 

RF Sensitivity 

RF Specificity 

XGB Sensitivity 

XGB Specificity 

Colon Cancer 

0.935 

1.000 

0.953 

0.996 

Lung Cancer 

0.850 

1.000 

0.733 

1.000 

Breast Cancer 

0.877 

1.000 

0.971 

1.000 

Prostate Cancer 

0.942 

0.987 

0.975 

0.991 

 

6. Conclusion 

This project demonstrated that machine learning models trained on blood microbiome data can accurately classify cancer types. Both Random Forest and XGBoost provide strong results, with XGBoost excelling in sensitivity for most cancer types.  

 
 

 
