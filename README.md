# SpamClassifier

## Table of Content
  * [Overview](#overview)
  * [Technical Aspect](#technical-aspect)
  * [Technologies Used](#technologies-used)

## Overview
This project is divided into 2 parts, a Spam Classifier 
- Trained with **Naive Bayes classifier**.
- Trained on a **Custom ditil-bert Model** ( Accuracy = 1.00 ).    
![image](https://user-images.githubusercontent.com/76872499/150648250-61ed0d7d-2f5a-4f51-bbc9-91bf36c78881.png)

## Technical Aspect
Dataset: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection  
Given below are the steps taken to build the model:  
For the NLP(Naive Bayes classifier) implementation code –
  - Imported necessary **Python libraries** (Pandas, re, nltk, sklearn, etc.)
  -	Imported the Dataset and **split** it into labels & messages.
  -	In **Data cleaning** & **Text pre-processing** Section, we removed unnecessary symbols & numbers. We lower cased all sentences & split sentences into words.
  -	Performed **stemming** & **removal of stopwords**.
  -	Created the **Bag of Words model** and selected the top 2000 frequent features.
  -	Performed **One Hot Encoding** for labels(y_test).
  -	Performed Train and Test Split.
  -	**Trained model** using **Naive Bayes classifier**.
  -	From sklearn we imported **metrics** to calculate how good our model was working:  
     No. of features Selected    **Accuracy**   
    -         2000     0.9838565022421525
    -         3000     0.9838565022421525
    -         4000     0.9838565022421525
    -  	      5000     0.9847533632286996



## Technologies Used
- Spyder IDE
-	ML model: Naive Bayes classifier (Selected)
-	Libraries: pandas, re, nltk, sk-learn.

