# Predicting Covid19 Cases Using Deep Learning
 
## Project Description

The year 2020 was a catastrophic year for humanity. Pneumonia of unknown aetiology was first reported in December 2019., since then, COVID-19 spread to the whole world and became a global pandemic. More than 200 countries were affected due to pandemic and many countries were trying to save precious lives of their people by imposing travel restrictions, quarantines, social distances, event postponements and lockdowns to prevent the spread of the virus. However, due to lackadaisical attitude, efforts attempted by the governments were jeopardised, thus, predisposing to the wide spread of virus and lost of lives.

Over time, the number of patiences are increasing burdening the health care system and the professional in Malaysia.
To overcome this problem, understanding the nature of the pandemic itself and forecasting the trend of Covid-19 cases are the crucial. Reliability of the forecasting trends of the COVID-19 spread can help predict the pandemic outbreak and increase the readiness of the health care system in tackling the pandemic and providing enough medical support. Moreover, accurate forecasting can provide feedback on whether the undertaken policy is effective in alleviating the stress on the healthcare system in this country. It at the same time provide a room for mitigation strategies and policies to be evaluated using the prediction model.   

The objective of this project is to predict new cases of Covid-19 in Malaysia based on the past 30 days of cases.

## Running the Project
This model run using Python programming and the libraries available. The training and validation graph are plotted using Tensorboard. TensorFlow library is used to develop and train this model.

## Project Insight
To achieve the objective of this project, deep learning model using LSTM neural network aprroach is used. LSTMs (long short-term memory) is one of the most and well known subset of recurrent neural networks. It is a type of artificial neural network designed to recognize patterns in sequences of data, such as numerical times series data which use in this project.

###### The detail architecture of this model shown as below:
![Model pyt](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/Model_arch.PNG)

![Model](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/Plot_model_arch.png)

## Accuracy
After cleaning and training the data, this model acheive up to 0.8 accuracy. 

###### Below shows the training model evalution which shows 95& accuracy.
![Training model evaluation](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/model_evaluation.PNG)

###### Based on the classification report this model give 0.95 accuracy with f1-score of more than 0.9. This shows that this model is able to predict the all five outcomes expected from this model. Therefore, the ability to categorize articles into Sport, Tech, Business, Entertainment and Politics can be achieve throught this model.
![Correlation](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/classification_report.PNG)

###### The best model out of all the aproach is Logistic Regression with Standard Scalar as they give a score of 0.824. Thus, will be selected for this project. 
![Best Model](https://github.com/noorhanifah/Heart-Attack-Prediction-Using-Machine-Learning/blob/main/Score/best_model.PNG)

###### Based on the classification report this model give 0.82 accuracy with f1-score of 0.82 and 0.83. This shows that this model is able to predict the two outcome expected from this model. Therefore, to know wheather someone has the possibility of having heart attack or not can be achieve throught this model.
![CR](https://github.com/noorhanifah/Heart-Attack-Prediction-Using-Machine-Learning/blob/main/Score/classification_report.PNG)

###### The training and the validation accuracy of this model can be observed from the plotted graph. From the graph, this model is able to learn at some point.
![Training and validation accuracy](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/validation_training_accuracy.PNG)

###### TensorBoard also is used to plot the all the training graph. 
![TensorBoard](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/Tensorboard/Tensorboard.PNG)

###### The accuracy graph shown on TensorBoard:
![TensorBoard Accuracy](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/Tensorboard/tensorboard_accuracy.PNG)

## A little discussion
This model is able to categorize articles and the training also gives a high accuracy of more than 90% and a high and balance f-1 score among the outcome, thus give a little sense of great achivement. Unfortunately for this model, it actually overfit. Overfitting happens when a model learns the detail and noise in the training data. The noise or random fluctuations in the training data is picked up and learned as concepts by this model making it unable to categorize new data correctly.

One of the way to prevent overfitting is by using early callback. However, this approach does not seems to fix the problem. Therefore, to prevent overfitting for this model, one of the approach that I think could solve the problem is by removing Stopwords from the dataset during the data cleaning step. Stopwords in English such as “a”, “the”, “is”, and “are” carry very little useful information in sentences. By removing these words, the low-level information from the text this model can focus more on important information thus preventing the model from learning too much.

## Build With
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![GoogleColab](	https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)

## Credit
The dataset is accessible from Github at https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv

Thank you, Susan Li (https://github.com/susanli2016) for making the dataset to be publicly accessible. It gives me the opportunity to learn and practice more on machine learning.  

