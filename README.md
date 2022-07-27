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
![Model pyt](https://github.com/noorhanifah/Predicting_Covid19_Cases/blob/main/Model_arch1.PNG)

![Model](https://github.com/noorhanifah/Predicting_Covid19_Cases/blob/main/Model_arch.png)

## Accuracy 

###### Below shows the training model evalution..
![Training model evaluation](https://github.com/noorhanifah/Predicting_Covid19_Cases/blob/main/model_evaluation.PNG)

###### The lowest MAPE (mean absolute percentage error) that this model can achieve is 9.3%. Since MAPE shows how far the model’s predictions are off from their corresponding outputs, the lowest the MAPE the closest the model to the test data. Therefore, this model has the ability to forcast the new cases based only the 30 days data. 
![MAPE](https://github.com/noorhanifah/Predicting_Covid19_Cases/blob/main/calculated_mape.PNG)

###### Below shows a plotted graph showing the actual versus the predicted trend. From this graph, we can see that the blue line (predicted) and the red line (actual) shows a similar trend of up and down. 
![Training vs actual](https://github.com/noorhanifah/Predicting_Covid19_Cases/blob/main/predicted_vs_actual.png)

###### TensorBoard also is used to plot the all the training process of this model. 
![TensorBoard](https://github.com/noorhanifah/Predicting_Covid19_Cases/blob/main/Tensorboard/Tensorboard.PNG)

###### All the MAPE training process of this model:
![TensorBoard MAPE](https://github.com/noorhanifah/Predicting_Covid19_Cases/blob/main/Tensorboard/MAPE.PNG)

###### The Tensorboard graph shows the MAPE at each epoch:
![TensorBoard Current MAPE](https://github.com/noorhanifah/Predicting_Covid19_Cases/blob/main/Tensorboard/mape_current_training.PNG)

## Build With
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![GoogleColab](	https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
 ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Credit
The dataset is accessible from Github at GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera (https://github.com/MoH-Malaysia/covid19-public/blob/main/epidemic/cases_malaysia.csv).

Thank you, Ministry of Health Malaysia, (https://github.com/MoH-Malaysia) for making the dataset to be publicly accessible. It gives me the opportunity to learn and practice more on machine learning and deep learning.  

