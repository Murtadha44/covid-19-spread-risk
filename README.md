# covid-19-spread-risk
## Using an LSTM network to forecast COVID-19 cases in the next two weeks at the county level

  We used the daily mobility data aggregated at the county level beside COVID-19 statistics and demographic information for short-term forecasting of COVID-19 outbreak in the United States. The daily data are fed to a deep model based on Long Short-Term Memory (LSTM) to predict the accumulated number of COVID-19 cases in the next two weeks. For more details about the data used and the method please read our paper entitled "The Forecast of COVID-19 Spread Risk at The County Level" that can be accessed on https://www.researchsquare.com/article/rs-415377/v1

  To run this code, create a Python environment that contains the following libraries (numpy, pickle, os, random, math, tensorflow, sklearn, scipy, matplotlib), then run main.py. The preprocessed data, trained models and results are saved in main_results folder. 
