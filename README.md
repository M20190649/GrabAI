# GrabAI

Grab organised the Grab AI for SEA Challenges in 2019, hosting 3 different AI challenges.
This is my submission for the first challenge - Traffic Management, which I completed over the past 3 weeks, spending my weekday nights after work and weekends on this project.

I have included the final end-to-end jupyter notebook, initial notebooks for preprocessing, EDA, model creation, 
as well as short write ups for each sections, in case anyone is interested.

https://www.aiforsea.com/

# Problem Statement: Traffic Management 

Economies in Southeast Asia are turning to AI to solve traffic congestion, which hinders mobility and economic growth. The first step in the push towards alleviating traffic congestion is to understand travel demand and travel patterns within the city.

Can we accurately forecast travel demand based on historical Grab bookings to predict areas and times with high travel demand?

In this challenge, participants are to build a model trained on a historical demand dataset, that can forecast demand on a Hold-out test dataset. The model should be able to accurately forecast ahead by T+1 to T+5 time intervals (where each interval is 15-min) given all data up to time T.

# Instructions for Execution

For Grab officials evaluating my submission, please:
  1. Setup virtualenv with required libraries beforehand using 'requirements.txt'.
  2. Place the train set (named 'training.csv') and holdout set (named 'holdout.csv'), into the folder named 'data_raw'. </br>
  Holdout set should be in the same format as train set and should have the demand values filled in (don't worry, holdout values are only used for labels, not features).
  3. Then, run the 'Full Pipeline' jupyter notebook. This notebook will handle the entire process, from preprocessing, feature generation to predicting using the model. (It may take 10-20mins for the entire process).
  
At the end of the notebook execution, the overall holdout rmse will be displayed, and the predicted T+1 to T+5 values are saved as a csv file for inspection.

_Note: As I am slightly confused regarding the way the holdout set is handled. There is likely not enough time to ask and modify my code as I am working on weekdays. I opted to assume that the holdout set contains only the T+1 -> T+5 labels. Thus, my script will read the holdout set and predict for every sample in the holdout set, saving predictions as a .csv file._ 

# Evaluation results

I have tested the model I have built, using the last 7 days in the train set as a holdout set, the day before that as a validation set, and training on the dates before that. 

This has achieved validation rmse: 0.0101 and holdout rmse: 0.0317

<p align="center">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/rmse_val.png" alt="RMSE scores">
</p>

# Summary of approach

Detailed writeups for each step (preprocessing, EDA, model building) has been added to the respective subdirectory. This section is to provide an intuition for the approach I have built.

First, I concat the training set and holdout set, and transform them such that the columns are now the timestamps, while each row refers to 1 of the 1329 locations given.
This format allows me to efficiently extract values for feature engineering and the T+1 -> T+5 labels in bulk.

<p align="center">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/df_train_to_transformed.png" alt="Transform train set to efficient format">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/extraction_feature_label.png" alt="Efficient features and labels extraction" width="650" height="270">
</p>

My key features used are: T and past 30 values, Moving averages of past values etc. (See "LGB Pipeline" page for more).
As the T value is the best feature by far, I opted to use a rolling forecast model, where we include values up to T to predict T+1,
up to T+1 t predict T+2 and so on. This approach further boosted the holdout rmse.

<p align="center">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/extraction_rolling_forecast.png" alt="Feature and label extraction in Rolling forecast model" width="650" height="270">
</p>

The prediction results are as follows:

<p align="center">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/rmse_val.png" alt="RMSE scores">
</p>

# Further Improvements:

Initially, I also used aggregated features by grouping the dataset according the cluster ids/lat/long I had previously generated during Geo_EDA,
which improved rmse by about 0.005 - 0.01. However, it worsened computation time significantly, so I have removed it for this competition.

Also, I built a basic lstm model, using the transformed dataset. It also achieved an rmse of about 0.032, possibly validating this LGBM model.
Given more time, perhaps I could have built an ensemble regressor which should outperform individual regressors.
