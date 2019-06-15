# GrabAI
Grab organised the Grab AI for SEA Challenges in 2019, hosting 3 different AI challenges.
This is my submission for the first challenge - Traffic Management.
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
  3. Then, run the 'Full Pipeline' jupyter notebook. This notebook will handle the entire process, from preprocessing, feature generation to demand prediction using the model. (It may take 10-20mins for the entire process).
  
At the end of the notebook execution, the overall holdout rmse will be displayed, and the predicted T+1 to T+5 values are saved as a csv file for inspection.

# Evaluation results

I have tested the model I have built, using the last 7 days in the train set as a holdout set, the day before that as a validation set, and training on the dates before that. 

This has achieved validation rmse: 0.0101 and holdout rmse: 0.0317

![RMSE scores](./images/rmse_val.png?raw=true "RMSE scores")

# Summary of approach

Detailed writeups for each step (preprocessing, EDA, model building) has been added to the respective subdirectory. This section is to provide an intuition for the approach I have built.

First, I concat the training set and holdout set, and transform them such that the columns are now the timestamps, while each row refers to 1 of the 1329 locations given.
This format allows me to efficiently extract values for feature engineering and the T+1 -> T+5 labels in bulk.

![Transform train set to efficient format](./images/df_train_to_transformed.png?raw=true "Transforming train set")

![Efficient features and labels extraction](./images/extraction_feature_label.png?raw=true "Efficient features and labels extraction")

My key features used are: T and past 30 values, Moving averages of past values etc. (See "LGB Pipeline" page for more).
As the T value is the best feature by far, I opted to use a rolling forecast model, where we include values up to T to predict T+1,
up to T+1 t predict T+2 and so on. This approach further boosted the holdout rmse.

![Feature and label extraction in Rolling forecast model](./images/extraction_rolling_forecast.png?raw=true "Feature and label extraction in Rolling forecast model")

As prediction time T is not specified by Grab, and it has mentioned at the briefing in Singapore to expect the holdout set to be similar to the training set given
(I assume this means in distribution of NaN and non-NaN values), I assumed that T would be random and multiple for every location and altered the approach accordingly.
This also gives us a good spread of prediction times similar to a practical scenario.

![Validation vs holdout set](./images/extraction_val_holdout.png?raw=true "Validation vs holdout set")

The prediction results are as follows:

![RMSE scores](./images/rmse_val.png?raw=true "RMSE scores")