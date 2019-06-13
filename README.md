# GrabAI
Grab organised the Grab AI for SEA Challenges in 2019, hosting 3 different AI challenges.
This is my submission for the first challenge - Traffic Management.
I have included the final end-to-end jupyter notebook, initial notebooks for preprocessing, EDA, model creation, 
as well as short write ups for each sections, in case anyone is interested.

https://www.aiforsea.com/

# Traffic Management: Problem Statement

Economies in Southeast Asia are turning to AI to solve traffic congestion, which hinders mobility and economic growth. The first step in the push towards alleviating traffic congestion is to understand travel demand and travel patterns within the city.

Can we accurately forecast travel demand based on historical Grab bookings to predict areas and times with high travel demand?

In this challenge, participants are to build a model trained on a historical demand dataset, that can forecast demand on a Hold-out test dataset. The model should be able to accurately forecast ahead by T+1 to T+5 time intervals (where each interval is 15-min) given all data up to time T.

# Instructions for Execution

For Grab officials evaluating my submission, please:
  1. Setup virtualenv with required libraries beforehand using 'requirements.txt'.
  2. Place the train set (named 'training.csv') and holdout set (named 'holdout.csv'), into the folder named 'data_raw'.
  3, Then, run the 'Full Pipeline' jupyter notebook. This notebook will handle the entire process, from preprocessing and feature       generation to demand prediction using the model I have built. (It may take 5-15mins for the entire process).
  
At the end of the notebook execution, the overall holdout rmse will be displayed, and the predicted T+1 to T+5 values are saved as a csv file for inspection.

# Evaluation results

I have tested the model I have built, using the last 7 days in the train set as a holdout set, the day before that as a validation set, and training on the dates before that. 

This has achieved validation rmse: X and holdout rmse: Y.
