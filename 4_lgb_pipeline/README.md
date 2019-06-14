# LightGBM Pipeline

Regression problems require either regression models (e.g. linear regression), random forests regressors, 
or neural networks. Due to prior experience in using LightGBM to forecast demand, I chose to use LightGBM again for this project.
One key advantage is the high speed in building the model (takes a minute or two for large amounts of data),
allowing us to prototype quickly throughout this 3-week competition. Another advantage is the compeitive accuracy
of this algorithm compared to other boosting random forests, and even neural networks!

I have attempted to use the same feature set in a simple feed-forward network, which yielded much poorer results
compared to LGBM even after tuning. I also built a LSTM model for comparison, which yielded simialr holdout rmse (0.033),
however, due to my amatuer knowledge in LSTM and lack of time, I have chose not to continue with this approach.
Given more time, perhaps I could have built an ensemble model which generally has even better performance than a single model.

# Features

Features we used are:
1. Past 30 demand values (T -> T-30).
2. Past 30 demand values, 1 week prior (T -> T-30, 1 week prior).
3. Next 30 demand values, 1 week prior (T+1 -> T+31, 1 week prior).
4. Common statistics such as moving average, diff, standard deviation etc.
for the past few demand values
5. Time-based features such as day, day of week (DoW), minute, hour etc.
6. Cluster ids based on location, generated in "Geo_EDA.ipynb"

The best feature (by far) is the T demand value. Other good features are the common statistics features.

We have also tried many, many other features, such as daily average and weekly average, but they showed negligible
improvments to the rmse score, probably due to high correlation with the above features.

# Key performance boost

One key trend we noticed was that the rmse will increase by about 0.05 as we go from T+1 -> T+2 and so on.
This is a rather substantial increase, as T+5 would be around 0.05 rmse. Considering that the T demand value
is the most important feature by far, I opted to use a rolling forecast approach, where we will generate separate
feature sets for each set of predictions T+1, T+2 etc. instead of using the same feature set which only includes 
information up to T. Using the rolling forecast approach, we cna include T+1 in the feature set for predicting T+2,
T+2 for predicting T+3 and so on. This negates the rise in rmse, resulting in a stable rmse of 0.033 for all prediction sets.

# Metrics

Validation rmse on T+1 for all locations ranges from 0.01 to 0.04. This variance depends on T, the time of prediction,
due to the high peak at noon resulting in larger rmse, while the dips have lower rmse.

We used a week of samples with various T values, to create a holdout set. I believe this represents a good distribution
of samples for the whole week, resulting in a holdout rmse that better represents the average error.
Holdout rmse: 0.033

# Further improvements

Initially, I applied groupby on the dataset, aggregating the dataset based on cluster ids previously generated in "Geo_EDA.ipynb".
I found that aggregating the dataset by latitude/longitude, and then adding the set of aggregated features, same as the existing features,
tends to reduce rmse by 0.005 to 0.01. However, this results in much higher computation time and complexity, thus I dropped this approach
for the competition.

Analysis on rmse range: