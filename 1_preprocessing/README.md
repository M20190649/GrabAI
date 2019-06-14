# Data cleansing

Other than doing some data cleansing, I did some preliminary EDA to get an intuition for the given dataset,
from which I crafted my approach. 

Here are some of our findings:

1. The dataset is missing roughly half the demand values, considering the whole time period and all locations.
I came up with a method to fill up the missing data, which is recorded in '3_fill_missing'.

Update: I received a notification on the updated FAQ from Grab, in which they instructed participants to simply fill the
missing values with 0. I have updated the final script "Full_Pipeline.ipynb" accordingly. However, the original scripts found 
in the subdirectories will maintain the approach of generating values for the missing data for reference's sake.
(Incidentally, there is hardly any change in the holdout rmse in both approaches, which is rather curious).

2. The (normalized) dataset is rather skewed, with majority of samples closer to 0, whereas a minority values lean to 1. 
Thus, the mean is much higher than the median. We may have to account for the skew later on, such as by partitioning the dataset
into groups based on demand values.

# Data Preprocessing

As mentioned, we transform the dataset such that the columns are the datetimes are the columns, while the geolocations
are the rows, and each cell is the demand for the time bucket at the specific location. This allows us to quickly
retrieve ranges of demand values for specific time frames to craft our features and labels.

Input: training.csv
Output: training_transformed.csv