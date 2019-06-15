# Data cleansing

Input: training.csv<br/>
Output: training_transformed.csv

Other than doing some data cleansing, I did some preliminary scrubbing/cleansing to get an intuition for the given dataset, from which I crafted my approach. 

Here are some of our findings:

1. There is a total of 4.2 million records, with demand values are normalized between 0 and 1 and no null values. <br/>
There are 1329 locations, 61 days/5856 timestamps (15min intervals). <br/> Datatypes and string formats are all correct, no duplicate timestamps etc.

2. The dataset should contain 1329 locations * 5856 timestamps demand values. <br/> However, it is missing roughly half the number of demand values. <br/> Furthermore, some timestamps have
no demand values for all locations. 

I came up with a method to fill up the missing data, which is recorded in '3_fill_missing'.

_Update: I received a notification on the updated FAQ from Grab, in which they instructed participants to simply fill the missing values with 0. I have updated the final script "Full_Pipeline.ipynb" accordingly. However, the original scripts found in the subdirectories will maintain the approach of generating values for the missing data for reference's sake. (Incidentally, there is hardly any change in the holdout rmse in both approaches, which is rather curious)._

3. The dataset is rather skewed, with majority of demand values closer to 0, with a trailing tail after demand = 0.25, towards 1. Thus, the mean (0.105) is much higher than the median (0.0504), instead it is closer to the 75th percentile (0.120) and standard deviation (0.159). We may have to account for the skew later on, such as by partitioning the dataset into groups based on demand values.

![Box plot of demand values](../images/box_plot.png?raw=true "Box Plot of Demand values"){:height="70%" width="70%"}

# Data Preprocessing

As mentioned, we transform the dataset such that the columns are the datetimes are the columns, while the geolocations are the rows, and each cell is the demand for the time bucket at the specific location. This allows us to efficiently retrieve ranges of demand values for specific time frames to craft our features and labels.

![Transform train set to efficient format](../images/df_train_to_transformed.png?raw=true "Transforming train set"){:height="70%" width="70%"}

![Efficient features and labels extraction](../images/extraction_feature_label.png?raw=true "Efficient features and labels extraction"){:height="70%" width="70%"}