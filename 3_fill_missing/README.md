# Data impuration/Filling in missing values

_Input: training_transformed.csv </br>
Output: training_transformed_filled.csv (no NaN values)_

_(This is now deprecated due to updates in the competition FAQ instructing participants to fill by zero instead.
The final script for submission ("Full_Pipeline.ipynb") has been updated accordingly.
However, the original scripts in the subdirectories will continue to use this approach to fill up the missing data,
for reference's sake)._

I used a simple approach to fill up the missing values. 

* 1. Due to the strong weekly seasonality/pattern I first fill as much missing values as possible using a seasonality-based approach (which filled 1/3 of the missing values).
** a. I separated the dataset by week, and averaged the data to obtain a weekly pattern.
** b. I obtained the weekly ratio of each week's average to the first week's average.
** c. I filled in the missing values for each week, by multiplying the week's ratio (b) on the weekly pattern (a).
_Due to Pandas' in-built way of avoiding NaNs in calculations, values for which we have no weekly-based data are ignored, while those that do are filled in._

![Weekly Seasonality](../images/graph_weekly.png?raw=true "Weekly Seasonality")
	
* 2. To fill in the remaining values, I used an average of bbfill, ffill and interpolation (provided by Pandas).

* 3. I also added some noise to the generated values to simulate a real-life scenario.

![Graph with all NaN values filled, and additional noise](../images/graph_noise.png?raw=true "Graph with all NaN values filled, and additional noise")

The output is a filled training set, for use in modelling and predictions.