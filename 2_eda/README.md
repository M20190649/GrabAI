For more in depth EDA, I approached the issue from two perspectives:
* Demand vs Geography, and 
* Gemand vs Time (graph)

# Geo EDA

_Input: training.csv </br>
Output: cluster_df.csv_

The locations given are in the form of a grid, on a anonymised region. I plotted the total demand vs location on a map, shown below.

<p align="center">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/demand_total.png" alt="Total Demand vs Location plotted" width="490" height="350">
</p>

On the assumption that demand in any location is likely to be influenced by demand in its neighboring locations,
I executed some basic clustering schemes on the locations based on demand and coordinates, such as grouping by latitude/longitude and DBScan.
The cluster ids for the locations, for each clustering schemes, are recorded in cluster_df.csv.
These can be used as additional features, or used to aggregate the dataset to generate aggregated features.

![Examples of clustering schemes](../images/clustering.png?raw=true "Examples of clustering schemes")

# Graph EDA

_Input: training.csv </br>
No output files._

We plotted the demand vs time per location, which gave us quite a few insights
based on time series analysis:

1. We can separate the locations into 3 zones, illustrated in the image below:
* a. Locations that are only missing a few values in the whole time period (e.g. 10 or so)
* b. Locations that are missing perhaps half their values
* c. Locations that are missing most of their values (95%+). </br>
 
_(Due to the different traits, I have attempted to separate the locations into 3 groups for modelling and predicting, but the overall rmse is similar to when I did not separate the locations. Thus, I gave up on this approach)._

![Examples of locations with different quantites of NaN values](../images/graph_nan.png?raw=true "Examples of locations with different quantites of NaN values")

2. Plotting total demand against time, we observe a trend where there is a dip in the middle of the time period, and then an upward climb. Also, at a glance, we observe a repeating weekly pattern/seasonality.

<p align="center">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/graph_trend.png" alt="General Trend" width="400" height="200">
</p>

3. There is a somewhat strong daily seasonality - every day, the demand curve for each day has the same general shape.
The demand climbs gradually and peaks at about 7am until 2pm, falling to a low at 7pm before picking up again. 

![Daily Seasonality](../images/graph_daily.png?raw=true "Daily Seasonality")

4. There is a strong weekly seasonality - assuming the first date to be Monday, the Monday following it is very similar compared to other days of the week,
same for Tuesday and so on.

![Weekly Seasonality](../images/graph_weekly.png?raw=true "Weekly Seasonality")

5. We also plotted heatmaps based on Day of Week (DoW) and based on hourly period, to see if demand is higher on certain days/hours compared to others. 
Each heatmap row represents a location, and the columns represent the day/hour. </br>
We can see that Day 4 and 5 (out of Day 0 to Day 6) generally have higher demands than other days, thus possibly being Saturday and Sunday.
Also, there is higher demand in the first half of the day (12am to 3pm) than the second half (3pm to 12am).

<p align="center">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/heatmap_dow.png" alt="Heatmap by day of week" width="770" height="420">
	<img src="https://raw.githubusercontent.com/ThunderXBlitZ/GrabAI/master/images/heatmap_hourly.png" alt="Heatmap by hours" width="770" height="150">
</p>