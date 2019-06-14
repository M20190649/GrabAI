For more in depth EDA, I approached the issue from two perspectives: demand vs geography and demand vs time (graph).

# Geo EDA

The locations given are in the form of a grid, on a anonymised region. I plotted the total demand vs location on a map, shown below:

On the assumption that demand in any location is likely to be influenced by demand in its neighboring locations,
I executed some basic clustering schemes on the locations based on demand and coordinates, such as grouping by latitude/longitude and DBScan.
The cluster ids for the locations, for each clustering schemes, are recorded in cluster_df.csv.
These can be used as additional features, or used to aggregate the dataset to generate aggregated features.

Input: training.csv
Output: cluster_df.csv

# Graph EDA

We plotted the demand vs time per location, which gave us quite a few insights:

1. We can separate the locations into 3 zones, illustrated in the image below:
	a. Locations that are only missing a few values in the whole time period (e.g. 10 or so)
	b. Locations that are missing perhaps half their values
	c. Locations that are missing most of their values (only have 10 or so demand values in the whole time period).

(Due to the different traits, I have attempted to separate the locations into 3 groups for modelling and predicting,
but the overall rmse is similar to when I did not separate the locations. Thus, I gave up on this approach).

2. There is strong daily seasonality - every day, the demand curve for each day has the same general shape.
The demand peak at about 12pm ....

3. There is strong weekly seasonality - assuming the first date to be Monday, the Monday following it is very similar compared to other days of the week,
same for Tuesday and so on.

4. We also plotted heatmaps based on Day of Week (DoW) and based on hourly period, to see if demand is higher on certain days/hours
compared to others.

Input: training.csv
No output files.