# WOWDataAnalysis

In the following project the dataset from https://data.world/helithumper/prices-of-world-of-warcrafttoken/workspace/file?filename=wowcointotal.csv 
was used. It contained the price of the monthly subscription measured at various days and times since 2015.

For the specific dataset the following analysis was performed:
1) Fit a model that predicts the price of the monthly subscription as a function of time.
2) Every time World of Warcraft releases an expansion, there is usually an inflation in the economy as monsters, quests 
  and other in-game activities reward more gold and the supply of gold goes up. Based on this fact, and by looking at
  the data, roughly estimate when an expansion was released during this time period.
3) Split the data into two time periods, one before the expansion and one after and fit two separate models that predict the price of the monthly subscriptions as a
   function of time.
