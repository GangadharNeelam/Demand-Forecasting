# Demand forecasting

https://github.com/GangadharNeelam/Demand_forecasting/assets/93145713/60ad4666-ab3a-400f-b823-74133cb6352f

This Demand Forecasting App uses a SARIMA (Seasonal AutoRegressive Integrated Moving Average) model to predict monthly Orders Demand based on historical data.
Here's the process involved in developing this app:

1. Data Preprocessing:
    - Load the data from a CSV file.
    - Performed necessary preprocessing steps such as handling missing values and converting the date column to a datetime format.

2. Model Training:
    - Trained SARIMA model using the preprocessed data.
    - The SARIMA model captures the seasonal and trend patterns in the data to make accurate predictions.

3. Model Deployment:
    - Saved the trained SARIMA model using the pickle library.
    - The saved model is then loaded.

 4. Forecasting:
    - The user can input the number of months to forecast in the app.
    - Based on this input, the app uses the trained SARIMA model to generate predictions for the specified number of months.

5. Results Visualization:
    - The predicted values are displayed in a tabular format.
    - Additionally, a line chart is plotted to visualize the forecasted order demand trends over time.

That's the overview of how this demand forecasting App works!

Data set : https://www.kaggle.com/datasets/felixzhao/productdemandforecasting
