import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.express as px

# Load the model
model = load_model('lstm_model_final.h5')

# Load and preprocess the data
data = pd.read_csv('data_month.csv', parse_dates=['Date'], index_col='Date')
order_demand = data['Order_Demand'].values.reshape(-1, 1)  # Reshape into 2D array

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(order_demand)
seq_length = 12

# Function to predict next month's order demand
def predict_next_month(data, model, scaler, seq_length):
    last_sequence = data[-seq_length:]  # Last sequence from the data
    last_sequence = last_sequence.reshape((1, seq_length, 1))  # Reshape for prediction
    next_month_scaled = model.predict(last_sequence)
    next_month = scaler.inverse_transform(next_month_scaled)
    return next_month[0][0]

# def visualize_result(predictions):
#     fig = px.line(predictions, x="Date", y='Order Demand', labels={'x': 'Date', 'y': 'Order Demand'},
#                   title='Forecasted Order Demand Over Time')
#     fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
#     fig.update_layout(
#     plot_bgcolor='rgba(0,0,0,0)',
#     paper_bgcolor='rgba(0,0,0,0)',
#     font_color='black')
#     fig.update_xaxes(showgrid=False)
#     fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')
        
#     st.plotly_chart(fig)
# Streamlit app
def main():
    st.set_page_config(page_title='Demand forecasting')

    # App name
    st.title('Monthly Order Demand Forecasting')
    future_months = st.number_input('Enter the number of months to forecast', min_value=1, value=6, step=1)

    if st.button('Predict'):
        predictions = []
        current_data = scaled_data
        

        for _ in range(future_months):
            prediction = predict_next_month(current_data, model, scaler, seq_length)
            predictions.append(prediction)
            current_data = np.concatenate((current_data[1:], prediction.reshape(-1, 1)))

        # Print the predicted order demand for the future months
        next_months_dates = pd.date_range(start=data.index[-1], periods=future_months + 1, freq='M')[1:]
        predicted_demand = pd.DataFrame(np.round(predictions), index=next_months_dates, columns=['Order Demand'])

        predicted_demand.reset_index(inplace=True)
        predicted_demand.rename(columns={'index': 'Date'}, inplace=True)
        predicted_demand["Date"] = pd.to_datetime(predicted_demand["Date"]).dt.strftime('%b %Y')
        
        st.subheader('Forecasted values:')
        st.dataframe(predicted_demand)
        
        fig = px.line(predicted_demand, x="Date", y='Order Demand', labels={'x': 'Date', 'y': 'Order Demand'},
                  title='Forecasted Order Demand Over Time')
        fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='black')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')
        
        st.plotly_chart(fig)
        
        
        # st.button("Visualize Result", on_click=visualize_result, args=(predicted_demand,))
        
    

    about_button = st.button("About")
    if about_button:  
        st.header("About")
        st.markdown("""
            This app predicts the future month-wise order demand using an LSTM model.
            Here's the process involved in developing this app:
            
            1. Data Preparation:
                - Loaded the historical order demand data.
                - Cleaned the data by handling missing values and outliers.
            
            2. Feature Engineering:
                - Extracted relevant features
                - Normalized or scaled the features if required.
            
            3. Train-Test Split:
                - Splitted the data into training and testing sets.
            
            4. Model Development:
                - Build and train an LSTM model using the training data.
            
            5. Model Evaluation:
                - Evaluate the model's performance using appropriate metrics.
            
            6. Model Deployment:
                - Deployed the trained model using Streamlit.
                - Allow users to input new data and generate predictions.
            
            That's the overview of how this demand forecasting App works! Enter the number of months to forecast, click the "Predict" button, and explore the forecasted values.
            The forecasted order demand is visualized.
            
            
            This app is developed by Gangadhar.
        """)
        
    st.markdown('---')
    st.markdown('Developed by Gangadhar')

if __name__ == '__main__':
    main()