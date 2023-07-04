import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

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

# Streamlit app
def main():
    st.set_page_config(page_title='Demand Forecasting App', page_icon=':chart_with_upwards_trend:')

    # App name and description
    st.title('Monthly Order Demand Forecasting')
    st.markdown("Welcome to the Monthly Order Demand Forecasting app! "
                "This app predicts the future order demand based on historical data.")

    # Sidebar section
    st.sidebar.title("Settings")
    future_months = st.sidebar.number_input('Enter the number of months to forecast', min_value=1, value=6, step=1)

    # Prediction section
    if st.sidebar.button('Predict', key='predict_button'):
        predictions = []
        current_data = scaled_data

        for _ in range(future_months):
            prediction = predict_next_month(current_data, model, scaler, seq_length)
            predictions.append(prediction)
            current_data = np.concatenate((current_data[1:], prediction.reshape(-1, 1)))

        # Display the predicted order demand for the future months
        next_months_dates = pd.date_range(start=data.index[-1], periods=future_months + 1, freq='M')[1:]
        predicted_demand = pd.DataFrame(np.round(predictions), index=next_months_dates, columns=['Order Demand'])

        st.subheader('Forecasted Values')
        st.dataframe(predicted_demand)

        # Plot the forecasted order demand over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predicted_demand.index, y=predicted_demand['Order Demand'],
                                 mode='lines+markers', name='Forecasted Demand',
                                 marker=dict(color='royalblue', size=8, line=dict(width=1, color='DarkSlateGrey'))))
        fig.update_layout(title='Forecasted Order Demand Over Time', xaxis_title='Date', yaxis_title='Order Demand')
        st.plotly_chart(fig)

        # Visual elements
        st.subheader('Forecasting Visualizations')
        st.markdown('Check out these visualizations related to forecasting and time series:')

        # Visualization sketch
        st.image('forecasting_sketch.jpg', use_column_width=True)

        # Additional visualizations
        st.markdown('### Seasonal Decomposition')
        st.image('seasonal_decomposition.png', use_column_width=True)

        st.markdown('### Autocorrelation Plot')
        st.image('autocorrelation_plot.png', use_column_width=True)

    # About section
    about_button = st.sidebar.button("About", key='about_button')
    if about_button:
        st.header("About")
        st.markdown("### Purpose")
        st.markdown("The purpose of the app is to forecast the monthly order demand.")
        st.markdown("### Addressing Order Shortage")
        st.markdown("The app aims to solve order shortage issues related to ocean shipping, where orders can take months or weeks to ship.")
        st.markdown("### Forecasting Process")
        st.markdown("To generate forecasts, simply enter the number of months to forecast. The app will predict the order demand and provide a graph for visualization.")
        st.markdown("### LSTM Model")
        st.markdown("The app is powered by an LSTM (Long Short-Term Memory) model, which is a type of recurrent neural network (RNN) commonly used for sequence prediction tasks.")
        st.markdown("### Dataset")
        st.markdown("The app utilizes a publicly available dataset to train the LSTM model and generate accurate predictions.")
        st.markdown("### Target Audience")
        st.markdown("The app is designed for anyone who wants to gain a basic understanding of forecasting and its applications in order management and supply chain.")
        st.markdown("### Future Enhancements")
        st.markdown("In the future, we plan to expand the model's capabilities by incorporating larger datasets and advanced techniques to improve accuracy and efficiency.")

    # Footer
    st.markdown('---')
    st.markdown('Developed by Gangadhar')

if __name__ == '__main__':
    main()