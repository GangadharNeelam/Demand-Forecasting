import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
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
    st.set_page_config(
        page_title='Demand Forecasting',
        layout='wide',
        initial_sidebar_state='collapsed',
        page_icon=":chart_with_upwards_trend:"
    )

    # Custom CSS style
    page_bg = '''
    <style>
    body {
        background-color: #f0f2f5;
    }
    .stTitle {
        color: #F63366;
        font-size: 48px;
        margin-bottom: 50px;
    }
    .stText {
        color: #1E88E5;
        font-size: 20px;
        margin-bottom: 30px;
    }
    .stButton button {
        background-color: #F63366;
        color: white;
        border-color: #F63366;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
        margin-top: 20px;
    }
    .stButton button:hover {
        background-color: #ED125B;
        border-color: #ED125B;
    }
    </style>
    '''
    st.markdown(page_bg, unsafe_allow_html=True)

    # App title and description
    st.title('Demand Forecasting')
    st.markdown("Forecast future order demand using LSTM model")
    st.markdown("---")

    # Sidebar
    st.sidebar.subheader("Settings")
    future_months = st.sidebar.number_input('Enter the number of months to forecast', min_value=1, value=6, step=1)

    if st.sidebar.button('Predict'):
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

        # Display forecasted values
        st.subheader('Forecasted values:')
        st.dataframe(predicted_demand.style.set_properties(**{'text-align': 'center'}))

        # Plot forecasted order demand over time
        fig = go.Figure(data=go.Scatter(x=predicted_demand['Date'], y=predicted_demand['Order Demand'], mode='lines',
                                        name='Forecasted Demand', line=dict(color='#F63366', width=2)))
        fig.update_layout(
            title='Forecasted Order Demand Over Time',
            xaxis_title='Date',
            yaxis_title='Order Demand',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='#f0f2f5',
            font_color='black',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig)

    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.markdown("The purpose of this app is to forecast the monthly order demand using an LSTM model.")
    st.sidebar.markdown("It addresses order shortage issues related to ocean shipping, where orders can take months or weeks to arrive.")
    st.sidebar.markdown("To use the app, enter the number of months to forecast in the sidebar and click the 'Predict' button.")
    st.sidebar.markdown("The app will generate forecasted values and display a graph for visualization.")

if __name__ == '__main__':
    main()
