import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from streamlit_ace import st_ace

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
        page_title='Demand forecasting',
        layout='wide',
        initial_sidebar_state='expanded',
        page_icon=":chart_with_upwards_trend:"
    )

    # Custom CSS style
    page_bg = '''
    <style>
    body {
        background-color: #f0f2f5;
    }
    .stButton button {
        background-color: #F63366;
        color: white;
        border-color: #F63366;
    }
    .stButton button:hover {
        background-color: #ED125B;
        border-color: #ED125B;
    }
    .stDataFrame {
        background-color: white;
        border: 1px solid #D3D3D3;
        border-radius: 5px;
        padding: 10px;
    }
    .animated {
        animation-duration: 1s;
        animation-fill-mode: both;
    }
    .fade-in {
        animation-name: fadeIn;
    }
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    </style>
    '''
    st.markdown(page_bg, unsafe_allow_html=True)

    # App title and description
    st.title('Monthly Order Demand Forecasting')
    st.markdown("Forecast future order demand using LSTM model")

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

        # Display forecasted values with fade-in animation
        st.subheader('Forecasted values:')
        predicted_demand_html = predicted_demand.style.set_properties(**{'text-align': 'center'}).render()
        st.markdown(
            f'<div class="animated fade-in">{predicted_demand_html}</div>',
            unsafe_allow_html=True
        )

        # Plot forecasted order demand over time
        fig = go.Figure(data=go.Scatter(x=predicted_demand['Date'], y=predicted_demand['Order Demand'], mode='lines',
                                        name='Forecasted Demand', line=dict(color='#F63366', width=2)))
        fig.update_layout(
            title='Forecasted Order Demand Over Time',
            xaxis_title='Date',
            yaxis_title='Order Demand',
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='black'
        )
        st.plotly_chart(fig)

    # About section
    st.markdown("---")
    st.subheader("About")
    st.markdown("The purpose of this app is to forecast the monthly order demand using an LSTM model.")
    st.markdown("It addresses order shortage issues related to ocean shipping, where orders can take months or weeks to arrive.")
    st.markdown("To use the app, enter the number of months to forecast in the sidebar and click the 'Predict' button.")
    st.markdown("The app will generate forecasted values and display graphs for visualization.")

    # Footer
    st.markdown("---")
    st.subheader("Developed by Gangadhar")
    st.markdown("Check out the [GitHub repository](https://github.com/GangadharNeelam) for the source code.")

    # Code highlighting with streamlit_ace
    st.subheader("Source Code")
    st_ace(
        value=open(__file__, 'r').read(),
        language="python",
        theme="github",
        keybinding="vscode",
        show_gutter=False,
        font_size=14,
        use_wrap_mode=True,
        wrap=True
    )

if __name__ == '__main__':
    main()