import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Define function to load data from file
def load_data(file):
    df = pd.read_excel(file)
    
    # Drop non-numeric columns or columns that cannot be converted to float
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    
    # Check if there are any columns with non-numeric values that need to be handled
    non_numeric_columns = df.columns.difference(df_numeric.columns)
    if non_numeric_columns.any():
        st.warning(f"Non-numeric columns detected: {', '.join(non_numeric_columns)}. They will be ignored.")
    
    return df_numeric

# Define function to train model
def train_model(df, target='Total Project Cost'):
    X = df.drop(columns=['Planned Total Project Cost', 'Actual Duration (days)'])  # Drop the target and time columns
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Define function to make predictions
def predict(model, data):
    return model.predict(data)

# Streamlit UI
def main():
    st.title('Construction Cost and Time Prediction App')

    uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        st.subheader('Preview Data')
        st.write(df.head(1001))

        model, X_test, y_test = train_model(df)

        # Compute evaluation metrics for total project cost prediction
        mse = mean_squared_error(y_test, predict(model, X_test))
        mae = mean_absolute_error(y_test, predict(model, X_test))
        rmse = np.sqrt(mse)

        # Create a bar chart for model evaluation
        eval_metrics = ['Mean Squared Error', 'Mean Absolute Error', 'Root Mean Squared Error']
        eval_values = [mse, mae, rmse]
        eval_df = pd.DataFrame({'Metric': eval_metrics, 'Value': eval_values})

        st.subheader('Model Evaluation Chart')
        st.bar_chart(eval_df.set_index('Metric'))

        # Create two columns for displaying Model Evaluation and Metrics for Quality Prediction side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Model Evaluation')
            
            # Display metric cards for total project cost prediction
            st.metric(label='Mean Squared Error', value=mse)
            st.metric(label='Mean Absolute Error', value=mae)
            st.metric(label='Root Mean Squared Error', value=rmse)

        with col2:
            st.subheader('Metrics for Quality Prediction')
            
            # Define features and target variables for quality prediction
            X_quality = df.drop(columns=['Planned Total Project Cost', 'Actual Duration (days)'])
            y_quality = df['Planned Total Project Cost']

            # Predict quality using the trained model
            y_pred_quality = model.predict(X_quality)

            # Compute MAE, MSE, RMSE for quality prediction
            mae_quality = mean_absolute_error(y_quality, y_pred_quality)
            mse_quality = mean_squared_error(y_quality, y_pred_quality)
            rmse_quality = np.sqrt(mse_quality)

            # Display metric card for quality prediction metrics
            st.metric(label='Mean Absolute Error (Quality Prediction)', value=mae_quality)
            st.metric(label='Mean Squared Error (Quality Prediction)', value=mse_quality)
            st.metric(label='Root Mean Squared Error (Quality Prediction)', value=rmse_quality)

        st.subheader('Make Predictions')
        input_data = st.text_input('Enter input data separated by commas (e.g., Area, Material, Labour, Time, etc.):')
        if st.button('Predict'):
            input_list = [float(x.strip()) for x in input_data.split(',')]
            prediction = predict(model, [input_list[:-1]])  # Exclude the time feature
            st.write('Predicted Cost:', prediction[0])
            st.write('Predicted Time:', input_list[-1])  # Output the provided time

if __name__ == '__main__':
    main()
