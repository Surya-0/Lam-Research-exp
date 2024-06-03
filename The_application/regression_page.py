import streamlit as st
import pandas as pd
import torch
from torch import nn, optim
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Define the forward model for regression data
class ForwardModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ForwardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the regression forward model
def train_regression_model(X_train, y_train, input_dim, output_dim, epochs=1000):
    model = ForwardModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    return model

# Function to perform inverse modeling for regression data
def inverse_model_regression(model, target, initial_input, num_iterations=1000, learning_rate=0.01):
    criterion = nn.MSELoss()
    target_tensor = torch.tensor(target, dtype=torch.float32).view(-1, 1)
    input_tensor = torch.tensor(initial_input, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([input_tensor], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

    return input_tensor.detach().numpy()

def show():

    # Streamlit app
    st.title("Capacity Planning using Neural Networks")

    st.header("Upload Regression Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(df.columns[0], axis=1)
        st.write(df.head())

        # Display Data Statistics
        st.subheader("Data Statistics")
        st.write(df.describe())

        # Display Correlation Matrix
        st.subheader("Correlation Matrix")
        correlation_matrix = df.corr()
        fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig_corr)

        # Feature Selector for Visualization
        feature = st.selectbox('Select a feature for visualization', df.columns[:-1])
        st.write(f'You selected: {feature}')
        fig_feature = px.scatter(df, x=feature, y=df.columns[-1])
        st.plotly_chart(fig_feature)

        target_value = st.number_input("Target Value", value=0.0)

        if 'forward_model' not in st.session_state:
            st.session_state['data'] = df
            st.session_state['X_train'] = None
            st.session_state['y_train'] = None
            st.session_state['scaler_X'] = None
            st.session_state['scaler_y'] = None
            st.session_state['forward_model'] = None
            st.session_state['input_dim'] = None
            st.session_state['output_dim'] = None
            st.session_state['optimized_inputs'] = None
            st.session_state['optimized_inputs_original_scale'] = None
            st.session_state['train_mse'] = None
            st.session_state['test_mse'] = None

        if st.button("Run Model"):
            # Split the data
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

            # Train the forward model
            input_dim = X_train.shape[1]
            output_dim = 1
            forward_model = train_regression_model(X_train, y_train, input_dim, output_dim)

            # Save model and data in session state
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['scaler_X'] = scaler_X
            st.session_state['scaler_y'] = scaler_y
            st.session_state['forward_model'] = forward_model
            st.session_state['input_dim'] = input_dim
            st.session_state['output_dim'] = output_dim

            # Display Model Performance Metrics
            st.subheader("Model Performance Metrics")
            train_predictions = forward_model(torch.tensor(X_train, dtype=torch.float32)).detach().numpy()
            train_mse = np.mean((train_predictions - y_train.reshape(-1, 1)) ** 2)
            test_predictions = forward_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
            test_mse = np.mean((test_predictions - y_test.reshape(-1, 1)) ** 2)

            st.session_state['train_mse'] = train_mse
            st.session_state['test_mse'] = test_mse

            st.write(f"Training MSE: {train_mse:.4f}")
            st.write(f"Test MSE: {test_mse:.4f}")

        if st.session_state['forward_model'] is not None:
            st.write(f"Training MSE: {st.session_state['train_mse']:.4f}")
            st.write(f"Test MSE: {st.session_state['test_mse']:.4f}")

            if st.button("Optimize Inputs"):
                # Perform inverse modeling
                desired_target = st.session_state['scaler_y'].transform([[target_value]])
                initial_input = torch.mean(torch.tensor(st.session_state['X_train'], dtype=torch.float32), dim=0).numpy()
                optimized_inputs = inverse_model_regression(st.session_state['forward_model'], desired_target, initial_input)
                optimized_input_2D = optimized_inputs.reshape(1, -1)
                optimized_inputs_original_scale = st.session_state['scaler_X'].inverse_transform(optimized_input_2D)

                st.session_state['optimized_inputs'] = optimized_inputs
                st.session_state['optimized_inputs_original_scale'] = optimized_inputs_original_scale
                st.session_state['desired_target'] = st.session_state['scaler_y'].inverse_transform(desired_target)

                st.write("Optimized Inputs to achieve the target:")
                st.write(optimized_inputs_original_scale)

        if st.session_state['optimized_inputs_original_scale'] is not None:
            y_test_original_scale = st.session_state['scaler_y'].inverse_transform(st.session_state['y_test'].reshape(-1, 1)).flatten()
            fig = px.scatter(x=st.session_state['X_test'][:, 0], y=y_test_original_scale, labels={'x': 'Feature 1', 'y': 'Target'})
            fig.add_scatter(x=st.session_state['optimized_inputs_original_scale'][0], y=st.session_state['desired_target'][0], mode='markers',
                            marker=dict(color='red', size=12), name='Optimized Input')
            st.plotly_chart(fig)

        # Add a slider for simulation
        if st.session_state['forward_model'] is not None:
            st.header("Simulation")
            input_sliders = []
            for i in range(st.session_state['input_dim']):
                input_sliders.append(st.slider(f'Input {i+1}', float(st.session_state['X_train'][:, i].min()), float(st.session_state['X_train'][:, i].max()), float(st.session_state['X_train'][:, i].mean())))

            if st.button("Simulate"):
                input_values = torch.tensor(input_sliders, dtype=torch.float32).unsqueeze(0)
                model_output = st.session_state['forward_model'](input_values).detach().numpy()
                model_output_original_scale = st.session_state['scaler_y'].inverse_transform(model_output)
                st.write(f"Predicted Output: {model_output_original_scale[0][0]}")

                y_train_original_scale = st.session_state['scaler_y'].inverse_transform(st.session_state['y_train'].reshape(-1, 1)).flatten()
                # Update scatter plot with new input
                fig = px.scatter(x=st.session_state['X_train'][:, 0], y=y_train_original_scale, labels={'x': 'Feature 1', 'y': 'Target'})
                fig.add_scatter(x=[input_values[0][0].item()], y=[model_output_original_scale[0][0]], mode='markers',
                                marker=dict(color='blue', size=12), name='Simulated Input')
                fig.add_scatter(x=st.session_state['optimized_inputs_original_scale'][0], y=st.session_state['desired_target'][0], mode='markers',
                                marker=dict(color='red', size=12), name='Optimized Input')
                st.plotly_chart(fig)
