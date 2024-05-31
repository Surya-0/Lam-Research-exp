import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

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

        if (i + 1) % 100 == 0:
            print(f'Iteration [{i + 1}/{num_iterations}], Loss: {loss.item():.4f}')

    return input_tensor.detach().numpy()


# Define the GCN model for graph data
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(dim_in, dim_h * 4)
        self.gcn2 = GCNConv(dim_h * 4, dim_h * 2)
        self.gcn3 = GCNConv(dim_h * 2, dim_h)
        self.linear = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn3(h, edge_index)
        h = torch.relu(h)
        h = self.linear(h)
        return h

    def fit(self, data, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = F.mse_loss(out.squeeze()[data.val_mask], data.y[data.val_mask])
                print(f"Epoch {epoch:>3} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}")

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask], data.y[data.test_mask].float())


# Class for inverse modeling of graph data
class InverseModel:
    def __init__(self, model):
        self.model = model

    def optimize_inputs(self, target_output, data, epochs=200):
        optimized_data = data.clone()
        optimized_data.x = torch.randn_like(data.x, requires_grad=True)
        optimizer = Adam([optimized_data.x], lr=0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(optimized_data.x, optimized_data.edge_index)
            loss = F.mse_loss(out.squeeze(), target_output)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch:>3} | Optimization Loss: {loss:.5f}")

        return optimized_data.x


# Helper function to train the GCN model
def train_graph_model(data, epochs=200):
    model = GCN(data.num_features, 128, 1)
    model.fit(data, epochs)
    return model


# Helper function to optimize inputs for graph data
def inverse_model_optimization(model, target_value, data, epochs=200):
    target_output = torch.tensor([target_value for _ in range(data.num_nodes)], dtype=torch.float).unsqueeze(1)
    inverse_model = InverseModel(model)
    return inverse_model.optimize_inputs(target_output, data, epochs)


# Streamlit app
st.title("Capacity Planning using GNN")

page = st.sidebar.selectbox("Choose a page", ["Regression Data", "Graph Data"])

if page == "Regression Data":
    st.header("Upload Regression Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    target_value = st.number_input("Target Value", value=0.0)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(df.columns[0],axis = 1)
        st.write(df.head())
        if st.button("Run Model"):
            # Split the data
            X = df.iloc[:,:-1].values
            y = df.iloc[:,-1].values
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

            # Perform inverse modeling
            desired_target = scaler_y.transform([[target_value]])
            initial_input = torch.mean(torch.tensor(X_train, dtype=torch.float32), dim=0).numpy()
            optimized_inputs = inverse_model_regression(forward_model, desired_target, initial_input)
            optimized_input_2D = optimized_inputs.reshape(1, -1)
            optimized_inputs_original_scale = scaler_X.inverse_transform(optimized_input_2D)

            st.write("Optimized Inputs to achieve the target:")
            st.write(optimized_inputs_original_scale)

elif page == "Graph Data":
    st.header("Upload Graph Data")
    uploaded_nodes_file = st.file_uploader("Choose a CSV file for Nodes", type="csv")
    uploaded_edges_file = st.file_uploader("Choose a CSV file for Edges", type="csv")
    target_value = st.number_input("Target Value", value=0.0)

    if uploaded_nodes_file is not None and uploaded_edges_file is not None:
        nodes_df = pd.read_csv(uploaded_nodes_file)
        edges_df = pd.read_csv(uploaded_edges_file)
        st.write("Nodes Data")
        st.write(nodes_df.head())
        st.write("Edges Data")
        st.write(edges_df.head())
        if st.button("Run Model"):
            # Convert to PyTorch tensors
            node_features = torch.tensor(nodes_df.values, dtype=torch.float)
            edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
            # Create PyTorch Geometric Data object
            data = Data(x=node_features, edge_index=edge_index)
            data.train_mask = torch.ones(node_features.size(0), dtype=torch.bool)  # Example train mask
            data.val_mask = torch.zeros(node_features.size(0), dtype=torch.bool)  # Example val mask
            data.test_mask = torch.zeros(node_features.size(0), dtype=torch.bool)  # Example test mask

            # Train model
            model = train_graph_model(data)

            # Perform inverse modeling
            optimized_inputs = inverse_model_optimization(model, target_value, data)

            st.write("Optimized Inputs to achieve the target:")
            st.write(optimized_inputs.detach().numpy())
