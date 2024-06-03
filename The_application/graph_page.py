import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import Adam


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


def train_graph_model(data, epochs=200):
    model = GCN(data.num_features, 128, 1)
    model.fit(data, epochs)
    return model


def inverse_model_optimization(model, target_value, data, epochs=200):
    target_output = torch.tensor([target_value for _ in range(data.num_nodes)], dtype=torch.float).unsqueeze(1)
    inverse_model = InverseModel(model)
    return inverse_model.optimize_inputs(target_output, data, epochs)


def show():
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
            node_features = torch.tensor(nodes_df.values, dtype=torch.float)
            edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
            data = Data(x=node_features, edge_index=edge_index)
            data.train_mask = torch.ones(node_features.size(0), dtype=torch.bool)
            data.val_mask = torch.zeros(node_features.size(0), dtype=torch.bool)
            data.test_mask = torch.zeros(node_features.size(0), dtype=torch.bool)

            model = train_graph_model(data)
            optimized_inputs = inverse_model_optimization(model, target_value, data)

            st.write("Optimized Inputs to achieve the target:")
            st.write(optimized_inputs.detach().numpy())
