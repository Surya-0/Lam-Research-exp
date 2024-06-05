import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import Adam
import plotly.express as px
from deap import base, creator, tools, algorithms
import numpy as np
from joblib import Parallel, delayed
import time


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

    def fit(self, data, epochs, progress_callback):
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
                progress_callback(epoch, loss.item(), val_loss.item())
                print(f"Epoch {epoch:>3} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}")

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask], data.y[data.test_mask].float())


class InverseModel:
    def __init__(self, model):
        self.model = model

    def optimize_inputs_gradient(self, target_output, data, node_idx, epochs=200, lr=0.01):
        optimized_data = data.clone()
        optimized_data.x[node_idx] = torch.randn_like(data.x[node_idx])
        optimized_data.x[node_idx].requires_grad = True
        optimizer = Adam([optimized_data.x[node_idx]], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(optimized_data.x, optimized_data.edge_index)
            loss = F.mse_loss(out[node_idx], target_output)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch:>3} | Optimization Loss: {loss:.5f}")

        return optimized_data.x[node_idx]

    def optimize_inputs_genetic(self, target_output, data, node_idx, num_generations=100, population_size=50):
        def evaluate(individual):
            input_tensor = torch.tensor(individual, dtype=torch.float32).view(1, -1)
            data_clone = data.clone()
            data_clone.x[node_idx] = input_tensor
            output = self.model(data_clone.x, data_clone.edge_index).detach().numpy()
            target_output_np = target_output.numpy()  # Convert Tensor to numpy array
            loss = np.mean((output[node_idx] - target_output_np) ** 2)  # Use the numpy array in the calculation
            return loss,

        # DEAP setup
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=data.num_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)

        population = toolbox.population(n=population_size)
        halloffame = tools.HallOfFame(1)

        start_time = time.time()

        # Optional parallel evaluation
        def parallel_evaluate(individual):
            return toolbox.evaluate(individual)

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = Parallel(n_jobs=-1)(delayed(parallel_evaluate)(ind) for ind in invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, halloffame=halloffame,
                            verbose=False)

        end_time = time.time()
        print(f"Genetic Algorithm Optimization Time: {end_time - start_time} seconds")

        return torch.from_numpy(np.array(halloffame[0]))


def train_graph_model(data, epochs=200):
    model = GCN(data.num_features, 128, 1)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_progress(epoch, train_loss, val_loss):
        progress_bar.progress(epoch / epochs)
        progress_text.text(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    model.fit(data, epochs, update_progress)
    return model


def inverse_model_optimization(model, target_value, data, node_idx, method, epochs=200, lr=0.01):
    target_output = torch.tensor([target_value], dtype=torch.float).unsqueeze(1)
    inverse_model = InverseModel(model)
    if method == 'Gradient-based':
        return inverse_model.optimize_inputs_gradient(target_output, data, node_idx, epochs, lr)
    elif method == 'Genetic Algorithm':
        return inverse_model.optimize_inputs_genetic(target_output, data, node_idx)


def show():
    st.header("Upload Graph Data")
    uploaded_nodes_file = st.file_uploader("Choose a CSV file for Nodes", type="csv")
    uploaded_edges_file = st.file_uploader("Choose a CSV file for Edges", type="csv")
    target_value = st.number_input("Target Value", value=0.0)
    node_idx = st.number_input("Node Index to Optimize", min_value=0, step=1, value=0)
    optimization_method = st.selectbox('Select Optimization Method', ['Gradient-based', 'Genetic Algorithm'])

    if uploaded_nodes_file is not None and uploaded_edges_file is not None:
        nodes_df = pd.read_csv(uploaded_nodes_file)
        edges_df = pd.read_csv(uploaded_edges_file)
        st.write("Nodes Data")
        st.write(nodes_df.head())
        st.write("Edges Data")
        st.write(edges_df.head())

        if 'model' not in st.session_state:
            st.session_state['model'] = None

        if st.button("Run Model"):
            node_features = torch.tensor(nodes_df.iloc[:, :-1].values, dtype=torch.float)
            edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.int64)
            y = torch.tensor(nodes_df.iloc[:, -1].values, dtype=torch.float32)
            data = Data(x=node_features, edge_index=edge_index, y=y)
            data.train_mask = torch.arange(1200)
            data.val_mask = torch.arange(1200, 1600)
            data.test_mask = torch.arange(1600, 2000)

            model = train_graph_model(data)
            st.session_state['model'] = model
            st.session_state['data'] = data

        if st.session_state['model'] is not None and st.button("Optimize Inputs"):
            optimized_input = inverse_model_optimization(st.session_state['model'], target_value,
                                                         st.session_state['data'], node_idx, optimization_method)
            st.write("Optimized Input for the target:")
            st.write(optimized_input)

            # Pass the optimized input back to the model to get an output
            data = st.session_state['data'].clone()
            data.x[node_idx] = optimized_input
            model_output = st.session_state['model'](data.x, data.edge_index).detach().numpy()
            st.write("Model Output after optimization:")
            st.write(model_output[node_idx])


