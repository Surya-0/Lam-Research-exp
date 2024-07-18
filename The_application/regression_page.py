import streamlit as st
import pandas as pd
import torch
from torch import nn, optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
import plotly.express as px
import altair as alt


class ForwardModel(nn.Module):
    def __init__(self, input_dim, layers):
        super(ForwardModel, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for layer_units in layers:
            self.layers.append(nn.Linear(prev_dim, layer_units))
            prev_dim = layer_units
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


def train_regression_model(X_train, y_train, input_dim, layers, learning_rate, epochs=1000):
    model = ForwardModel(input_dim, layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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


def custom_loss(output, target, input_tensor, input_constraints, penalty_weight=10.0):
    criterion = nn.MSELoss()
    mse_loss = criterion(output, target)
    penalty = 0.0

    for j in range(len(input_constraints)):
        if input_tensor[j] < input_constraints[j][0]:
            penalty += (input_constraints[j][0] - input_tensor[j]) ** 2
        elif input_tensor[j] > input_constraints[j][1]:
            penalty += (input_tensor[j] - input_constraints[j][1]) ** 2

    return mse_loss + penalty_weight * penalty


def inverse_model_regression(model, target, initial_input, input_constraints, num_iterations=1000, learning_rate=0.01):
    target_tensor = torch.tensor(target, dtype=torch.float32).view(-1, 1)
    input_tensor = torch.tensor(initial_input, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([input_tensor], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = custom_loss(output, target_tensor, input_tensor, input_constraints)
        loss.backward()
        optimizer.step()

        for j in range(len(input_constraints)):
            input_tensor.data[j] = torch.clamp(input_tensor.data[j], input_constraints[j][0], input_constraints[j][1])

    return input_tensor.detach().numpy()


def inverse_model_genetic(model, target, initial_input, num_generations=100, population_size=50):
    from deap import base, creator, tools, algorithms

    def evaluate(individual):
        input_tensor = torch.tensor(individual, dtype=torch.float32).view(1, -1)
        output = model(input_tensor).detach().numpy()
        loss = np.mean((output - target) ** 2)
        return loss,

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(initial_input))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=population_size)
    halloffame = tools.HallOfFame(1)

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, halloffame=halloffame,
                        verbose=False)

    return np.array(halloffame[0])


class MultiObjectiveOptimization(ElementwiseProblem):
    def __init__(self, model, target, scaler_X, scaler_y, n_var, maximize_indices, minimize_indices, input_constraints,
                 xl, xu):
        super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu)
        self.model = model
        self.target = target
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.maximize_indices = maximize_indices
        self.minimize_indices = minimize_indices
        self.input_constraints = input_constraints

    def _evaluate(self, x, out, *args, **kwargs):
        input_tensor = torch.tensor(self.scaler_X.transform([x]), dtype=torch.float32)
        output = self.model(input_tensor).detach().numpy()
        target_diff = np.abs(output - self.target).flatten()[0]  # Difference from target, flattened to scalar

        maximize_sum = -np.sum(x[self.maximize_indices])  # Negate to maximize
        minimize_sum = np.sum(x[self.minimize_indices])  # Minimize

        penalty = 0
        for i in range(len(x)):
            if x[i] < self.xl[i]:
                penalty += (self.xl[i] - x[i]) ** 2
            elif x[i] > self.xu[i]:
                penalty += (x[i] - self.xu[i]) ** 2

        # Adding the input constraint penalty
        constraint_penalty = 0
        for j in range(len(self.input_constraints)):
            if x[j] < self.input_constraints[j][0]:
                constraint_penalty += (self.input_constraints[j][0] - x[j]) ** 2
            elif x[j] > self.input_constraints[j][1]:
                constraint_penalty += (x[j] - self.input_constraints[j][1]) ** 2

        out["F"] = [target_diff + penalty, maximize_sum, minimize_sum + constraint_penalty]


def handle_file_upload():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(df.columns[0], axis=1)
        return df
    else:
        return None


def visualize_data(df):
    st.write(df.head())
    fig = px.scatter_matrix(df)
    st.plotly_chart(fig)

    st.subheader("Data Statistics")
    st.write(df.describe())

    st.subheader("Correlation Matrix")
    correlation_matrix = df.corr()
    fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr)

    feature = st.selectbox('Select a feature for visualization', df.columns)
    st.write(f'You selected: {feature}')

    scatter = alt.Chart(df).mark_circle().encode(
        x=feature,
        y=df.columns[-1],
        tooltip=[feature, df.columns[-1]]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

    with st.expander("See explanation"):
        st.write("""
            This scatter plot shows the relationship between the selected feature and the target variable.
            You can zoom and pan the plot, and you can hover over the points to see their values.
        """)


def configure_and_train_model(df):
    learning_rate = st.slider('Select Learning Rate', 0.0001, 0.1, 0.01, step=0.0001)
    epochs = st.slider('Select Number of Epochs', 100, 5000, 1000, step=100)

    st.header("Model Configuration")
    num_layers = st.number_input("Number of layers", min_value=1, max_value=10, value=3)
    layers = []
    for i in range(num_layers):
        units = st.number_input(f"Number of units in layer {i + 1}", min_value=1, max_value=512, value=64)
        layers.append(units)

    if st.button("Run Model"):
        st.header("Training the Model")
        st.write("Splitting the data into training and testing sets, and scaling the data...")

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        input_dim = X_train.shape[1]
        forward_model = train_regression_model(X_train, y_train, input_dim, layers, learning_rate, epochs)

        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['scaler_X'] = scaler_X
        st.session_state['scaler_y'] = scaler_y
        st.session_state['forward_model'] = forward_model
        st.session_state['input_dim'] = input_dim

        st.subheader("Model Performance Metrics")
        st.write("Evaluating model performance on training and testing data...")
        train_predictions = forward_model(torch.tensor(X_train, dtype=torch.float32)).detach().numpy()
        train_mse = np.mean((train_predictions - y_train.reshape(-1, 1)) ** 2)
        test_predictions = forward_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
        test_mse = np.mean((test_predictions - y_test.reshape(-1, 1)) ** 2)

        st.session_state['train_mse'] = train_mse
        st.session_state['test_mse'] = test_mse

        st.write(f"Training MSE: {train_mse:.4f}")
        st.write(f"Test MSE: {test_mse:.4f}")

        st.session_state['model_trained'] = True


def handle_optimization():
    if st.session_state.get('model_trained'):
        st.header("Optimization")
        st.write(
            "Now you can set input constraints, select the optimization method, and optimize the inputs to achieve the desired target.")

        target_value = st.number_input("Target Value", value=200.0)
        input_constraints = []
        for i in range(st.session_state['input_dim']):
            min_val = float(st.session_state['data'].iloc[:, i].min())
            max_val = float(st.session_state['data'].iloc[:, i].max())
            constraint = st.slider(f'Input {i + 1} Constraints', min_val, max_val, (min_val, max_val), step=0.01)
            input_constraints.append((constraint[0], constraint[1]))

        optimization_method = st.selectbox('Select Optimization Method',
                                           ['Gradient-based', 'Genetic Algorithm', 'Multi-objective'])

        if optimization_method == 'Multi-objective':
            available_indices = list(range(st.session_state['input_dim']))
            maximize_indices = st.multiselect('Select Variables to Maximize', available_indices, key='maximize')
            remaining_indices = [i for i in available_indices if i not in maximize_indices]
            minimize_indices = st.multiselect('Select Variables to Minimize', remaining_indices, key='minimize')

        if st.button("Optimize Inputs"):
            st.write("Optimizing inputs to achieve the desired target...")
            desired_target = st.session_state['scaler_y'].transform([[target_value]])
            initial_input = torch.mean(torch.tensor(st.session_state['X_train'], dtype=torch.float32), dim=0).numpy()

            if optimization_method == 'Gradient-based':
                optimized_inputs = inverse_model_regression(st.session_state['forward_model'], desired_target,
                                                            initial_input, input_constraints=input_constraints)
            elif optimization_method == 'Genetic Algorithm':
                optimized_inputs = inverse_model_genetic(st.session_state['forward_model'], desired_target,
                                                         initial_input)
            elif optimization_method == 'Multi-objective':
                xl = np.min(st.session_state['X_train'], axis=0)
                xu = np.max(st.session_state['X_train'], axis=0)

                problem = MultiObjectiveOptimization(
                    model=st.session_state['forward_model'],
                    target=desired_target,
                    scaler_X=st.session_state['scaler_X'],
                    scaler_y=st.session_state['scaler_y'],
                    n_var=st.session_state['input_dim'],
                    maximize_indices=maximize_indices,
                    minimize_indices=minimize_indices,
                    input_constraints=input_constraints,
                    xl=xl,
                    xu=xu
                )

                algorithm = NSGA2(pop_size=200, n_offsprings=100, eliminate_duplicates=True)
                res = minimize(problem, algorithm, ("n_gen", 200), seed=1, verbose=False)

                # st.write("Optimization Results")
                # st.write("Pareto front solutions:")
                # for sol in res.F:
                #     st.write(f"Target Difference: {sol[0]:.4f}, Maximize Objective: {-sol[1]:.4f}, Minimize Objective: {sol[2]:.4f}")

                optimized_inputs = res.X[0]

            optimized_input_2D = np.array(optimized_inputs).reshape(1, -1)
            optimized_inputs_original_scale = st.session_state['scaler_X'].inverse_transform(optimized_input_2D)

            st.session_state['optimized_inputs'] = optimized_inputs
            st.session_state['optimized_inputs_original_scale'] = optimized_inputs_original_scale
            st.session_state['desired_target'] = st.session_state['scaler_y'].inverse_transform(desired_target)

        if st.session_state.get('optimized_inputs') is not None:
            st.subheader("Simulation Results")
            st.write("Here are the optimized inputs to achieve the target:")
            st.write(st.session_state['optimized_inputs_original_scale'])

            st.header("Simulation")
            st.write("You can use the sliders to simulate different inputs and observe the output.")

            input_sliders = []
            for i in range(st.session_state['input_dim']):
                input_sliders.append(st.slider(f'Input {i + 1}', float(np.min(st.session_state['data'].iloc[:, i])),
                                               float(np.max(st.session_state['data'].iloc[:, i])),
                                               float(st.session_state['optimized_inputs_original_scale'][0, i])))

            simulated_inputs = np.array(input_sliders).reshape(1, -1)
            scaled_simulated_inputs = st.session_state['scaler_X'].transform(simulated_inputs)
            simulated_output = st.session_state['forward_model'](
                torch.tensor(scaled_simulated_inputs, dtype=torch.float32)).detach().numpy()
            simulated_output_original_scale = st.session_state['scaler_y'].inverse_transform(simulated_output)
            y_test_2d = st.session_state['y_test'].reshape(-1, 1)
            y_test_rescaled = st.session_state['scaler_y'].inverse_transform(y_test_2d)
            st.write("Simulated Output:")
            st.write(simulated_output_original_scale[0][0])
            y_test_rescaled_1D = y_test_rescaled.flatten()

            fig_optimized = px.scatter(x=st.session_state['X_test'][:, 0], y=y_test_rescaled_1D,
                                       labels={'x': 'Feature 1', 'y': 'Target'})
            fig_optimized.add_scatter(x=st.session_state['optimized_inputs_original_scale'][0],
                                      y=st.session_state['desired_target'][0],
                                      mode='markers', marker=dict(color='red', size=12), name='Optimized Input')
            st.plotly_chart(fig_optimized)

            fig_simulated = px.scatter(x=st.session_state['X_test'][:, 0], y=y_test_rescaled_1D,
                                       labels={'x': 'Feature 1', 'y': 'Target'})
            fig_simulated.add_scatter(x=simulated_inputs[0], y=simulated_output_original_scale[0], mode='markers',
                                      marker=dict(color='blue', size=12), name='Simulated Input')
            st.plotly_chart(fig_simulated)


def show():
    st.title("Capacity Planning using Neural Networks")

    st.header("Upload Regression Data")
    df = handle_file_upload()

    if df is not None:
        visualize_data(df)

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

        configure_and_train_model(df)
        handle_optimization()

# import streamlit as st
# import pandas as pd
# import torch
# from torch import nn, optim
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.optimize import minimize
# from pymoo.core.problem import ElementwiseProblem
# import plotly.express as px
# import altair as alt
# import gymnasium as gym
# from gymnasium import spaces
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
#
#
#
# class ForwardModel(nn.Module):
#     def __init__(self, input_dim, layers):
#         super(ForwardModel, self).__init__()
#         self.layers = nn.ModuleList()
#         prev_dim = input_dim
#         for layer_units in layers:
#             self.layers.append(nn.Linear(prev_dim, layer_units))
#             prev_dim = layer_units
#         self.output_layer = nn.Linear(prev_dim, 1)
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = torch.relu(layer(x))
#         x = self.output_layer(x)
#         return x
#
#
# def train_regression_model(X_train, y_train, input_dim, layers, learning_rate, epochs=1000):
#     model = ForwardModel(input_dim, layers)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#
#     return model
#
#
# def custom_loss(output, target, input_tensor, input_constraints, penalty_weight=10.0):
#     criterion = nn.MSELoss()
#     mse_loss = criterion(output, target)
#     penalty = 0.0
#
#     for j in range(len(input_constraints)):
#         if input_tensor[j] < input_constraints[j][0]:
#             penalty += (input_constraints[j][0] - input_tensor[j]) ** 2
#         elif input_tensor[j] > input_constraints[j][1]:
#             penalty += (input_tensor[j] - input_constraints[j][1]) ** 2
#
#     return mse_loss + penalty_weight * penalty
#
#
# def inverse_model_regression(model, target, initial_input, input_constraints, num_iterations=1000, learning_rate=0.01):
#     target_tensor = torch.tensor(target, dtype=torch.float32).view(-1, 1)
#     input_tensor = torch.tensor(initial_input, dtype=torch.float32, requires_grad=True)
#     optimizer = optim.Adam([input_tensor], lr=learning_rate)
#
#     for i in range(num_iterations):
#         optimizer.zero_grad()
#         output = model(input_tensor)
#         loss = custom_loss(output, target_tensor, input_tensor, input_constraints)
#         loss.backward()
#         optimizer.step()
#
#         for j in range(len(input_constraints)):
#             input_tensor.data[j] = torch.clamp(input_tensor.data[j], input_constraints[j][0], input_constraints[j][1])
#
#     return input_tensor.detach().numpy()
#
#
# def inverse_model_genetic(model, target, initial_input, num_generations=100, population_size=50):
#     from deap import base, creator, tools, algorithms
#
#     def evaluate(individual):
#         input_tensor = torch.tensor(individual, dtype=torch.float32).view(1, -1)
#         output = model(input_tensor).detach().numpy()
#         loss = np.mean((output - target) ** 2)
#         return loss,
#
#     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMin)
#
#     toolbox = base.Toolbox()
#     toolbox.register("attr_float", np.random.uniform, -1, 1)
#     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(initial_input))
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
#     toolbox.register("mate", tools.cxBlend, alpha=0.5)
#     toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
#     toolbox.register("select", tools.selTournament, tournsize=3)
#     toolbox.register("evaluate", evaluate)
#
#     population = toolbox.population(n=population_size)
#     halloffame = tools.HallOfFame(1)
#
#     algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, halloffame=halloffame,
#                         verbose=False)
#
#     return np.array(halloffame[0])
#
#
# class MultiObjectiveOptimization(ElementwiseProblem):
#     def __init__(self, model, target, scaler_X, scaler_y, n_var, maximize_indices, minimize_indices, input_constraints,
#                  xl, xu):
#         super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu)
#         self.model = model
#         self.target = target
#         self.scaler_X = scaler_X
#         self.scaler_y = scaler_y
#         self.maximize_indices = maximize_indices
#         self.minimize_indices = minimize_indices
#         self.input_constraints = input_constraints
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         input_tensor = torch.tensor(self.scaler_X.transform([x]), dtype=torch.float32)
#         output = self.model(input_tensor).detach().numpy()
#         target_diff = np.abs(output - self.target).flatten()[0]  # Difference from target, flattened to scalar
#
#         maximize_sum = -np.sum(x[self.maximize_indices])  # Negate to maximize
#         minimize_sum = np.sum(x[self.minimize_indices])  # Minimize
#
#         penalty = 0
#         for i in range(len(x)):
#             if x[i] < self.xl[i]:
#                 penalty += (self.xl[i] - x[i]) ** 2
#             elif x[i] > self.xu[i]:
#                 penalty += (x[i] - self.xu[i]) ** 2
#
#         # Adding the input constraint penalty
#         constraint_penalty = 0
#         for j in range(len(self.input_constraints)):
#             if x[j] < self.input_constraints[j][0]:
#                 constraint_penalty += (self.input_constraints[j][0] - x[j]) ** 2
#             elif x[j] > self.input_constraints[j][1]:
#                 constraint_penalty += (x[j] - self.input_constraints[j][1]) ** 2
#
#         out["F"] = [target_diff**2 + penalty + constraint_penalty, maximize_sum, minimize_sum]
#
#
# class CapacityPlanningEnv(gym.Env):
#     def __init__(self, model, scaler_X, scaler_y, target, input_dim, maximize_indices, minimize_indices, input_constraints):
#         super(CapacityPlanningEnv, self).__init__()
#         self.model = model
#         self.scaler_X = scaler_X
#         self.scaler_y = scaler_y
#         self.target = target
#         self.input_dim = input_dim
#         self.maximize_indices = maximize_indices
#         self.minimize_indices = minimize_indices
#         self.input_constraints = input_constraints
#
#         # Action space: Continuous values for each input variable
#         self.action_space = spaces.Box(low=-1, high=1, shape=(input_dim,), dtype=np.float32)
#
#         # Observation space: Scaled input variables
#         self.observation_space = spaces.Box(low=0, high=1, shape=(input_dim,), dtype=np.float32)
#
#         # Initial state
#         self.state = self._get_initial_state()
#
#     def _get_initial_state(self):
#         initial_input = torch.mean(
#             torch.tensor(self.scaler_X.transform(st.session_state['X_train']), dtype=torch.float32), dim=0).numpy()
#         return self.scaler_X.transform([initial_input])[0].astype(np.float32)
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.state = self._get_initial_state()
#         return self.state, {}  # Return a tuple with the state and an empty info dictionary
#
#     def step(self, action):
#         action = np.clip(action, -1, 1)
#         scaled_action = self.scaler_X.inverse_transform([action])[0]
#
#         # Apply constraints
#         for i in range(self.input_dim):
#             scaled_action[i] = np.clip(scaled_action[i], self.input_constraints[i][0], self.input_constraints[i][1])
#
#         input_tensor = torch.tensor([scaled_action], dtype=torch.float32)
#         output = self.model(input_tensor).detach().numpy()[0]
#         target_diff = np.abs(output - self.target).flatten()[0]
#
#         # Calculate reward
#         maximize_sum = np.sum(scaled_action[self.maximize_indices])
#         minimize_sum = np.sum(scaled_action[self.minimize_indices])
#         reward = -target_diff - 0.1 * maximize_sum + 0.1 * minimize_sum
#
#         self.state = self.scaler_X.transform([scaled_action])[0].astype(np.float32)
#         done = target_diff < 1e-3  # Consider done if the target difference is very small
#
#         return self.state, reward, done, {}
#
#     def render(self, mode='human'):
#         pass
#
# def handle_file_upload():
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         df = df.drop(df.columns[0], axis=1)
#         return df
#     else:
#         return None
#
#
# def visualize_data(df):
#     st.write(df.head())
#     fig = px.scatter_matrix(df)
#     st.plotly_chart(fig)
#
#     st.subheader("Data Statistics")
#     st.write(df.describe())
#
#     st.subheader("Correlation Matrix")
#     correlation_matrix = df.corr()
#     fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
#     st.plotly_chart(fig_corr)
#
#     feature = st.selectbox('Select a feature for visualization', df.columns)
#     st.write(f'You selected: {feature}')
#
#     scatter = alt.Chart(df).mark_circle().encode(
#         x=feature,
#         y=df.columns[-1],
#         tooltip=[feature, df.columns[-1]]
#     ).interactive()
#     st.altair_chart(scatter, use_container_width=True)
#
#     with st.expander("See explanation"):
#         st.write("""
#             This scatter plot shows the relationship between the selected feature and the target variable.
#             You can zoom and pan the plot, and you can hover over the points to see their values.
#         """)
#
#
# def configure_and_train_model(df):
#     learning_rate = st.slider('Select Learning Rate', 0.0001, 0.1, 0.01, step=0.0001)
#     epochs = st.slider('Select Number of Epochs', 100, 5000, 1000, step=100)
#
#     st.header("Model Configuration")
#     num_layers = st.number_input("Number of layers", min_value=1, max_value=10, value=3)
#     layers = []
#     for i in range(num_layers):
#         units = st.number_input(f"Number of units in layer {i + 1}", min_value=1, max_value=512, value=64)
#         layers.append(units)
#
#     if st.button("Run Model"):
#         st.header("Training the Model")
#         st.write("Splitting the data into training and testing sets, and scaling the data...")
#
#         X = df.iloc[:, :-1].values
#         y = df.iloc[:, -1].values
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         scaler_X = StandardScaler()
#         scaler_y = StandardScaler()
#         X_train = scaler_X.fit_transform(X_train)
#         X_test = scaler_X.transform(X_test)
#         y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
#         y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
#
#         input_dim = X_train.shape[1]
#         forward_model = train_regression_model(X_train, y_train, input_dim, layers, learning_rate, epochs)
#
#         st.session_state['X_train'] = X_train
#         st.session_state['X_test'] = X_test
#         st.session_state['y_train'] = y_train
#         st.session_state['y_test'] = y_test
#         st.session_state['scaler_X'] = scaler_X
#         st.session_state['scaler_y'] = scaler_y
#         st.session_state['forward_model'] = forward_model
#         st.session_state['input_dim'] = input_dim
#
#         st.subheader("Model Performance Metrics")
#         st.write("Evaluating model performance on training and testing data...")
#         train_predictions = forward_model(torch.tensor(X_train, dtype=torch.float32)).detach().numpy()
#         train_mse = np.mean((train_predictions - y_train.reshape(-1, 1)) ** 2)
#         test_predictions = forward_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
#         test_mse = np.mean((test_predictions - y_test.reshape(-1, 1)) ** 2)
#
#         st.session_state['train_mse'] = train_mse
#         st.session_state['test_mse'] = test_mse
#
#         st.write(f"Training MSE: {train_mse:.4f}")
#         st.write(f"Test MSE: {test_mse:.4f}")
#
#         st.session_state['model_trained'] = True
#
#
# def handle_optimization():
#     if st.session_state.get('model_trained'):
#         st.header("Optimization")
#         st.write(
#             "Now you can set input constraints, select the optimization method, and optimize the inputs to achieve "
#             "the desired target.")
#
#         target_value = st.number_input("Target Value", value=200.0)
#         input_constraints = []
#         for i in range(st.session_state['input_dim']):
#             min_val = float(st.session_state['data'].iloc[:, i].min())
#             max_val = float(st.session_state['data'].iloc[:, i].max())
#             constraint = st.slider(f'Input {i + 1} Constraints', min_val, max_val, (min_val, max_val), step=0.01)
#             input_constraints.append((constraint[0], constraint[1]))
#
#         optimization_method = st.selectbox('Select Optimization Method',
#                                            ['Gradient-based', 'Genetic Algorithm', 'Multi-objective',
#                                             'Reinforcement Learning'])
#
#         if optimization_method == 'Multi-objective' or optimization_method == 'Reinforcement Learning':
#             available_indices = list(range(st.session_state['input_dim']))
#             maximize_indices = st.multiselect('Select Variables to Maximize', available_indices, key='maximize')
#             remaining_indices = [i for i in available_indices if i not in maximize_indices]
#             minimize_indices = st.multiselect('Select Variables to Minimize', remaining_indices, key='minimize')
#
#         if st.button("Optimize Inputs"):
#             st.write("Optimizing inputs to achieve the desired target...")
#             desired_target = st.session_state['scaler_y'].transform([[target_value]])
#             initial_input = torch.mean(torch.tensor(st.session_state['X_train'], dtype=torch.float32), dim=0).numpy()
#
#             if optimization_method == 'Gradient-based':
#                 optimized_inputs = inverse_model_regression(st.session_state['forward_model'], desired_target,
#                                                             initial_input, input_constraints=input_constraints)
#             elif optimization_method == 'Genetic Algorithm':
#                 optimized_inputs = inverse_model_genetic(st.session_state['forward_model'], desired_target,
#                                                          initial_input)
#             elif optimization_method == 'Multi-objective':
#                 xl = np.min(st.session_state['X_train'], axis=0)
#                 xu = np.max(st.session_state['X_train'], axis=0)
#
#                 problem = MultiObjectiveOptimization(
#                     model=st.session_state['forward_model'],
#                     target=desired_target,
#                     scaler_X=st.session_state['scaler_X'],
#                     scaler_y=st.session_state['scaler_y'],
#                     n_var=st.session_state['input_dim'],
#                     maximize_indices=maximize_indices,
#                     minimize_indices=minimize_indices,
#                     input_constraints=input_constraints,
#                     xl=xl,
#                     xu=xu
#                 )
#
#                 algorithm = NSGA2(pop_size=200, n_offsprings=100, eliminate_duplicates=True)
#                 res = minimize(problem, algorithm, ("n_gen", 200), seed=1, verbose=False)
#
#                 optimized_inputs = res.X[0]
#
#             elif optimization_method == 'Reinforcement Learning':
#                 env = CapacityPlanningEnv(
#                     model=st.session_state['forward_model'],
#                     scaler_X=st.session_state['scaler_X'],
#                     scaler_y=st.session_state['scaler_y'],
#                     target=desired_target,
#                     input_dim=st.session_state['input_dim'],
#                     maximize_indices=maximize_indices,
#                     minimize_indices=minimize_indices,
#                     input_constraints=input_constraints
#                 )
#
#                 check_env(env)
#                 model = PPO("MlpPolicy", env, verbose=1)
#                 model.learn(total_timesteps=10000)
#
#                 obs = env.reset()
#                 optimized_inputs, _ = model.predict(obs, deterministic=True)
#
#             optimized_input_2D = np.array(optimized_inputs).reshape(1, -1)
#             optimized_inputs_original_scale = st.session_state['scaler_X'].inverse_transform(optimized_input_2D)
#
#             st.session_state['optimized_inputs'] = optimized_inputs
#             st.session_state['optimized_inputs_original_scale'] = optimized_inputs_original_scale
#             st.session_state['desired_target'] = st.session_state['scaler_y'].inverse_transform(desired_target)
#
#         if st.session_state.get('optimized_inputs') is not None:
#             st.subheader("Simulation Results")
#             st.write("Here are the optimized inputs to achieve the target:")
#             st.write(st.session_state['optimized_inputs_original_scale'])
#
#             st.header("Simulation")
#             st.write("You can use the sliders to simulate different inputs and observe the output.")
#
#             input_sliders = []
#             for i in range(st.session_state['input_dim']):
#                 input_sliders.append(st.slider(f'Input {i + 1}', float(np.min(st.session_state['data'].iloc[:, i])),
#                                                float(np.max(st.session_state['data'].iloc[:, i])),
#                                                float(st.session_state['optimized_inputs_original_scale'][0, i])))
#
#             simulated_inputs = np.array(input_sliders).reshape(1, -1)
#             scaled_simulated_inputs = st.session_state['scaler_X'].transform(simulated_inputs)
#             simulated_output = st.session_state['forward_model'](
#                 torch.tensor(scaled_simulated_inputs, dtype=torch.float32)).detach().numpy()
#             simulated_output_original_scale = st.session_state['scaler_y'].inverse_transform(simulated_output)
#             y_test_2d = st.session_state['y_test'].reshape(-1, 1)
#             y_test_rescaled = st.session_state['scaler_y'].inverse_transform(y_test_2d)
#             st.write("Simulated Output:")
#             st.write(simulated_output_original_scale[0][0])
#             y_test_rescaled_1D = y_test_rescaled.flatten()
#
#             fig_optimized = px.scatter(x=st.session_state['X_test'][:, 0], y=y_test_rescaled_1D,
#                                        labels={'x': 'Feature 1', 'y': 'Target'})
#             fig_optimized.add_scatter(x=st.session_state['optimized_inputs_original_scale'][0],
#                                       y=st.session_state['desired_target'][0],
#                                       mode='markers', marker=dict(color='red', size=12), name='Optimized Input')
#             st.plotly_chart(fig_optimized)
#
#             fig_simulated = px.scatter(x=st.session_state['X_test'][:, 0], y=y_test_rescaled_1D,
#                                        labels={'x': 'Feature 1', 'y': 'Target'})
#             fig_simulated.add_scatter(x=simulated_inputs[0], y=simulated_output_original_scale[0], mode='markers',
#                                       marker=dict(color='blue', size=12), name='Simulated Input')
#             st.plotly_chart(fig_simulated)
#
#
# def show():
#     st.title("Capacity Planning using Neural Networks")
#
#     st.header("Upload Regression Data")
#     df = handle_file_upload()
#
#     if df is not None:
#         visualize_data(df)
#
#         if 'forward_model' not in st.session_state:
#             st.session_state['data'] = df
#             st.session_state['X_train'] = None
#             st.session_state['y_train'] = None
#             st.session_state['scaler_X'] = None
#             st.session_state['scaler_y'] = None
#             st.session_state['forward_model'] = None
#             st.session_state['input_dim'] = None
#             st.session_state['output_dim'] = None
#             st.session_state['optimized_inputs'] = None
#             st.session_state['optimized_inputs_original_scale'] = None
#             st.session_state['train_mse'] = None
#             st.session_state['test_mse'] = None
#
#         configure_and_train_model(df)
#         handle_optimization()
#
#
# if __name__ == "__main__":
#     show()
