import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.pcx import PCX
from pymoo.operators.mutation.pm import PM

class SupplyChainProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([0.0, 0.0]), xu=np.array([1.0, 1.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]**2 + x[1]**2  #  objective: minimize cost
        f2 = (x[0] - 1)**2 + (x[1] - 1)**2  # objective: minimize delivery time
        out["F"] = [f1, f2]

problem = SupplyChainProblem()

algorithm = NSGA2(
    pop_size=100,
    sampling=LHS(), # Latin Hypercube Sampling (LHS
    crossover=PCX(), # Parent-Centric Crossover
    mutation=PM(), # Polynomial Mutation
    eliminate_duplicates=True
)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True)

# Print Pareto-optimal solutions
print("Pareto-optimal solutions:")
for i in range(len(res.X)):
    print(f"Solution {i + 1}: {res.X[i]}, Objectives: {res.F[i]}")
