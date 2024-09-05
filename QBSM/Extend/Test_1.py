import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from ortools.linear_solver import pywraplp
from scipy.optimize import linear_sum_assignment, linprog
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpInteger, value

def BILP_1():

    # Input data
    agents_position = np.array([[2, 2], [2, 23], [23, 2], [23, 23]])
    cluster_center = np.array([[11, 11], [12, 12], [13, 13], [14, 14]])

    # Calculate the distance matrix
    distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
    row, col = distance_matrix.shape

    # Create the problem
    prob = LpProblem("AssignmentProblem", LpMinimize)

    # Variables
    x = LpVariable.dicts("x", [(i, j) for i in range(row) for j in range(col)], 0, 1, LpInteger)

    # Objective
    prob += lpSum(distance_matrix[i, j] * x[i, j] for i in range(row) for j in range(col))

    # Constraints
    for i in range(row):

        prob += lpSum(x[i, j] for j in range(col)) == 1

    for j in range(col):

        prob += lpSum(x[i, j] for i in range(row)) <= 1

    # Solve the problem
    prob.solve()

    # Extract the results
    col_ind = np.zeros(row, dtype=int)

    for i in range(row):

        for j in range(col):

            if value(x[i, j]) > 0:

                col_ind[i] = j

    print("Column Indices using PuLP:\n", col_ind)

def BILP_2():

    # Input data
    agents_position = np.array([[2, 2], [2, 23], [23, 2], [23, 23]])
    cluster_center = np.array([[11, 11], [12, 12], [13, 13], [14, 14]])

    # Calculate the distance matrix
    distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
    row, col = distance_matrix.shape

    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables
    x = {}
    for i in range(row):
        for j in range(col):
            x[i, j] = solver.BoolVar(f'x[{i},{j}]')

    # Constraints
    for i in range(row):
        solver.Add(sum(x[i, j] for j in range(col)) == 1)

    for j in range(col):
        solver.Add(sum(x[i, j] for i in range(row)) <= 1)

    # Objective
    solver.Minimize(solver.Sum(distance_matrix[i, j] * x[i, j] for i in range(row) for j in range(col)))

    # Solve the problem
    status = solver.Solve()

    # Extract the results
    if status == pywraplp.Solver.OPTIMAL:
        col_ind = np.zeros(row, dtype=int)
        for i in range(row):
            for j in range(col):
                if x[i, j].solution_value() > 0:
                    col_ind[i] = j
        print("Column Indices using OR-Tools:\n", col_ind)
    else:
        print("No optimal solution found.")

def BILP_3():

     # Input data
    agents_position = np.array([[2, 2], [2, 23], [23, 2], [23, 23]])
    cluster_center = np.array([[11, 11], [12, 12], [13, 13], [14, 14]])

    # Calculate the distance matrix
    distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
    # print("distance matrix: \n", distance_matrix)

    row, col = np.shape(distance_matrix)
    C = distance_matrix.flatten()
    # print("C: ", C)

    # A Matrix Formulation
    A = np.zeros((row + col, row * col))

    for i in range(row):

        A[i, i * col:(i + 1) * col] = -1
    for j in range(col):

        A[row + j, j::col] = 1
    # print("A: \n", A)

    # B Formulation
    B = np.zeros(row + col)
    B[:row] = -1
    Re = np.ones(col)  # Assuming Re is all ones based on common constraints
    B[row:] = Re
    # print("B: \n", B)

    # Bounds for each variable (x0 and x1) as binary (0 or 1)
    bounds = [(0, 1)] * len(C)
    bounds = tuple(bounds)
    # print("bounds: ", bounds)

    # Solve the binary linear programming problem
    res = linprog(C, A_ub=A, b_ub=B, bounds=bounds, method='highs')

def BILP_4():

    # Input data
    agents_position = np.array([[2, 2], [2, 23], [23, 2], [23, 23]])
    cluster_center = np.array([[11, 11], [12, 12], [13, 13], [14, 14]])

    # Calculate the distance matrix
    distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
    row, col = distance_matrix.shape

    # Create the model
    m = gp.Model("assignment")

    # Variables
    x = m.addVars(row, col, vtype=GRB.BINARY, name="x")

    # Constraints
    m.addConstrs((x.sum(i, '*') == 1 for i in range(row)), "row_constraints")
    m.addConstrs((x.sum('*', j) <= 1 for j in range(col)), "col_constraints")

    # Objective
    m.setObjective(gp.quicksum(distance_matrix[i, j] * x[i, j] for i in range(row) for j in range(col)), GRB.MINIMIZE)

    # Solve the problem
    m.optimize()

    # Extract the results
    col_ind = np.zeros(row, dtype=int)
    if m.status == GRB.OPTIMAL:

        for i in range(row):

            for j in range(col):

                if x[i, j].x > 0:

                    col_ind[i] = j
        print("Column Indices using Gurobi:\n", col_ind)
    else:

        print("No optimal solution found.")

def BILP_5():

    # Input data
    agents_position = np.array([[6.2, 18], [13.5, 3.9], [4.3, 14.65], [21.3, 14.5], [8, 6.5], [16.7, 12.2], [9.4, 21.8], [20, 20]])
    cluster_center = np.array([[11, 11], [12, 12], [13, 13], [14, 14], [2, 2], [2, 23], [23, 2], [23, 23]])

    # Calculate the distance matrix
    distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
    # print("distance matrix: ", distance_matrix)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    print("row_ind, col_ind: ", row_ind, col_ind)

    # Calculate the total cost
    total_cost = sum(distance_matrix[row_ind[i], value] for i, value in enumerate(col_ind))
    print("Total cost:", total_cost, "\n")

def BILP_6():

     # Input data
    # agents_position = np.array([[2, 2], [2, 23], [23, 2], [23, 23]])
    # cluster_center = np.array([[11, 11], [12, 12], [13, 13], [14, 14]])
    agents_position = np.array([[6.2, 18], [13.5, 3.9], [4.3, 14.65], [21.3, 14.5], [8, 6.5], [16.7, 12.2], [9.4, 21.8], [20, 20]])
    cluster_center = np.array([[11, 11], [12, 12], [13, 13], [14, 14], [2, 2], [2, 23], [23, 2], [23, 23]])

    # Calculate the distance matrix
    cost_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)

    # Number of agents and tasks
    num_agents, num_tasks = cost_matrix.shape

    # Initialize lists for available agents and tasks
    available_agents = list(range(num_agents))
    available_tasks = list(range(num_tasks))

    # Initialize the task assignment array
    task_assignment = [-1] * num_agents

    # Greedy algorithm for task assignment
    for _ in range(num_agents):

        min_cost = float('inf')
        best_agent = -1
        best_task = -1

        for agent in available_agents:

            for task in available_tasks:

                if cost_matrix[agent, task] < min_cost:

                    min_cost = cost_matrix[agent, task]
                    best_agent = agent
                    best_task = task

        # Assign the best task to the best agent
        task_assignment[best_agent] = best_task

        # Remove the assigned agent and task from the available lists
        available_agents.remove(best_agent)
        available_tasks.remove(best_task)

        # Print the task assignment result
        print("Task assignment:", task_assignment)

        # Calculate the total cost
        total_cost = sum(cost_matrix[agent, task_assignment[agent]] for agent in range(num_agents))
        print("Total cost:", total_cost)

def BILP_7():

    agents_position = np.array([[6.2, 18], [13.5, 3.9], [4.3, 14.65], [21.3, 14.5], [8, 6.5], [16.7, 12.2], [9.4, 21.8], [20, 20]])
    cluster_center = np.array([[11, 11], [12, 12], [13, 13], [14, 14], [2, 2], [2, 23], [23, 2], [23, 23]])

    # Calculate the distance matrix
    distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
    # print("distance matrix: \n", distance_matrix)

    row, col = np.shape(distance_matrix)
    # C = distance_matrix.reshape((1, row*col))[0]
    # print("C: ", C)
    C = distance_matrix.flatten()
    # print("C: ", C)

    # A Matrix Formulation
    A = np.zeros((row + col, row * col))

    for i in range(row):

        A[i, i * col:(i + 1) * col] = -1
    for j in range(col):

        A[row + j, j::col] = 1
    # print("A: \n", A)

    # B Formulation
    B = np.zeros(row + col)
    B[:row] = -1
    B[row:] = 1
    # print("B: \n", B)

    # Bounds for each variable (x0 and x1) as binary (0 or 1)
    bounds = [(0, 1)] * len(C)
    bounds = tuple(bounds)
    # print("bounds: ", bounds)

    # Solve the binary linear programming problem
    res = linprog(C, A_ub=A, b_ub=B, bounds=bounds, method='highs')

    x_reshaped = res.x.reshape(row, col)
    col_ind = np.argmax(x_reshaped, axis=1)
    print("col_ind: ", col_ind)

if __name__ == "__main__":

    # CT_1_S = time.time()
    # BILP_1()
    # CT_1 = time.time() - CT_1_S

    # CT_2_S = time.time()
    # BILP_2()
    # CT_2 = time.time() - CT_2_S

    # CT_3_S = time.time()
    # BILP_3()
    # CT_3 = time.time() - CT_3_S

    # CT_4_S = time.time()
    # BILP_4()
    # CT_4 = time.time() - CT_4_S

    CT_5_S = time.time()
    BILP_5()
    CT_5 = time.time() - CT_5_S

    CT_6_S = time.time()
    BILP_6()
    CT_6 = time.time() - CT_6_S

    CT_7_S = time.time()
    BILP_7()
    CT_7 = time.time() - CT_7_S

    # print("CT_1: ", CT_1)
    # print("CT_2: ", CT_2)
    # print("CT_3: ", CT_3)
    # print("CT_4: ", CT_4)
    print("CT_5: ", CT_5)
    print("CT_6: ", CT_6)
    print("CT_7: ", CT_7)