import osqp
import numpy as np
import gurobipy as gp
from scipy import sparse
from datetime import datetime
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation

# Create an OSQP object
# prob = osqp.OSQP()

# P = sparse.csc_matrix([[1.0, 0.0], [0.0, 1.0]])
# q = np.array([1.0, 1.0])
# A = sparse.csc_matrix([1, 1])
# l = np.array([0.0])
# u = np.array([1.0])

# # Setup the problem
# prob.setup(P, q, A, l, u, verbose=False, warm_start=True)

# def CBF(point1, point2, u1_nom, u2_nom):

#     # Distance
#     d = 1*(point1-point2)
#     dmin = 1.0
#     h = np.linalg.norm(d)**2 - dmin**2

#     # Define the QP problem - x is [V1_x, V1_y]
#     P = sparse.csc_matrix([ [1.0, 0.0], [0.0, 1.0] ])
#     q = np.array([-2*u1_nom[0], -2*u1_nom[1]])
#     A = sparse.csc_matrix([ [-2*d[0], -2*d[1]]])
#     l = np.array([0.0])
#     u = np.array([h - 2*np.dot(d, u2_nom)])

#     # Update the problem
#     prob.update(q=q, l=l, u=u, Ax=A.data)

#     # Ininial Guess
#     prob.warm_start(x=u1_nom)

#     # Solve the problem
#     result = prob.solve()

#     # Print the solution
#     print("Solution status:", result.info.status)
#     print("Optimal solution:", result.x, "\n")

#     u_des = np.array([result.x[0], result.x[1]])

#     return u_des

# A, b, P, q = 0, 0, 0, 0
# # Define the objective function
# def objective(x):

#     global P, q

#     return 0.5 * np.dot(x, np.dot(P, x)) + np.dot(q, x)

# # Define the linear inequality constraint
# def constraint(x):

#     global A, b

#     return np.dot(A, x) - b

# def CBF(point1, point2, u1_nom, u2_nom):

#     # Distance
#     d = 1*(point1-point2)
#     # print("d: ", d)
#     dmin = 1.0
#     h = np.linalg.norm(d)**2 - dmin**2

#     global A, b, P, q
#     # Objective function: minimize 0.5 * x^T * P * x + q^T * x
#     P = np.array([[2.0, 0.0], [0.0, 2.0]])
#     q = np.array([-2*u1_nom[0], -2.0*u1_nom[1]])

#     # Linear inequality constraint: A * x <= b
#     A = np.array([[-2*d[0], -2*d[1]]])
#     b = np.array([h-2*np.dot(d, u2_nom)])
#     # print("b: ", b)

#     # Variable bounds: l <= x <= u
#     bounds = [(-1.0, 1.0), (-1.0, 1.0)]

#     # Solve the optimization problem
#     result = minimize(objective, x0=[u1_nom[0], u1_nom[1]], constraints={'type': 'ineq', 'fun': constraint}, bounds=bounds)

#     print("Optimal Solution: ", result.x , "\n")

#     u_des = np.array([result.x[0], result.x[1]])

#     return u_des

solvers.options['show_progress'] = False

def CBF(point1, point2, point3, GCM, C, u1_nom, u2_nom, u3_nom, dmin, theta_max, x_boundary, y_boundary, x_bmin, y_bmin):

    # Distance
    d = 1*(point1-point2)
    # d3 = 1*(point1-point3)
    # print("d: ", d)
    h = np.linalg.norm(d)**2 - dmin**2
    # h3 = np.linalg.norm(d3)**2 - dmin**2

    # Herding
    V = (C-GCM)/np.linalg.norm(C-GCM)
    L = np.dot(((point1-C)/np.linalg.norm(point1-C)), V)
    h_theta = theta_max - np.arccos(L)
    A_2 = (-1/np.sqrt(1-L**2))*( V/np.linalg.norm(point1-C) - (np.dot((point1-C), V)/(np.linalg.norm(point1-C)**3))*(point1-C) )

    # Boundary
    hx = x_bmin**2 - (point1[0] - x_boundary[1]/2)**2
    hy = y_bmin**2 - (point1[1] - y_boundary[1]/2)**2

    # QP
    Q = 2*matrix([[1.0, 0.0],[0.0, 1.0]])
    # print("Q: ", Q)
    p = matrix([-2*u1_nom[0], -2*u1_nom[1]])
    # print("p: ", p)

    # G = matrix([ [-2*d[0], -2*d[1]] ], (1,2))
    # G = matrix([ [-2*d[0], -2*d[1]], [-2*d3[0], -2*d3[1]] ], (2,2))
    # print("G: ", G)
    # G = matrix([ [1.0*A_2[0], 1.0*A_2[1]] ], (1,2))
    # print("G: ", G)
    # G = matrix([ [-2*d[0], 1.0*A_2[0]], [-2*d[1], 1.0*A_2[1]] ])
    # print("G: ", G)
    G = matrix([ [2.0*(point1[0] - x_bmin), 0.0], [0.0, 2.0*(point1[1] - y_bmin)] ])
    print("G: ", G)

    # h = matrix([0.5*h-2*np.dot(d, u2_nom)])
    # h = matrix([h-2*np.dot(d, u2_nom), h3-2*np.dot(d3, u3_nom)])
    # h = matrix([h_theta])
    # h = matrix([1.0*h-2*np.dot(d, u2_nom), h_theta])
    h = matrix([hx, hy])
    print("h: ", h)

    # A = matrix([0.0, 0.0], (1,2))
    # print("A: ", A)
    # b = matrix(0.0)
    # print("b: ", b)
    # sol = solvers.qp(Q, p, G, h)
    sol = solvers.coneqp(Q, p, G, h)

    print("solve: ", sol["x"], "\n")

    u_des = np.array([sol["x"][0], sol["x"][1]])

    return u_des

# m,x = None,None

# def qp_ini():

#     global m,x

#     m = gp.Model("qp")
#     m.setParam("NonConvex", 2.0)
#     m.setParam("LogToConsole",0)
#     x = m.addVars(2, ub=0.5, lb=-0.5, name="x")

# def controller(time_):

#     obj = (x[0] - u_des[0])**2 + (x[1] - u_des[1])**2
#     m.setObjective(obj)

#     m.remove(m.getConstrs())

#     for i in range (b.size):

#         addCons(i)

#     m.optimize()
#     u_opt = m.getVars()

# def addCons(i):

#     global m

#     m.addConstr(A[i,0]*x[0] + A[i,1]*x[1] <= b[i], "c"+str(i))


# Initial positions of the points
# point1 = np.array([7.5, 15.0])
point1 = np.array([10.0, 10.0])
point2 = np.array([10.0, 10.0])
# point1 = np.array([10.0, 10.0])
# point2 = np.array([1.0, 10.5])
point3 = np.array([10.0, 8.3])

GCM = np.array([10, 10]) 
C = np.array([8.0, 12.0])

x_boundary = [0.0, 20.0]
y_boundary = [0.0, 20.0]
x_bmin, y_bmin = x_boundary[1]/2-1, y_boundary[1]/2-1

# Safety Constraints
dmin = 1.5
theta_max = 30*(np.pi/180)
t = 0.01

# Function to update the plot in each frame
def update(frame):

    global point1, point2, point3, step_size, u_des, t, theta_max
    myobj = datetime.now()

    u1_nominal = np.array([0.2, -0.08])
    # u1_nominal = np.array([10.0, 10.0]) - point1
    u2_nominal = -np.array([0.1, 0.0])
    # u2_nominal = -np.array([-0.3, 0.0])
    u3_nominal = -np.array([0.0, 0.0])

    # Update the positions of the points
    u1_des = 1.0*CBF(point1, point2, point3, GCM, C, u1_nominal, u2_nominal, u3_nominal, dmin, theta_max,\
                    x_boundary, y_boundary, x_bmin, y_bmin) + 0.0*u1_nominal
    point1 += u1_des
    # point2 += u2_nominal
    # point2 = np.array([5*np.cos(t)+10, 5*np.sin(t)+10])
    # t += 0.09
    # theta_max = max(0.995*theta_max, 10*(np.pi/180))

    # Clear the previous plot
    plt.clf()

    # Plot the points
    plt.scatter(point1[0], point1[1], c='red', label='Point 1')
    # plt.scatter(point2[0], point2[1], c='blue', label='Point 2')
    # plt.scatter(point3[0], point3[1], c='green', label='Point 3')
    # plt.scatter(GCM[0], GCM[1], color='black', label='GCM')
    # plt.scatter(C[0], C[1], color='green', label='C')

    # Draw a line connecting the two points
    # plt.plot([point1[0], point2[0]], [point1[1], point2[1]], '--', c='black')
    # plt.plot([point1[0], point3[0]], [point1[1], point3[1]], '--', c='black')

    line_t = (C-GCM)/np.linalg.norm(C-GCM)
    # print("line_t: ", line_t)
    R = np.array([[np.cos(theta_max), -np.sin(theta_max)], [np.sin(theta_max), np.cos(theta_max)]])
    line_l = np.matmul(R, line_t)
    # print("line_l: ", line_l)
    R = np.array([[np.cos(-theta_max), -np.sin(-theta_max)], [np.sin(-theta_max), np.cos(-theta_max)]])
    line_r = np.matmul(R, line_t)
    # print("line_r: ", line_r)

    # plt.plot([ GCM[0], GCM[0]+10*line_t[0] ], [ GCM[1], GCM[1]+10*line_t[1] ], c="red")
    # plt.plot([ GCM[0], GCM[0]+10*line_l[0] ], [ GCM[1], GCM[1]+10*line_l[1] ], c="blue")
    # plt.plot([ GCM[0], GCM[0]+10*line_r[0] ], [ GCM[1], GCM[1]+10*line_r[1] ], c="green")
    # plt.scatter(point2[0], point2[1], s = 1500, facecolors='none', edgecolors='black')
    # plt.scatter(point3[0], point3[1], s = 1500, facecolors='none', edgecolors='black')

    # Set plot limits
    plt.xlim(0, 20)
    plt.ylim(0, 20)

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

# Create an animation
animation = FuncAnimation(plt.figure(), update, frames=np.arange(0, 50), interval=100)

# Display the animation
plt.show()