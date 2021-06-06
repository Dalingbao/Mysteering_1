import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.optimal as opt
ct.use_fbs_defaults()
import control.flatsys as fs


# Function to take states, inputs and return the flat flag
def vehicle_flat_forward(x, u, params={}):
    # Get the parameter values
    b = params.get('wheelbase', 3.)

    # Create a list of arrays to store the flat output and its derivatives
    zflag = [np.zeros(3), np.zeros(3)]

    # Flat output is the x, y position of the rear wheels
    zflag[0][0] = x[0]  # x position
    zflag[1][0] = x[1]  # y position

    # First derivatives of the flat output
    zflag[0][1] = u[0] * np.cos(x[2])  # dx/dt  u[0]=v  x[2]=theta
    zflag[1][1] = u[0] * np.sin(x[2])  # dy/dt

    # First derivative of the angle
    thdot = (u[0] / b) * np.tan(u[1])  # thetadot  u[1]=delta

    # Second derivatives of the flat output (setting vdot = 0)
    zflag[0][2] = -u[0] * thdot * np.sin(x[2])  # d2x/dt2
    zflag[1][2] = u[0] * thdot * np.cos(x[2])  # d2y/dt2

    return zflag


# Function to take the flat flag and return states, inputs
def vehicle_flat_reverse(zflag, params={}):
    # Get the parameter values
    b = params.get('wheelbase', 3.)

    # Create a vector to store the state and inputs
    x = np.zeros(3)
    u = np.zeros(2)

    # Given the flat variables, solve for the state
    x[0] = zflag[0][0]  # x position
    x[1] = zflag[1][0]  # y position
    x[2] = np.arctan2(zflag[1][1], zflag[0][1])  # tan(theta) = ydot/xdot        get theta

    # And next solve for the inputs
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])  # get v
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])  # get thetadot*v
    u[1] = np.arctan2(thdot_v, u[0] ** 2 / b)  # get delta

    return x, u


# 创建FlatSystem对象
vehicle_flat = fs.FlatSystem(vehicle_flat_forward, vehicle_flat_reverse, inputs=2, states=3)


# Utility function to plot lane change trajectory
def plot_vehicle_lanechange(traj):
    # Create the trajectory
    t = np.linspace(0, Tf, 100)  # 等差数列 样本数据量100，0-Tf
    x, u = traj.eval(t)  # 评估子函数

    # Configure matplotlib plots to be a bit bigger and optimize layout
    plt.figure(figsize=[9, 4.5])

    # Plot the trajectory in xy coordinate
    plt.subplot(1, 4, 2)
    plt.plot(x[1], x[0])
    plt.xlabel('(a) y [m]')
    plt.ylabel('x [m]')

    # Add lane lines and scale the axis
    plt.plot([-4, -4], [0, x[0, -1]], 'k-', linewidth=1)
    plt.plot([0, 0], [0, x[0, -1]], 'k--', linewidth=1)
    plt.plot([4, 4], [0, x[0, -1]], 'k-', linewidth=1)
    plt.axis([-10, 10, -5, x[0, -1] + 5])  # 轴限制，画大马路用的

    # Time traces of the state and input
    plt.subplot(2, 4, 3)  # 2行4列用索引3
    plt.plot(t, x[1])
    plt.ylabel('y [m]')
    plt.xlabel('(b) Time t [sec]')

    plt.subplot(2, 4, 4)
    plt.plot(t, x[2])
    plt.ylabel('theta [rad]')
    plt.xlabel('(c) Time t [sec]')

    plt.subplot(2, 4, 7)
    plt.plot(t, u[0])
    plt.xlabel('(d) Time t [sec]')
    plt.ylabel('v [m/s]')
    # plt.axis([0, t[-1], u0[0] - 1, uf[0] + 1])

    plt.subplot(2, 4, 8)
    plt.plot(t, u[1])
    plt.xlabel('(e) Time t [sec]')
    plt.ylabel('$r\delta$ [rad]')
    plt.tight_layout()  # 自动调整

#
# To find a trajectory from an initial state $x_0$ to a final state $x_\text{f}$ in time $T_\text{f}$ we solve a
# point-to-point trajectory generation problem.  We also set the initial and final inputs, which sets the vehicle
# velocity $v$ and steering wheel angle $\delta$ at the endpoints.


# Define the endpoints of the trajectory
x0 = [0., 2., 0.]
u0 = [15, 0.]
xf = [75, -2., 0.]
uf = [15, 0.]
Tf = xf[0] / uf[0]

# Define a set of basis functions to use for the trajectories
poly = fs.PolyFamily(10)

# Find a trajectory between the initial condition and the final condition
traj1 = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=poly)
plot_vehicle_lanechange(traj1)


# ## Change of basis function
# ##
bezier = fs.BezierFamily(8)  # 创建阶数8的多项式基础

# Find a trajectory between the initial condition and the final condition
traj2 = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=bezier)
plot_vehicle_lanechange(traj2)


# ###  Added cost function
# ###
timepts = np.linspace(0, Tf, 12)
poly = fs.PolyFamily(8)
# Create quadratic cost function
# Returns a quadratic cost function that can be used for an optimal control problem.
traj_cost = opt.quadratic_cost(vehicle_flat, np.diag([0, 0.1, 0]), np.diag([0.1, 10]), x0=xf, u0=uf)
# 返回由约束类型和参数值组成的元组
constraints = [opt.input_range_constraint(vehicle_flat, [8, -0.1], [12, 0.1])]

# Find a trajectory between the initial condition and the final condition
traj3 = fs.point_to_point(vehicle_flat, timepts, x0, u0, xf, uf, cost=traj_cost, basis=poly)
plot_vehicle_lanechange(traj3)


# show the picture
plt.show()
