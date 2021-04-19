import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.optimal as opt
ct.use_fbs_defaults()


# ## Vehicle steering dynamics p102
# ##
def vehicle_update(t, x, u, params):
    # Get the parameters for the model
    a = params.get('refoffset', 1.5)  # offset to vehicle reference point
    b = params.get('wheelbase', 3.)  # vehicle wheelbase
    maxsteer = params.get('maxsteer', 0.5)  # max steering angle (rad)

    # Saturate the steering input
    delta = np.clip(u[1], -maxsteer, maxsteer)  #设置饱和值
    alpha = np.arctan2(a * np.tan(delta), b)  #计算alpha

    # Return the derivative of the state
    return np.array([
        u[0] * np.cos(x[2] + alpha),  # xdot = cos(theta + alpha) v
        u[0] * np.sin(x[2] + alpha),  # ydot = sin(theta + alpha) v
        (u[0] / b) * np.tan(delta)  # thdot = v/b tan(delta)#
    ])


def vehicle_output(t, x, u, params):
    return x[0:2]


# Default vehicle parameters (including nominal velocity)
vehicle_params = {'refoffset': 1.5, 'wheelbase': 3, 'velocity': 15,
                  'maxsteer': 0.5}

# Define the vehicle steering dynamics as an input/output system
vehicle = ct.NonlinearIOSystem(
    vehicle_update, vehicle_output, states=3, name='vehicle',
    inputs=('v', 'delta'), outputs=('x', 'y'), params=vehicle_params)
#


# 在前例的基础上衍生出
import control.flatsys as fs


# another dynamics model p289
# Function to take states, inputs and return the flat flag
def vehicle_flat_forward(x, u, params={}):
    # Get the parameter values
    b = params.get('wheelbase', 3.)

    # Create a list of arrays to store the flat output and its derivatives
    zflag = [np.zeros(3), np.zeros(3)]

    # Flat output is the x, y position of the rear wheels
    zflag[0][0] = x[0]
    zflag[1][0] = x[1]

    # First derivatives of the flat output
    zflag[0][1] = u[0] * np.cos(x[2])  # dx/dt
    zflag[1][1] = u[0] * np.sin(x[2])  # dy/dt

    # First derivative of the angle
    thdot = (u[0] / b) * np.tan(u[1])

    # Second derivatives of the flat output (setting vdot = 0)
    zflag[0][2] = -u[0] * thdot * np.sin(x[2])
    zflag[1][2] = u[0] * thdot * np.cos(x[2])

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
    x[2] = np.arctan2(zflag[1][1], zflag[0][1])  # tan(theta) = ydot/xdot

    # And next solve for the inputs
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])
    u[1] = np.arctan2(thdot_v, u[0] ** 2 / b)

    return x, u


vehicle_flat = fs.FlatSystem(vehicle_flat_forward, vehicle_flat_reverse, inputs=2, states=3)

# In[ ]:


# Utility function to plot lane change trajectory
def plot_vehicle_lanechange(traj):
    # Create the trajectory
    t = np.linspace(0, Tf, 100)
    x, u = traj.eval(t)

    # Configure matplotlib plots to be a bit bigger and optimize layout
    plt.figure(figsize=[9, 4.5])

    # Plot the trajectory in xy coordinate
    plt.subplot(1, 4, 2)
    plt.plot(x[1], x[0])
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')

    # Add lane lines and scale the axis
    plt.plot([-4, -4], [0, x[0, -1]], 'k-', linewidth=1)
    plt.plot([0, 0], [0, x[0, -1]], 'k--', linewidth=1)
    plt.plot([4, 4], [0, x[0, -1]], 'k-', linewidth=1)
    plt.axis([-10, 10, -5, x[0, -1] + 5])

    # Time traces of the state and input
    plt.subplot(2, 4, 3)
    plt.plot(t, x[1])
    plt.ylabel('y [m]')

    plt.subplot(2, 4, 4)
    plt.plot(t, x[2])
    plt.ylabel('theta [rad]')

    plt.subplot(2, 4, 7)
    plt.plot(t, u[0])
    plt.xlabel('Time t [sec]')
    plt.ylabel('v [m/s]')
    # plt.axis([0, t[-1], u0[0] - 1, uf[0] + 1])

    plt.subplot(2, 4, 8)
    plt.plot(t, u[1])
    plt.xlabel('Time t [sec]')
    plt.ylabel('$r\delta$ [rad]')
    plt.tight_layout()

#
# To find a trajectory from an initial state $x_0$ to a final state $x_\text{f}$ in time $T_\text{f}$ we solve a
# point-to-point trajectory generation problem.  We also set the initial and final inputs, which sets the vehicle
# velocity $v$ and steering wheel angle $\delta$ at the endpoints.

# In[ ]:


# Define the endpoints of the trajectory
x0 = [0., 2., 0.]
u0 = [15, 0.]
xf = [75, -2., 0.]
uf = [15, 0.]
Tf = xf[0] / uf[0]

# Define a set of basis functions to use for the trajectories
poly = fs.PolyFamily(8)

# Find a trajectory between the initial condition and the final condition
traj1 = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=poly)
plot_vehicle_lanechange(traj1)

#
#
#

# ## Change of basis function
# ##
bezier = fs.BezierFamily(8)

# Find a trajectory between the initial condition and the final condition
traj2 = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=bezier)
plot_vehicle_lanechange(traj2)

#
#

# ###  Added cost function
# ###
timepts = np.linspace(0, Tf, 12)
poly = fs.PolyFamily(8)
traj_cost = opt.quadratic_cost(vehicle_flat, np.diag([0, 0.1, 0]), np.diag([0.1, 10]), x0=xf, u0=uf)
constraints = [opt.input_range_constraint(vehicle_flat, [8, -0.1], [12, 0.1])]

# Find a trajectory between the initial condition and the final condition
traj3 = fs.point_to_point(vehicle_flat, timepts, x0, u0, xf, uf, cost=traj_cost, basis=poly)
plot_vehicle_lanechange(traj3)


# show the picture
plt.show()
