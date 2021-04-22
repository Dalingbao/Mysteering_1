#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.optimal as opt

ct.use_fbs_defaults()


# ## Vehicle steering dynamics
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





# ## Vehicle driving on a curvy road (Figure 8.6a)
# ##用到动态模型
# System parameters
wheelbase = vehicle_params['wheelbase']  # coming from part1
v0 = vehicle_params['velocity']  # Dictionary parameters membership key

# Control inputs
T_curvy = np.linspace(0, 7, 500)
v_curvy = v0 * np.ones(T_curvy.shape)
delta_curvy = 0.1 * np.sin(T_curvy) * np.cos(4 * T_curvy) + 0.0025 * np.sin(T_curvy * np.pi / 7)
u_curvy = [v_curvy, delta_curvy]
X0_curvy = [0, 0.8, 0]

# Simulate the system + estimator
t_curvy, y_curvy, x_curvy = ct.input_output_response(
    vehicle, T_curvy, u_curvy, X0_curvy, params=vehicle_params, return_x=True)

# Configure matplotlib plots to be a bit bigger and optimize layout
plt.figure(figsize=[9, 4.5])

# Plot the resulting trajectory (and some road boundaries)
plt.subplot(1, 4, 2)  # Add an Axes to the current figure or retrieve an existing Axes
plt.plot(y_curvy[1], y_curvy[0])  # Plot y versus x as lines and/or markers.
plt.plot(y_curvy[1] - 9 / np.cos(x_curvy[2]), y_curvy[0], 'k-', linewidth=1)
plt.plot(y_curvy[1] - 3 / np.cos(x_curvy[2]), y_curvy[0], 'k--', linewidth=1)
plt.plot(y_curvy[1] + 3 / np.cos(x_curvy[2]), y_curvy[0], 'k-', linewidth=1)

plt.xlabel('y [m]')  # Set the label for the x-axis.
plt.ylabel('x [m]')  # Set the label for the y-axis.
plt.axis('Equal')  # Convenience method to get or set some axis properties.

# Plot the lateral position
plt.subplot(2, 2, 2)
plt.plot(t_curvy, y_curvy[1])
plt.ylabel('Lateral position $y$ [m]')

# Plot the steering angle
plt.subplot(2, 2, 4)
plt.plot(t_curvy, delta_curvy)
plt.ylabel('Steering angle $\\delta$ [rad]')
plt.xlabel('Time t [sec]')
plt.tight_layout()

# show plot
plt.show()