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





# ## Linearization of lateral steering dynamics (Example 6.13)
# ##用到动态模型
# Define the lateral dynamics as a subset of the full vehicle steering dynamics
lateral = ct.NonlinearIOSystem(
    lambda t, x, u, params: vehicle_update(  # anonymous function
        t, [0., x[0], x[1]], [params.get('velocity', 1), u[0]], params)[1:],  # slice
    lambda t, x, u, params: vehicle_output(
        t, [0., x[0], x[1]], [params.get('velocity', 1), u[0]], params)[1:],
    states=2, name='lateral', inputs=('phi'), outputs=('y')
)

# Compute the linearization at velocity v0 = 15 m/sec
lateral_linearized = ct.linearize(lateral, [0, 0], [0], params=vehicle_params)

# Normalize dynamics using state [x1/b, x2] and timescale v0 t / b
b = vehicle_params['wheelbase']
v0 = vehicle_params['velocity']
lateral_transformed = ct.similarity_transform(
    lateral_linearized, [[1 / b, 0], [0, 1]], timescale=v0 / b)

# Set the output to be the normalized state x1/b
lateral_normalized = lateral_transformed * (1 / b)
print("Linearized system dynamics:\n")
print(lateral_normalized)

# Save the system matrices for later use
A = lateral_normalized.A
B = lateral_normalized.B
C = lateral_normalized.C





# ## Eigenvalue placement controller design (Example 7.4)
# ##自提的ABC
# Utility function to place poles for the normalized vehicle steering system
def normalized_place(wc, zc):
    # Get the dynamics and input matrices, for later use
    A, B = lateral_normalized.A, lateral_normalized.B

    # Compute the eigenvalues from the characteristic polynomial
    eigs = np.roots([1, 2 * zc * wc, wc ** 2])

    # Compute the feedback gain using eigenvalue placement
    K = ct.place_varga(A, B, eigs)

    # Create a new system representing the closed loop response
    clsys = ct.StateSpace(A - B @ K, B, lateral_normalized.C, 0)

    # Compute the feedforward gain based on the zero frequency gain of the closed loop
    kf = np.real(1 / clsys(0))

    # Scale the input by the feedforward gain
    clsys *= kf

    # Return gains and closed loop system dynamics
    return K, kf, clsys


# Utility function to plot simulation results for normalized vehicle steering system
def normalized_plot(t, y, u, inpfig, outfig):
    plt.sca(outfig)
    plt.plot(t, y)
    plt.sca(inpfig)
    plt.plot(t, u[0])


# Utility function to label plots of normalized vehicle steering system 
def normalized_label(inpfig, outfig):
    plt.sca(inpfig)
    plt.xlabel('Normalized time $v_0 t / b$')
    plt.ylabel('Steering angle $\delta$ [rad]')

    plt.sca(outfig)
    plt.ylabel('Lateral position $y/b$')
    plt.plot([0, 20], [0.95, 0.95], 'k--')
    plt.plot([0, 20], [1.05, 1.05], 'k--')


# Configure matplotlib plots to be a bit bigger and optimize layout
plt.figure(figsize=[9, 4.5])

# Explore range of values for omega_c, with zeta_c = 0.7
outfig = plt.subplot(2, 2, 1)
inpfig = plt.subplot(2, 2, 3)
zc = 0.7
for wc in [0.5, 0.7, 1]:
    # Place the poles of the system
    K, kf, clsys = normalized_place(wc, zc)

    # Compute the step response
    t, y, x = ct.step_response(clsys, np.linspace(0, 20, 100), return_x=True)

    # Compute the input used to generate the control response
    u = -K @ x + kf * 1

    # Plot the results
    normalized_plot(t, y, u, inpfig, outfig)

# Add labels to the figure
normalized_label(inpfig, outfig)
plt.legend(('$\omega_c = 0.5$', '$\omega_c = 0.7$', '$\omega_c = 0.1$'))

# Explore range of values for zeta_c, with omega_c = 0.07
outfig = plt.subplot(2, 2, 2)
inpfig = plt.subplot(2, 2, 4)
wc = 0.7
for zc in [0.5, 0.7, 1]:
    # Place the poles of the system
    K, kf, clsys = normalized_place(wc, zc)

    # Compute the step response
    t, y, x = ct.step_response(clsys, np.linspace(0, 20, 100), return_x=True)

    # Compute the input used to generate the control response
    u = -K @ x + kf * 1

    # Plot the results
    normalized_plot(t, y, u, inpfig, outfig)

# Add labels to the figure
normalized_label(inpfig, outfig)
plt.legend(('$\zeta_c = 0.5$', '$\zeta_c = 0.7$', '$\zeta_c = 1$'))  # Place a legend on the Axes.
plt.tight_layout()  # Adjust the padding between and around subplots.





# ## Eigenvalue placement observer design (Example 8.3)
# ##用到ABC
# Find the eigenvalue from the characteristic polynomial
wo = 1  # bandwidth for the observer
zo = 0.7  # damping ratio for the observer
eigs = np.roots([1, 2 * zo * wo, wo ** 2])

# Compute the estimator gain using eigenvalue placement利用特征值布局计算估计增益
L = np.transpose(
    ct.place(np.transpose(A), np.transpose(C), eigs))
print("L = ", L)

# Create a linear model of the lateral dynamics driving the estimator
est = ct.StateSpace(A - L @ C, np.block([[B, L]]), np.eye(2), np.zeros((2, 2)))





# ##  Linear observer applied to nonlinear system output
# ##  用到上面一个的变量
# Convert the curvy trajectory into normalized coordinates
x_ref = x_curvy[0] / wheelbase
y_ref = x_curvy[1] / wheelbase
theta_ref = x_curvy[2]
tau = v0 * T_curvy / b

# Simulate the estimator, with a small initial error in y position
t, y_est, x_est = ct.forced_response(est, tau, [delta_curvy, y_ref], [0.5, 0], return_x=True)

# Configure matplotlib plots to be a bit bigger and optimize layout
plt.figure(figsize=[9, 4.5])

# Plot the actual and estimated states
ax = plt.subplot(2, 2, 1)
plt.plot(t, y_ref)
plt.plot(t, x_est[0])
ax.set(xlim=[0, 10])
plt.legend(['actual', 'estimated'])
plt.ylabel('Lateral position $y/b$')

ax = plt.subplot(2, 2, 2)
plt.plot(t, x_est[0] - y_ref)
ax.set(xlim=[0, 10])
plt.ylabel('Lateral error')

ax = plt.subplot(2, 2, 3)
plt.plot(t, theta_ref)
plt.plot(t, x_est[1])
ax.set(xlim=[0, 10])
plt.xlabel('Normalized time $v_0 t / b$')
plt.ylabel('Vehicle angle $\\theta$')

ax = plt.subplot(2, 2, 4)
plt.plot(t, x_est[1] - theta_ref)
ax.set(xlim=[0, 10])
plt.xlabel('Normalized time $v_0 t / b$')
plt.ylabel('Angle error')
plt.tight_layout()





# ## Output Feedback Controller (Example 8.4)
# ##用到ABC
# Compute the feedback gains
# K, kf, clsys = normalized_place(1, 0.707)     # Gains from MATLAB
# K, kf, clsys = normalized_place(0.07, 0.707)  # Original gains
K, kf, clsys = normalized_place(0.7, 0.707)  # Final gains

# Print out the gains
print("K = ", K)
print("kf = ", kf)

# Construct an output-based controller for the system
clsys = ct.StateSpace(
    np.block([[A, -B @ K], [L @ C, A - B @ K - L @ C]]),
    np.block([[B], [B]]) * kf,
    np.block([[C, np.zeros(C.shape)], [np.zeros(C.shape), C]]),
    np.zeros((2, 1)))

# Simulate the system
t, y, x = ct.forced_response(clsys, tau, y_ref, [0.4, 0, 0.0, 0], return_x=True)

# Calcaluate the input used to generate the control response
u_sfb = kf * y_ref - K @ x[0:2]
u_ofb = kf * y_ref - K @ x[2:4]

# Configure matplotlib plots to be a bit bigger and optimize layout
plt.figure(figsize=[9, 4.5])

# Plot the actual and estimated states
ax = plt.subplot(1, 2, 1)
plt.plot(t, x[0])
plt.plot(t, x[2])
plt.plot(t, y_ref, 'k-.')
ax.set(xlim=[0, 30])
plt.legend(['state feedback', 'output feedback', 'reference'])
plt.xlabel('Normalized time $v_0 t / b$')
plt.ylabel('Lateral position $y/b$')

ax = plt.subplot(2, 2, 2)
plt.plot(t, x[1])
plt.plot(t, x[3])
plt.plot(t, theta_ref, 'k-.')
ax.set(xlim=[0, 15])
plt.ylabel('Vehicle angle $\\theta$')

ax = plt.subplot(2, 2, 4)
plt.plot(t, u_sfb[0])
plt.plot(t, u_ofb[0])
plt.plot(t, delta_curvy, 'k-.')
ax.set(xlim=[0, 15])
plt.xlabel('Normalized time $v_0 t / b$')
plt.ylabel('Steering angle $\\delta$')
plt.tight_layout()

# show the picture
plt.show()