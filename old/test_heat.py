from numpy import linspace, zeros, linspace, exp, sin, cos, pi, sqrt
import time
from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt


def ode_FE(f, U_0, dt, T):
    N_t = int(round(float(T) / dt))
    # Ensure that any list/tuple returned from f_ is wrapped as array
    f_ = lambda u, t: asarray(f(u, t))
    u = zeros((N_t + 1, len(U_0)))
    t = linspace(0, N_t * dt, len(u))
    u[0] = U_0
    for n in range(N_t):
        u[n + 1] = u[n] + dt * f_(u[n], t[n])
    return u, t


def U(x, t):
    """True temperature"""
    return A + B * exp(-r * x) * sin(omega * t - r * x)


def rhs(u, t):
    N = len(u) - 1
    rhs = zeros(N + 1)
    rhs[0] = dsdt(t)
    for i in range(1, N):
        rhs[i] = (beta / dx**2) * (u[i + 1] - 2 * u[i] + u[i - 1]) + f(x[i], t)
    i = N
    rhs[i] = (beta / dx**2) * (2 * u[i - 1] - 2 * u[i]) + f(x[N], t)
    return rhs


def dudx(t):
    return 0


def s(t):
    return T0 + Ta * sin((2 * pi / P) * t)


def dsdt(t):
    return (Ta * 2 * pi / P) * cos((2 * pi / P) * t)


def f(x, t):
    return 0


def I(x):
    """Initial temp distribution"""
    return A + B * exp(-r * x) * sin(-r * x)


def I(x):
    """Initial temp distribution."""
    return A


beta = 1e-6
T0 = 283  # just some choice
Ta = 20  # amplitude of temp osc
P = 24 * 60 * 60  # period, 24 hours
A = T0
B = Ta
omega = 2 * pi / P
r = sqrt(omega / (2 * beta))
L = 2  # depth vertically down in the ground
N = 100
x = linspace(0, L, N + 1)
dx = x[1] - x[0]
u = zeros(N + 1)

U_0 = zeros(N + 1)
U_0[0] = s(0)
U_0[1:] = I(x[1:])

dt = dx**2 / float(2 * beta)
print("stability limit:", dt)
T = (24 * 60 * 60) * 6  # simulate 6 days
u, t = ode_FE(rhs, U_0, dt, T)


# Make movie
import os

import matplotlib.pyplot as plt

plt.ion()
lines = plt.plot(x, u[0, :], x, U(x, 0))
plt.axis([x[0], x[-1], 283 - 30, 283 + 30])
plt.xlabel("x")
plt.ylabel("u(x,t) and U(x,t)")
counter = 0
for i in range(0, u.shape[0]):
    lines[0].set_ydata(u[i, :])
    lines[1].set_ydata(U(x, t[i]))
    plt.legend(["t=%.3f" % t[i]])
    plt.draw()
    if i % 10 == 0:  # plot every x steps
        # plt.savefig('tmp_%04d.png' % counter)
        counter += 1
    # time.sleep(0.2)
