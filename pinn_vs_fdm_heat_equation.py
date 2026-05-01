import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde

# --- ১. Numerical Method (Finite Difference) ---
def solve_fdm(L, T, nx, nt, alpha):
    dx = L / (nx - 1)
    dt = T / nt
    u = np.zeros((nx, nt + 1))
    x = np.linspace(0, L, nx)
    
    # Initial Condition: u(x,0) = sin(pi*x)
    u[:, 0] = np.sin(np.pi * x)
    
    # FDM Stepping (Explicit)
    for j in range(0, nt):
        for i in range(1, nx - 1):
            u[i, j+1] = u[i, j] + alpha * dt / dx**2 * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    return x, u

# --- ২. PINN Method (DeepXDE) ---
def solve_pinn(L, T):
    def pde(x, u):
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        return du_t - 0.01 * du_xx # alpha = 0.01

    geom = dde.geometry.Interval(0, L)
    timedomain = dde.geometry.TimeDomain(0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2000, num_boundary=100, num_initial=100)
    net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(iterations=10000)
    return model

# --- ৩. এক্সিকিউশন ও তুলনা ---
L, T, alpha = 1.0, 1.0, 0.01
x_fdm, u_fdm = solve_fdm(L, T, nx=50, nt=1000, alpha=alpha)
model_pinn = solve_pinn(L, T)

# PINN প্রেডিকশন
t_test = 0.5 # t=0.5 সেকেন্ডে তুলনা
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
t_flat = np.full_like(x_test, t_test)
xt_test = np.hstack([x_test, t_flat])
u_pinn = model_pinn.predict(xt_test)

# --- ৪. ভিজ্যুয়ালাইজেশন ---
plt.figure(figsize=(10, 5))
plt.plot(x_fdm, u_fdm[:, int(0.5/(1.0/1000))], 'r--', label='Numerical (FDM) at t=0.5')
plt.plot(x_test, u_pinn, 'b-', label='PINN Prediction at t=0.5')
plt.title("Comparison: PINN vs Finite Difference Method")
plt.xlabel("Position (x)")
plt.ylabel("Temperature (u)")
plt.legend()
plt.grid(True)
plt.show()
