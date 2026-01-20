import numpy as np
import matplotlib.pyplot as plt
from numba import njit
Ra = 1000.0
Pr = 0.71

L = 1.0 
t_final = 1.0

N = 25
tol = 1e-5
Nx=N
Ny=N
dx = L/N
dy = L/N
dt = 1.e-4

omega = 1.5

u = np.zeros((N+1, N+2))
v = np.zeros((N+2, N+1))
p = np.zeros((N+2, N+2))

theta = np.zeros((N+2, N+2))
for j in range(0, N+1):
  theta[-1, j] = 2.0 - theta[0, j]
theta_new = np.copy(theta)

u_star = np.copy(u)
v_star = np.copy(v)

t = 0.0

def calcular_ustar(N,u,v,Pr,dx,dy,dt):
    for i in range(1, N):
        for j in range(0, N):
            v_int = 0.25*(v[i,j+1] + v[i-1,j+1] + v[i,j] + v[i-1,j])
            adv = u[i,j]*(u[i+1,j] - u[i-1,j])/(2.0*dx)
            adv += v_int*(u[i,j+1] - u[i,j-1])/(2.0*dy)
            visc = Pr*(u[i+1,j] - 2.0*u[i,j] + u[i-1,j])/((dx**2.0))
            visc += Pr*(u[i,j+1] - 2.0*u[i,j] + u[i,j-1])/((dy**2.0))
            u_star[i,j] = u[i,j] + dt*(-adv + visc)

        for i in range(0, N+1):
            u_star[i,-1] = -u_star[i,0]
            u_star[i,N] = -u_star[i,N-1]

    return u_star

def calcular_vstar(N,u,v,Pr,dx,dy,dt):
    for i in range(0, N):
        for j in range(1, N):
            u_int = 0.25*(u[i+1,j] + u[i,j] + u[i+1,j-1] + u[i,j-1])
            adv = u_int*(v[i+1,j] - v[i-1,j])/(2.0*dx)
            adv += v[i,j]*(v[i,j+1] - v[i,j-1])/(2.0*dy)
            visc = Pr*(v[i+1,j] - 2.0*v[i,j] + v[i-1,j])/((dx**2.0))
            visc += Pr*(v[i,j+1] - 2.0*v[i,j] + v[i,j-1])/((dy**2.0)) 
            theta_int = 0.5*(theta[i,j] + theta[i,j-1])
            emp = Ra*Pr*theta_int
            v_star[i,j] = v[i,j] + dt*(-adv + visc + emp)

    for j in range(0, N+1):
        v_star[-1,j] = -v_star[0,j]
        v_star[N,j] = -v_star[N-1,j]

    return v_star

def calcular_theta(N,theta,u,v,dx,dy,dt):
    for i in range(0, N):
        for j in range(0, N):
            dif = (theta[i+1,j] - 2.0*theta[i,j] + theta[i-1,j])/(dx**2)
            dif += (theta[i,j+1] - 2.0*theta[i,j] + theta[i,j-1])/(dy**2) 
            u_int = 0.5*(u[i+1,j] + u[i,j])
            v_int = 0.5*(v[i,j+1] + v[i,j])
            adv = u_int*(theta[i+1,j] - theta[i-1,j])/(2.0*dx)
            adv += v_int*(theta[i,j+1] - theta[i,j-1])/(2.0*dy)
            theta_new[i,j] = theta[i,j] + dt*(-adv + dif)

    # Paredes verticais.
    for j in range(-1, N+1):
        theta_new[-1, j] = 2.0 - theta_new[0, j]
        theta_new[N, j] = 0.0 - theta_new[N-1, j]

    # Paredes horizontais.
    for i in range(-1, N+1):
        theta_new[i, -1] = theta_new[i, 0]
        theta_new[i, N] = theta_new[i, N-1]

    theta = np.copy(theta_new)
    return theta


def resolver_pressao(p, u_star, v_star, Nx, Ny, dx, dy, dt, omega, tol):
    error = 100

    while error > tol:
        r_max = 0.0
        for i in range(0, Nx):
            for j in range(0, Ny):
                div = (u_star[i+1, j] - u_star[i, j]) / (dt*dx) + (v_star[i, j+1] - v_star[i, j]) / (dt*dy)

                if   i == 0   and j == 0:
                    Lambda = -(1/dx**2 + 1/dy**2)
                    r = div - ((p[i+1, j] - p[i, j]) / dx**2 + (p[i, j+1] - p[i, j]) / dy**2)

                elif i == 0   and j == Ny-1:
                    Lambda = -(1/dx**2 + 1/dy**2)
                    r  = div - ((p[i+1, j] - p[i, j]) / dx**2 + (- p[i, j] + p[i, j-1]) / dy**2)

                elif i == Nx-1 and j == 0:
                    Lambda = -(1/dx**2 + 1/dy**2)
                    r  = div- ((- p[i, j] + p[i-1, j]) / dx**2 + (p[i, j+1] - p[i, j]) / dy**2)

                elif i == Nx-1 and j == Ny-1:
                    Lambda = -(1/dx**2 + 1/dy**2)
                    r  = div -  ((- p[i, j] + p[i-1, j]) / dx**2 + (- p[i, j] + p[i, j-1]) / dy**2)

                elif i == 0 and 0 < j < Ny-1:
                    Lambda = -(1/dx**2 + 2/dy**2)
                    r = div - ((p[i+1, j] - p[i, j]) / dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2)

                elif i == Nx-1 and 0 < j < Ny-1:
                    Lambda = -(1/dx**2 + 2/dy**2)
                    r  = div - ((- p[i, j] + p[i-1, j]) / dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2)

                elif j == 0 and 0 < i < Ny-1:
                    Lambda = -(2/dx**2 + 1/dy**2)
                    r  = div - ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 + (p[i, j+1] - p[i, j]) / dy**2)

                elif j == Nx-1 and 0 < i < Ny-1:
                    Lambda = -(2/dx**2 + 1/dy**2)
                    r  = div - ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 + (- p[i, j] + p[i, j-1]) / dy**2)

                else:
                    Lambda = -(2/dx**2 + 2/dy**2)
                    r = div -  ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2)

                r = r / Lambda
                p[i, j] += omega * r

                if abs(r) > r_max:
                    r_max = abs(r)

        error = r_max

# Boundary Update

    for i in range(Nx):
        p[i, -1] = p[i,  0]
        p[i,  Ny] = p[i, Ny-1]
    for j in range(Ny):
        p[-1, j] = p[ 0, j]
        p[Nx,  j] = p[Nx-1, j]

    p[-1, -1] = p[0, 0]
    p[-1,  Ny] = p[0, Ny-1]
    p[Nx,  -1] = p[Nx-1, 0]
    p[Nx,   Ny] = p[Nx-1, Ny-1]

    return p


def corrigir_u(u, u_star, p, Nx, Ny, dx, dt):
    for i in range(Nx):
        for j in range(Ny+1):
            u[i,j] = u_star[i,j] - dt * (p[i,j] - p[i-1,j])/dx
    return u


def corrigir_v(v, v_star, p, Nx, Ny, dy, dt):
    for i in range(-1,Nx):
        for j in range(1,Ny):
            v[i,j]= v_star[i,j] - dt * (p[i,j] - p[i,j-1])/dy
    return v

for i in range(0, N):
  for j in range(0, N):
    div = (u[i+1,j] - u[i,j])/dx + (v[i,j+1] - v[i,j])/dy
   #print(div)
print("Entrando no loop de tempo")

while t < t_final:
    u_star = calcular_ustar(N,u,v,Pr,dx,dy,dt)
    v_star = calcular_vstar(N,u,v,Pr,dx,dy,dt)
    theta = calcular_theta(N,theta,u,v,dx,dy,dt)
    p =resolver_pressao(p, u_star, v_star, Nx, Ny, dx, dy, dt, omega, tol)
    u =  corrigir_u(u, u_star, p, Nx, Ny, dx, dt)
    v = corrigir_v(v, v_star, p, Nx, Ny, dy, dt)
  # psi = calcular_stream_function(u, v, Nx, Ny, dx, dy, tol, omega)
    t=t+dt
    #print(time.time()-inicio)
#print(psi)

u_plot = np.zeros((N+1, N+1), float)
v_plot = np.zeros((N+1, N+1), float)
for i in range(0, N+1):
    for j in range(0, N+1):
        u_plot[i,j] = 0.5 * (u[i,j] + u[i,j-1])
        v_plot[i,j] = 0.5 * (v[i,j] + v[i-1,j])

theta_plot = np.zeros((N+1, N+1), float)
for i in range(0, N+1):
    for j in range(0, N+1):
        theta_plot[i,j] = 0.25 * (theta[i,j] + theta[i-1,j] + theta[i,j-1] + theta[i-1,j-1])

x = np.linspace(0.0, L, N+1)
y = np.linspace(0.0, L, N+1)
# Plot both on the same figure
plt.figure(figsize=(6,5))

# Filled contour plot of temperature
contour = plt.contourf(x, y, np.transpose(theta_plot), levels=50, cmap='jet')

plt.contourf(x, y, np.transpose(theta_plot), levels=50, cmap='jet')

# Streamlines of the velocity field
plt.streamplot(x, y, np.transpose(u_plot), np.transpose(v_plot), color='white', linewidth=1)

# Optional: add colorbar
plt.colorbar(contour, label='Temperature')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Streamlines over Temperature Field')
plt.tight_layout()
plt.show()
Nu = 0
for j in range(0, N):
  Nu = Nu - (theta[0, j] - theta[-1, j])*(dy/dx)
  
print(Nu)