def runge_kutta_2_step(t, dt, u, u_dot):
        k1 = dt*u_dot(t, u)
        k2 = dt*u_dot(t + dt, u + k1)

        return u + (k1+k2)/2

def runge_kutta_4_step(t, dt, u, u_dot):
        k1 = dt*u_dot(t, u)
        k2 = dt*u_dot(t + dt/2, u + k1/2)
        k3 = dt*u_dot(t + dt/2, u + k2/2)
        k4 = dt*u_dot(t + dt,   u + k3)

        return u + (k1 + 2*k2 + 2*k3 + k4)/6

RK2_step = runge_kutta_2_step
RK4_step = runge_kutta_4_step