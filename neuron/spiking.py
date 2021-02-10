import numpy as np
from neuron.neuron import NeuronModel

class Izhikevich(NeuronModel):
    '''...'''
    def __init__(self, a=0.01, b=0.2, c=-65, d=2, t0=0, v0=0, u0=0, dt=0.1, i0=1):
        # Integration parameters.
        self.t0 = t0
        self.dt = dt

        # Constants.
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        # Variables.
        self.x = np.array([v0, u0])
        self.t = t0
        self.i = i0

        # Containers.
        self.T = np.array([], dtype=np.float64) # Time.
        self.V = np.array([], dtype=np.float64) # Membrane potential.
        self.U = np.array([], dtype=np.float64) # 
        self.I = np.array([], dtype=np.float64) # Current.
        self.S = np.array([], dtype=np.float64) # Spike times.

        # Others.
        self.name = "Izhikevich"

    def _v_dot(self, v, u, t):
        # Depends on v, u; and t (implicit).
        return 0.04*v*v + 5*v + 140 - u + self.i
    
    def _u_dot(self, v, u, t):
        # Depends on v and u.
        return self.a*(self.b*v - u)

    def _x_dot(self, x, t):
        return np.array([
            self._v_dot(*x,t),
            self._u_dot(*x,t)
        ])

    def run(self, n):
        for _ in range(n):
            self.run_step()

    def run_step(self):
        # Set containers if first time.
        if self.T.size == 0:
            self.T = np.append(self.T, self.t)
        if self.V.size == 0:
            self.V = np.append(self.V, self.x[0])
        if self.U.size == 0:
            self.U = np.append(self.U, self.x[1])
        if self.I.size == 0:
            self.I = np.append(self.I, self.i)

        # Test for spike
        if self.x[0] >= 30:
            self.S = np.append(self.S, self.t)
            self.x[0] =  self.c 
            self.x[1] += self.d

        # Integration step
        k1 = self.dt*self._x_dot(self.x     , self.t)
        k2 = self.dt*self._x_dot(self.x + k1, self.t + self.dt)
        self.x = self.x + (k1 + k2)/2

        # Update containers
        self.T = np.append(self.T, self.t)
        self.V = np.append(self.V, self.x[0])
        self.U = np.append(self.U, self.x[1])
        self.I = np.append(self.I, self.i)

        # Update time
        self.t = self.t + self.dt


class HodgkinHuxley(NeuronModel):
    '''...'''
    def __init__(self, t0=0, v0=0, n0=0, m0=0, h0=0, dt=0.1, i0=1):
        # Integration parameters.
        self.t0 = t0
        self.dt = dt

        # Constants, and rate constants.
        self.C_m  =  0.01  # uF/cm^2
        self.E_Na =  55.17 # mV
        self.E_K  = -72.14 # mV
        self.E_l  = -49.42 # mV
        self.g_Na =  1.2   # mS/cm^2
        self.g_K  =  0.36  # mS/cm^2
        self.g_l  =  0.003 # mS/cm^2
        
        self.alpha_n = lambda v: 0.01*(v+50)/(1-np.exp((-v-50)/10))
        self.beta_n  = lambda v: 0.125*np.exp((-v-60)/80)
        self.alpha_m = lambda v: 0.1*(v+35)/(1-np.exp((-v-35)/10))
        self.beta_m  = lambda v: 4.0*np.exp(-0.0556*(v+60))
        self.alpha_h = lambda v: 0.07*np.exp(-0.05*(v+60))
        self.beta_h  = lambda v: 1/(1+np.exp(-0.1*(v+30)))

        # Variables.
        self.x = np.array([v0, n0, m0, h0])
        self.t = t0
        self.i = i0

        # Containers.
        self.T = np.array([], dtype=np.float64) # Time.
        self.V = np.array([], dtype=np.float64) # Membrane potential.
        self.N = np.array([], dtype=np.float64) # 
        self.M = np.array([], dtype=np.float64) # 
        self.H = np.array([], dtype=np.float64) # 
        self.I = np.array([], dtype=np.float64) # Current.
        self.S = np.array([], dtype=np.float64) # Spike times.

        # Others.
        self.name = "Hodgkin-Huxley"

    def _v_dot(self, v, n, m, h, t):
        # Depends on v, n, m, h; and t (implicit).
        i_Na = self.g_Na*m*m*m*h*(v - self.E_Na)
        i_K  = self.g_K*n*n*n*n*(v - self.E_K)
        i_l  = self.g_l*(v - self.E_l)
        return (self.i - i_Na - i_K - i_l)/self.C_m
    
    def _n_dot(self, v, n, m, h, t):
        # Depends on v and n.
        return self.alpha_n(v)*(1-n) - self.beta_n(v)*n

    def _m_dot(self, v, n, m, h, t):
        # Depends on v and m.
        return self.alpha_m(v)*(1-m) - self.beta_m(v)*m

    def _h_dot(self, v, n, m, h, t):
        # Depends on v and h.
        return self.alpha_h(v)*(1-h) - self.beta_h(v)*h

    def _x_dot(self, x, t):
        return np.array([
            self._v_dot(*x,t),
            self._n_dot(*x,t),
            self._m_dot(*x,t),
            self._h_dot(*x,t),
        ])

    def run(self, n):
        for _ in range(n):
            self.run_step()

    def run_step(self):
        # Set containers if first time.
        if self.T.size == 0:
            self.T = np.append(self.T, self.t)
        if self.V.size == 0:
            self.V = np.append(self.V, self.x[0])
        if self.N.size == 0:
            self.N = np.append(self.N, self.x[1])
        if self.M.size == 0:
            self.M = np.append(self.M, self.x[2])
        if self.H.size == 0:
            self.H = np.append(self.H, self.x[3])
        if self.I.size == 0:
            self.I = np.append(self.I, self.i)

        # Test for spike
        #if self.u > self.u_thr:
        #    self.S = np.append(self.S, self.t)
        #    self.u = self.u_rest

        # Integration step
        k1 = self.dt*self._x_dot(self.x     , self.t)
        k2 = self.dt*self._x_dot(self.x + k1, self.t + self.dt)
        self.x = self.x + (k1 + k2)/2

        # Update containers
        self.T = np.append(self.T, self.t)
        self.V = np.append(self.V, self.x[0])
        self.N = np.append(self.N, self.x[1])
        self.M = np.append(self.M, self.x[2])
        self.H = np.append(self.H, self.x[3])
        self.I = np.append(self.I, self.i)

        # Update time
        self.t = self.t + self.dt


class FitzHughNagumo(NeuronModel):
    '''...'''
    def __init__(self, R=1, a=0, b=0, tau=1, t0=0, v0=0, w0=0, dt=0.1, i0=1):
        # Model parameters.
        self.R   = R
        self.a   = a
        self.b   = b
        self.tau = tau
        
        # Integration parameters.
        self.t0 = t0
        self.dt = dt
        
        # Variables.
        self.x = np.array([v0, w0])
        self.t = t0
        self.i = i0

        # Containers.
        self.T = np.array([], dtype=np.float64) # Time
        self.V = np.array([], dtype=np.float64) # Voltage
        self.W = np.array([], dtype=np.float64) # W
        self.I = np.array([], dtype=np.float64) # Current
        self.S = np.array([], dtype=np.float64) # Spike times

        # Other
        self.name = "FitzHugh-Nagumo"

    def _v_dot(self, v, w, t):
        # Depends of v, w; and t (implicit).
        return v - v*v*v/3 - w + self.R*self.i

    def _w_dot(self, v, w, t):
        # Depends of v and w.
        return (v + self.a - self.b*w)/self.tau

    def _x_dot(self, x, t):
        return np.array([
            self._v_dot(*x, t),
            self._w_dot(*x, t)
        ])

    def run(self, n):
        for _ in range(n):
            self.run_step()

    def run_step(self):
        # Set containers if first time.
        if self.T.size == 0:
            self.T = np.append(self.T, self.t)
        if self.V.size == 0:
            self.V = np.append(self.V, self.x[0])
        if self.W.size == 0:
            self.W = np.append(self.W, self.x[1])
        if self.I.size == 0:
            self.I = np.append(self.I, self.i)

        # Test for spike
        #if self.u > self.u_thr:
        #    self.S = np.append(self.S, self.t)
        #    self.u = self.u_rest

        # Integration step
        k1 = self.dt*self._x_dot(self.x     , self.t)
        k2 = self.dt*self._x_dot(self.x + k1, self.t + self.dt)
        self.x = self.x + (k1 + k2)/2
        
        # Update containers
        self.T = np.append(self.T, self.t)
        self.V = np.append(self.V, self.x[0])
        self.W = np.append(self.W, self.x[1])
        self.I = np.append(self.I, self.i)

        # Update time
        self.t = self.t + self.dt


class LeakyIntegrateAndFire(NeuronModel):
    '''...'''
    def __init__(self, R=1, C=1, v0=-1, v_rest=-1, v_thr=0.5, t0=0, dt=0.1, i0=1):
        # Model parameters.
        self.R      = R
        self.C      = C
        self.v_rest = v_rest
        self.v_thr  = v_thr

        # Integration parameters.
        self.t0 = t0
        self.dt = dt

        # Variables. (init. cond.)
        self.t = t0
        self.v = v0
        self.i = i0

        # Containers. (Historic)
        self.T = np.array([], dtype=np.float64) # Time
        self.V = np.array([], dtype=np.float64) # Potential
        self.I = np.array([], dtype=np.float64) # Current
        self.S = np.array([], dtype=np.float64) # Spike times

        # Other
        self.name = "Leaky integrate-and-fire"

    def _v_dot(self, v, t):
        return (self.i - v/self.R)/self.C

    def run(self, n):
        for _ in range(n):
            self.run_step()

    def run_step(self):
        # Set containers if first time.
        if self.T.size == 0:
            self.T = np.append(self.T, self.t)
        if self.V.size == 0:
            self.V = np.append(self.V, self.v)
        if self.I.size == 0:
            self.I = np.append(self.I, self.i)

        # Test for spike
        if self.v > self.v_thr:
            self.S = np.append(self.S, self.t)
            self.v = self.v_rest

        # Integration step
        k1 = self.dt*self._v_dot(self.v     , self.t)
        k2 = self.dt*self._v_dot(self.v + k1, self.t + self.dt)
        self.v = self.v + (k1 + k2)/2

        # Update containers
        self.T = np.append(self.T, self.t)
        self.V = np.append(self.V, self.v)
        self.I = np.append(self.I, self.i)

        # Update time
        self.t = self.t + self.dt


class IntegrateAndFire(NeuronModel):
    '''...'''
    def __init__(self, C=1, v0=-1, v_rest=-1, v_thr=0.5, t0=0, dt=0.1, i0=1):
        # Model parameters.
        self.C      = C
        self.v_rest = v_rest
        self.v_thr  = v_thr

        # Integration parameters.
        self.t0 = t0
        self.dt = dt

        # Variables. (init. cond.)
        self.t = t0
        self.v = v0
        self.i = i0

        # Containers (Historic)
        self.T = np.array([], dtype=np.float64) # Time
        self.V = np.array([], dtype=np.float64) # Potential
        self.I = np.array([], dtype=np.float64) # Current
        self.S = np.array([], dtype=np.float64) # Spike times

        # Other
        self.name = "Integrate-and-fire"

    def _v_dot(self, v, t):
        return self.i/self.C

    def run(self, n):
        for _ in range(n):
            self.run_step()

    def run_step(self):
        # Set containers if first time.
        if self.T.size == 0:
            self.T = np.append(self.T, self.t)
        if self.V.size == 0:
            self.V = np.append(self.V, self.v)
        if self.I.size == 0:
            self.I = np.append(self.I, self.i)

        # Test for spike
        if self.v > self.v_thr:
            self.S = np.append(self.S, self.t)
            self.v = self.v_rest

        # Integration step
        k1 = self.dt*self._v_dot(self.v     , self.t)
        k2 = self.dt*self._v_dot(self.v + k1, self.t + self.dt)
        self.v = self.v + (k1 + k2)/2

        # Update containers
        self.T = np.append(self.T, self.t)
        self.V = np.append(self.V, self.v)
        self.I = np.append(self.I, self.i)

        # Update time
        self.t = self.t + self.dt


Izh = Izhikevich
HH  = HodgkinHuxley
FHN = FitzHughNagumo
LIF = LeakyIntegrateAndFire
IF  = IntegrateAndFire