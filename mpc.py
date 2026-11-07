# Linear MPC for stabilizing a double inverted pendulum on a cart.
# Linearni MPC za stabilizaciju dvostrukog inverznog klatna na kolicima.
#
# EN: The model is linearized around the upright position; at every step the optimal force
#     sequence is found by solving a linear system and only the first force is applied.
# SR: Model se linearizuje oko uspravnog položaja; u svakom koraku se niz sila nalazi
#     rešavanjem linearnog sistema, a primenjuje se samo prva sila.

import numpy as np
import matplotlib.pyplot as plt

# ==================== Hiperparametri i konstante ====================

# Fizika okruzenja
# EN: rod lengths (m), masses (kg), gravity (m/s^2), simulation step (s)
# SR: dužine štapova (m), mase (kg), gravitacija (m/s^2), korak simulacije (s)
L1 = 1.0
L2 = 1.0

MK = 2.0
M1 = 1.0
M2 = 1.0

G = 9.81
DT = 0.01

# MPC
STATE_DIM = 6
# EN: maximum force in N; F = u * MAX_FORCE with u in [-1, 1]. Set to 25.0 for the 25 N experiment.
# SR: maksimalna sila u N; F = u * MAX_FORCE, gde je u u [-1, 1]. Za eksperiment sa 25 N postavi 25.0.
MAX_FORCE = 10.0
# EN: number of simulation steps (1000 * 0.01 s = 10 s)
# SR: broj koraka simulacije (1000 * 0.01 s = 10 s)
TimeSteps = 1000
# EN: N = prediction horizon in steps
# SR: N = horizont predviđanja u koracima
N = 60  # Horizon
# EN: state weights Q for [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
# SR: težine stanja Q za [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
Q = np.diag([1.0, 0.1, 10.0, 0.1, 10.0, 0.1])  # Matrica cena
# EN: weight of the force penalty (on the normalized input u)
# SR: težina kazne za silu (na bezdimenzioni ulaz u)
R = 0.05

# Reward (isti kao u PPO)
# EN: reward is used only to report the score; it is the same as in PPO training
# SR: nagrada se koristi samo za ispis rezultata; ista je kao u PPO treningu
LMIN = 1
HEALTHY = 10
W_FORCE = 0.1
W_VEL1 = 0.01
W_VEL2 = 0.05
# ======================================================================


# EN: simulator of the double inverted pendulum on a cart (semi-implicit Euler)
# SR: simulator dvostrukog inverznog klatna na kolicima (semi-implicitni Ojler)
class DoublePendalumEnv:
    def __init__(self):
        self.mk, self.m1, self.m2 = MK, M1, M2
        self.l1, self.l2 = L1, L2
        self.g = G
        self.dt = DT
        self.reset()

    # EN: start state: only theta2 is random, uniform in [-0.1, 0.1] rad
    # SR: početno stanje: samo je theta2 nasumičan, uniformno u [-0.1, 0.1] rad
    def reset(self):
        self.x0 = 0
        self.theta1 = 0
        self.theta20 = np.random.uniform(-0.1, 0.1)
        self.v0, self.omega10, self.omega20 = 0, 0, 0
        return self.get_state()

    # EN: state vector [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    # SR: vektor stanja [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    def get_state(self):
        return np.array([self.x0, self.v0, self.theta1, self.omega10, self.theta20, self.omega20])

    # EN: advances the real simulation by one step using test()
    # SR: pomera pravu simulaciju za jedan korak pomoću test()
    def step(self, u):
        s = self.test(self.get_state(), u)
        self.x0, self.v0, self.theta1, self.omega10, self.theta20, self.omega20 = s
        return s

    # EN: pure function: next state from state s and normalized input u (does not change the environment).
    #     It is also used for the linearization.
    # SR: čista funkcija: sledeće stanje iz stanja s i bezdimenzionog ulaza u (ne menja okruženje).
    #     Koristi se i za linearizaciju.
    def test(self, s, u):
        # EN: scale the normalized input to a real force (the only place where MAX_FORCE is used)
        # SR: skaliranje bezdimenzionog ulaza na stvarnu silu (jedino mesto gde se koristi MAX_FORCE)
        F = u * MAX_FORCE
        mk, m1, m2 = self.mk, self.m1, self.m2
        l1, l2, g = self.l1, self.l2, self.g
        theta1, theta2 = s[2], s[4]
        omega1, omega2 = s[3], s[5]

        # EN: equations of motion M(q) q_ddot = D (Newton-Euler)
        # SR: jednačine kretanja M(q) q_ddot = D (Njutn-Ojler)
        M = np.array([
            [mk + m1 + m2, -(l1/2)*np.cos(theta1)*(m1+2*m2), -(l2/2)*m2*np.cos(theta2)],
            [-6*np.cos(theta1)*(m1+2*m2), 4*l1*(m1+3*m2), 6*m2*l2*np.cos(theta1-theta2)],
            [-6*np.cos(theta2), 6*l1*np.cos(theta1-theta2), 4*l2]
        ])
        D = np.array([
            F - l1*np.sin(theta1)*omega1**2*(m1/2 + m2) - m2*(l2/2)*np.sin(theta2)*omega2**2,
            6*g*np.sin(theta1)*(m1 + 2*m2) - 6*l2*m2*omega2**2*np.sin(theta1 - theta2),
            6*(g*np.sin(theta2) + l1*omega1**2*np.sin(theta1 - theta2))
        ])
        # EN: accelerations of the cart, rod 1 and rod 2
        # SR: ubrzanja kolica, prvog i drugog štapa
        a0, u10, u20 = np.linalg.solve(M, D)

        # EN: semi-implicit Euler: velocities first, then positions with the new velocities
        # SR: semi-implicitni Ojler: prvo brzine, pa pozicije sa novim brzinama
        v = a0 * self.dt + s[1]
        omega1 = u10 * self.dt + s[3]
        omega2 = u20 * self.dt + s[5]
        x0 = s[0] + v * self.dt
        theta1 = s[2] + omega1 * self.dt
        theta20 = s[4] + omega2 * self.dt

        return np.array([x0, v, theta1, omega1, theta20, omega2])


# EN: linearization around the upright position: A and B by central differences of one simulation step
# SR: linearizacija oko uspravnog položaja: A i B centralnom razlikom jednog koraka simulacije
def Matrices(env, eps=1e-6):
    s0 = np.zeros(6)
    A = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        A[:, i] = (env.test(s0 + d, 0.0) - env.test(s0 - d, 0.0)) / (2 * eps)
    B = (env.test(s0, eps) - env.test(s0, -eps)) / (2 * eps)
    return A, B.reshape(6, 1)


# EN: prediction matrices over the horizon: S = Phi * s0 + Gamma * U
# SR: matrice predviđanja na horizontu: S = Phi * s0 + Gamma * U
def horizon_matrices(A, B, N):
    Phi = np.zeros((6*N, 6))
    Gamma = np.zeros((6*N, N))

    Phi[0:6, :] = A
    Gamma[0:6, 0] = B[:, 0]

    for i in range(1, N):
        r = slice(6*i, 6*i + 6)
        p = slice(6*(i-1), 6*i)

        Phi[r, :] = A @ Phi[p, :]
        for j in range(i):
            Gamma[r, j] = A @ Gamma[p, j]
        Gamma[r, i] = B[:, 0]

    return Phi, Gamma


# EN: cost matrices: Qbar = block-diagonal Q, H = Gamma^T Qbar Gamma + R I
# SR: matrice cene: Qbar = blok-dijagonalno Q, H = Gamma^T Qbar Gamma + R I
def cost_matrices(Gamma, N):
    Qbar = np.kron(np.eye(N), Q)
    H = Gamma.T @ Qbar @ Gamma + R * np.eye(N)
    return Qbar, H


# EN: optimal sequence U* = -H^-1 f; only the first force is applied, clipped to [-1, 1]
# SR: optimalni niz U* = -H^-1 f; primenjuje se samo prva sila, odsečena na [-1, 1]
def compute_action(state, Phi, Gamma, Qbar, H):
    x0 = state.reshape(-1, 1)
    f = (Phi @ x0).T @ Qbar @ Gamma
    u = -np.linalg.solve(H, f.T)
    return np.clip(u[0, 0], -1, 1)


# EN: same reward as in PPO training (used only for comparison)
# SR: ista nagrada kao u PPO treningu (koristi se samo za poređenje)
def compute_reward(sensors, u):
    x, x_dot, theta1_rad, theta1_dot, theta2_rad, theta2_dot = sensors

    coordx = x - L1*np.sin(theta1_rad) - L2*np.sin(theta2_rad)
    coordy = L1*np.cos(theta1_rad) + L2*np.cos(theta2_rad)

    distance_penalty = 0.01 * coordx**2 + (coordy - (L1 + L2))**2
    velocity_penalty = W_VEL1 * theta1_dot**2 + W_VEL2 * theta2_dot**2

    terminated = bool(coordy <= LMIN)

    alive_bonus = HEALTHY if not terminated else 0
    reward = alive_bonus - distance_penalty - velocity_penalty
    reward -= W_FORCE * u**2

    return reward, terminated




# EN: build the model and the prediction/cost matrices once (they do not change)
# SR: model i matrice predviđanja i cene računaju se jednom (ne menjaju se)
env = DoublePendalumEnv()
A, B = Matrices(env)
Phi, Gamma = horizon_matrices(A, B, N)
Qbar, H = cost_matrices(Gamma, N)

# EN: closed-loop simulation: compute force, step the system, accumulate reward
# SR: simulacija sa povratnom spregom: izračunaj silu, pomeri sistem, saberi nagradu
state = env.reset()
hist = []
u_hist = []
total = 0.0

for t in range(TimeSteps):
    u = compute_action(state, Phi, Gamma, Qbar, H)
    state = env.step(u)
    r, terminated = compute_reward(state, u)
    total += r
    hist.append(state)
    u_hist.append(u)
    if terminated:
        break

# EN: print results and plot (below)
# SR: ispis rezultata i grafici (ispod)
#Ispisivanje rezultata

hist = np.array(hist)
print("koraka:", len(hist))
print("ukupna nagrada:", total)
print("prosecna po koraku:", total / len(hist), "(maksimum je", HEALTHY, ")")

time = DT * np.arange(len(hist))

plt.figure()
plt.plot(time, hist[:, 0], label='x')
plt.plot(time, hist[:, 2], label='theta1')
plt.plot(time, hist[:, 4], label='theta2')
plt.xlabel('vreme (s)')
plt.legend()
plt.title('MPC')

plt.figure()
plt.plot(time, np.array(u_hist) * MAX_FORCE)
plt.xlabel('vreme (s)')
plt.ylabel('F (N)')
plt.title('Sila')

plt.show()