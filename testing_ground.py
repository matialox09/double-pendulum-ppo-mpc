# Testing a trained PPO policy on the double inverted pendulum (10 s, deterministic).
# Testiranje naučene PPO politike na dvostrukom inverznom klatnu (10 s, deterministički).
#
# EN: Loads a checkpoint, runs the policy without random sampling and plots the angles and the force.
# SR: Učitava checkpoint, pokreće politiku bez slučajnog uzorkovanja i crta uglove i silu.

import os
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

l1 = 1
l2 = 1

# EN: which checkpoint to load: ppo_V{VERSION}_checkpoint.pt
# SR: koji checkpoint se učitava: ppo_V{VERSION}_checkpoint.pt
VERSION = 1
CHECKPOINT_PATH = f"ppo_V{VERSION}_checkpoint.pt"
# EN: must match MAX_FORCE used when training this checkpoint (10.0 or 25.0)
# SR: mora da odgovara MAX_FORCE iz treninga ovog checkpointa (10.0 ili 25.0)
max_force = 10.0
# EN: test length: 1000 steps * 0.01 s = 10 s
# SR: dužina testa: 1000 koraka * 0.01 s = 10 s
steps = 1000


# EN: same simulator as in training (semi-implicit Euler)
# SR: isti simulator kao u treningu (semi-implicitni Ojler)
class DoublePendulumEnv:
    def __init__(self):
        self.mk, self.m1, self.m2 = 2, 1, 1
        self.l1, self.l2 = l1, l2
        self.g = 9.81
        self.dt = 0.01
        self.reset()

    def reset(self):
        self.x0, self.theta10 = 0, 0
        # EN: start state: only theta2 is random, uniform in [-0.1, 0.1] rad (no fixed seed, every run differs)
        # SR: početno stanje: samo je theta2 nasumičan, uniformno u [-0.1, 0.1] rad (nema fiksnog seed-a, svako pokretanje je drugačije)
        self.theta20 = np.random.uniform(-0.1, 0.1)
        self.v0, self.o10, self.o20 = 0, 0, 0
        return np.array([self.x0, self.v0, self.theta10, self.o10, self.theta20, self.o20])

    def step(self, F):
        mk, m1, m2 = self.mk, self.m1, self.m2
        l1, l2, g = self.l1, self.l2, self.g
        theta10, theta20 = self.theta10, self.theta20
        o10, o20 = self.o10, self.o20

        # EN: equations of motion M(q) q_ddot = D (Newton-Euler)
        # SR: jednačine kretanja M(q) q_ddot = D (Njutn-Ojler)
        M = np.array([
            [mk+m1+m2, -(l1/2)*np.cos(theta10)*(m1+2*m2), -(l2/2)*m2*np.cos(theta20)],
            [-6*np.cos(theta10)*(m1+2*m2), 4*l1*(m1+3*m2), 6*m2*l2*np.cos(theta10-theta20)],
            [-6*np.cos(theta20), 6*l1*np.cos(theta10-theta20), 4*l2]
        ])
        D = np.array([
            F - l1*np.sin(theta10)*o10**2*(m1/2 + m2) - m2*(l2/2)*np.sin(theta20)*o20**2,
            6*g*np.sin(theta10)*(m1 + 2*m2) - 6*l2*m2*o20**2*np.sin(theta10 - theta20),
            6*(g*np.sin(theta20) + l1*o10**2*np.sin(theta10 - theta20))
        ])
        a0, u10, u20 = np.linalg.solve(M, D)

        # EN: semi-implicit Euler: velocities first, then positions with the new velocities
        # SR: semi-implicitni Ojler: prvo brzine, pa pozicije sa novim brzinama
        self.v0 += a0 * self.dt
        self.o10 += u10 * self.dt
        self.o20 += u20 * self.dt
        self.x0 += self.v0 * self.dt
        self.theta10 += self.o10 * self.dt
        self.theta20 += self.o20 * self.dt

        return np.array([self.x0, self.v0, self.theta10, self.o10, self.theta20, self.o20])


# EN: normalizer with statistics loaded from the checkpoint; it is not updated during testing (frozen)
# SR: normalizator sa statistikom učitanom iz checkpointa; ne ažurira se tokom testiranja (zamrznut)
class RunningNormalizer:
    def __init__(self, dim, epsilon=1e-8):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = epsilon

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# EN: same network architecture as in training
# SR: ista arhitektura mreže kao u treningu
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        a = self.actor_net(state)
        mean = self.actor_mean(a)
        std = torch.exp(self.actor_logstd)

        c = self.critic_net(state)
        value = self.critic(c)
        return Normal(mean, std), value


# EN: load trained weights and normalizer statistics (if the checkpoint exists)
# SR: učitavanje naučenih težina i statistike normalizatora (ako checkpoint postoji)
model = ActorCritic(state_dim=6, action_dim=1)
normalizer = RunningNormalizer(dim=6)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    normalizer.mean = checkpoint.get("norm_mean", normalizer.mean)
    normalizer.var = checkpoint.get("norm_var", normalizer.var)
    normalizer.count = checkpoint.get("norm_count", normalizer.count)
# EN: evaluation mode
# SR: režim evaluacije
model.eval()

env = DoublePendulumEnv()
# EN: run one test episode from a random start state
# SR: jedan test iz nasumičnog početnog stanja
sensors = env.reset()

x_hist, theta1_hist, theta2_hist, F_hist = [], [], [], []

for step in range(steps):
    norm_sensors = normalizer.normalize(sensors)
    state_t = torch.FloatTensor(norm_sensors)

    with torch.no_grad():
        dist, _ = model(state_t)
        # EN: deterministic evaluation: use the mean of the distribution, not a random sample
        # SR: deterministička evaluacija: koristi se srednja vrednost raspodele, a ne slučajan uzorak
        raw_action = dist.mean
        action = torch.tanh(raw_action)

    # EN: scale the normalized action to a real force
    # SR: skaliranje akcije na stvarnu silu
    F = float(action.numpy()[0]) * max_force
    next_sensors = env.step(F)

    x0, v0, theta10, o10, theta20, o20 = next_sensors
    x_hist.append(x0)
    theta1_hist.append(theta10)
    theta2_hist.append(theta20)
    F_hist.append(F)

    sensors = next_sensors

theta1_hist = np.array(theta1_hist)
theta2_hist = np.array(theta2_hist)
# EN: wrap the angles to [-pi, pi] (a falling pendulum then shows jumps at +-pi)
# SR: svođenje uglova na [-pi, pi] (pad klatna se tada vidi kao skokovi na +-pi)
theta1_hist = (theta1_hist + np.pi) % (2*np.pi) - np.pi
theta2_hist = (theta2_hist + np.pi) % (2*np.pi) - np.pi

t = env.dt * np.arange(steps)

plt.figure()
#plt.plot(t, x_hist, label='x (kolica)')
plt.plot(t, theta1_hist, label='theta1')
plt.plot(t, theta2_hist, label='theta2')
plt.xlabel('vreme (s)')
plt.legend()
plt.title('Ponasanje istreniranog modela')


plt.figure()
plt.plot(t, F_hist)
plt.xlabel('vreme (s)')
plt.ylabel('F (sila)')
plt.title('Akcije modela kroz vreme')

plt.show()