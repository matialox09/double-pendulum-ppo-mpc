# double-pendulum-ppo-mpc
Double inverted pendulum on a cart: PPO and linear MPC (Petnica Research Center project)



# Comparison of PPO and Linear MPC in Stabilizing a Double Inverted Pendulum

Code accompanying the paper "Comparison of PPO and linear MPC in stabilizing a double inverted pendulum" (Petnica Research Center).

We compare the reinforcement learning algorithm PPO, which does not use a model of the system during control, with a linear model predictive controller (MPC), which does. Both output a number in [-1, 1] that is multiplied by the maximum force `MAX_FORCE`, and that value is the only thing we vary (10 N and 25 N).

## Files
- `ppo_training.py` – pendulum simulator and PPO training
- `testing_ground.py` – testing the trained policy (10 s)
- `mpc.py` – linear MPC (same simulator)
- `models/` – trained models for F_max = 10 N and 25 N

## Running
    pip install numpy torch matplotlib
    python ppo_training.py
    python testing_ground.py
    python mpc.py

The maximum force is set by the `MAX_FORCE` variable at the top of each file.

---

# Poređenje PPO-a i linearnog MPC-a u stabilizaciji dvostrukog inverznog klatna

Kod uz rad „Poređenje PPO-a i linearnog MPC-a u stabilizaciji dvostrukog inverznog klatna" (Istraživačka stanica Petnica).

## Fajlovi
- `ppo_training.py` – simulator klatna i treniranje PPO-a
- `testing_ground.py` – testiranje naučene politike (10 s)
- `mpc.py` – linearni MPC (isti simulator)
- `models/` – naučeni modeli za F_max = 10 N i 25 N

## Pokretanje
    pip install numpy torch matplotlib
    python ppo_training.py
    python testing_ground.py
    python mpc.py

Maksimalna sila se zadaje promenljivom `MAX_FORCE` na vrhu svakog fajla.
