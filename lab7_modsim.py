from lab7_core import *   # (en el notebook esta línea no existe: el modelo está en las celdas anteriores)
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# %%
os.makedirs('figs', exist_ok=True); os.makedirs('results', exist_ok=True)
RES = {}                       # resultados numéricos que luego se citan en el informe
def save_res(name):
    with open(f'results/{name}.json', 'w') as fh:
        json.dump(RES, fh, indent=1, default=float)

# %% [markdown]
# ## Task 1.2 – Equilibrio con capacidad fija (c = 4, sin rechazos)
# Con r = 0: W(U) = C(4, aU/μ)/(4μ − aU) + 1/μ  (válida para aU < cμ, es decir U < 16 000).
# Equilibrio: (p + qU/M)(M − U) = [δ0 + κW·max(0, W(U) − Wok)]·U.

# %%
def f_eq(U, c=4):
    """Lado derecho del SD con r = 0 y W = W_mmc(U): adopción neta menos abandono."""
    W = mmc_theory(a * U, c)['W']
    return sd_rhs(U, W, 0.0)

Wok_U = brentq(lambda U: mmc_theory(a * U, 4)['W'] - Wok, 5_000, 15_900)
Ustar = brentq(f_eq, 11_000, 15_900)
th = mmc_theory(a * Ustar, 4)
print(f"U valida para a*U < c*mu -> U < {4 * mu / a:.0f};  W = Wok en U = {Wok_U:.1f}")
print(f"U* = {Ustar:.1f}, rho = {th['rho']:.4f}, W = {th['W']:.4f} s")
print("f(U):", [(u, round(f_eq(u), 1)) for u in (12_000, 13_000)])
print("equilibrio de la cascada (kW=kR=0):", brentq(lambda U: sd_rhs(U, 0, 0, kW_=0, kR_=0), 1_000, M))
