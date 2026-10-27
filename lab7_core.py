# %% [markdown]
# # Laboratorio 7 – Modelación y Simulación (Grupo 11)
# Híbrido SD + DES para Aula Abierta (usuarios activos con System Dynamics, clúster de servidores con eventos discretos).
# Parámetros del grupo: g = 11 → q = 0.20 + 0.05·((g−1) mod 4) = 0.30 y D = 3 + ((g−1) mod 4) = 5 días.
#
# Este bloque contiene el modelo; los experimentos de cada Task están en las celdas siguientes.

# %%
import heapq, math
import numpy as np
from math import factorial
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --- Parámetros fijos del enunciado ---
p, M, U0 = 0.002, 20_000, 100             # Bass: innovación (p), mercado (M), usuarios iniciales
d0, kW, kR, Wok = 0.003, 1.0, 0.5, 0.75   # abandono base, sensibilidad a W y a r, latencia aceptable (s)
a, mu, K = 0.0005, 2.0, 60                # solicitudes/s por usuario, tasa por servidor (1/0.5 s), capacidad M/M/c/K
theta, c0 = 0.7, 4                        # umbral de autoescalado y servidores iniciales
T_WIN, WARM = 720.0, 120.0                # ventana de hora pico y calentamiento (s)
g = 11                                    # grupo
q = 0.20 + 0.05 * ((g - 1) % 4)           # imitación (boca a boca) = 0.30
D = 3 + ((g - 1) % 4)                     # retraso de entrega de servidores (días) = 5

# %% [markdown]
# ## Teoría de colas M/M/c (sesión de colas)

# %%
def erlang_c(c, A):
    """Probabilidad de esperar C(c, A) de Erlang. Válida para A < c (ρ = A/c < 1)."""
    top = A**c / factorial(c) * c / (c - A)
    return top / (sum(A**k / factorial(k) for k in range(c)) + top)

def mmc_theory(lam, c, mu=mu):
    """Métricas teóricas de M/M/c: W = C/(cμ−λ) + 1/μ, Wq = C/(cμ−λ), ρ = λ/(cμ), L = λW, Lq = λWq."""
    A = lam / mu
    C = erlang_c(c, A)
    Wq = C / (c * mu - lam)
    W = Wq + 1 / mu
    return dict(C=C, Wq=Wq, W=W, rho=lam / (c * mu), L=lam * W, Lq=lam * Wq, busy=lam / mu)

def mmck_blocking(lam, c, Kcap, mu=mu):
    """Probabilidad de bloqueo exacta de M/M/c/K (ecuaciones de balance de nacimiento y muerte): P_K."""
    pn = [1.0]
    for n in range(1, Kcap + 1):
        pn.append(pn[-1] * lam / (mu * min(n, c)))
    return pn[-1] / sum(pn)

def mmck_theory(lam, c, Kcap, mu=mu):
    """M/M/c/K exacta (balance de nacimiento y muerte): bloqueo P_K y latencia de las aceptadas W = L/(λ(1−P_K))."""
    pn = [1.0]
    for n in range(1, Kcap + 1):
        pn.append(pn[-1] * lam / (mu * min(n, c)))
    tot = sum(pn); pn = [x / tot for x in pn]
    L = sum(n * x for n, x in enumerate(pn))
    return dict(PK=pn[-1], W=L / (lam * (1 - pn[-1])), L=L)

# %% [markdown]
# ## Capa SD: usuarios activos
# dU/dt = (p + q·U/M)(M − U) − δU,   δ = δ0 + κW·max(0, W − Wok) + κR·r   (modelo de Bass con abandono, semana 3)

# %%
def sd_rhs(U, W=0.0, r=0.0, q_=None, kW_=None, kR_=None):
    """Lado derecho del SD. Con kW=kR=0 (cascada) el DES no afecta a los usuarios."""
    q_ = q if q_ is None else q_
    kW_ = kW if kW_ is None else kW_
    kR_ = kR if kR_ is None else kR_
    delta = d0 + kW_ * max(0.0, W - Wok) + kR_ * r
    return (p + q_ * U / M) * (M - U) - delta * U

def rk4_step(f, U, h):
    """Un paso de Runge-Kutta de cuarto orden para dU/dt = f(U) (autónoma en la ventana: W y r constantes)."""
    k1 = f(U)
    k2 = f(U + 0.5 * h * k1)
    k3 = f(U + 0.5 * h * k2)
    k4 = f(U + h * k3)
    return U + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def euler_step(f, U, h):
    return U + h * f(U)

def integrate_fixed(f, U_init, t_end, h, step=rk4_step):
    """Integra con paso fijo h hasta t_end (el último paso se acorta si hace falta). Devuelve t, U."""
    ts, Us = [0.0], [U_init]
    t, U = 0.0, U_init
    while t < t_end - 1e-12:
        hh = min(h, t_end - t)
        U = step(f, U, hh); t += hh
        ts.append(t); Us.append(U)
    return np.array(ts), np.array(Us)

# Solución analítica del SD con abandono constante δ0 (Riccati): dU/dt = α + βU − γU², raíces U+ > 0 > U−
def sd_exact(t, U_init=U0, q_=None, delta=d0):
    q_ = q if q_ is None else q_
    al, be, ga = p * M, q_ - p - delta, q_ / M
    sq = math.sqrt(be**2 + 4 * al * ga)
    Up, Um = (be + sq) / (2 * ga), (be - sq) / (2 * ga)
    R0 = (U_init - Up) / (U_init - Um)
    R = R0 * np.exp(-ga * (Up - Um) * np.asarray(t))
    return (Up - Um * R) / (1 - R)
