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
# ## Capa DES: cola M/M/c/K simulada por eventos discretos
# Las llegadas son Poisson (tiempos entre llegadas exponenciales de tasa λ = a·U) y el servicio es exponencial de media 1/μ.
# Disciplina FIFO con c servidores idénticos: cada solicitud aceptada empieza en max(llegada, primer servidor libre).
# Una solicitud que llega con K solicitudes en el sistema es rechazada.

# %%
def des_window(lam, c, Kcap=K, T=T_WIN, rng=None):
    """Simula una ventana de T segundos con el sistema inicialmente vacío.
    Devuelve, por solicitud: llegada, aceptada (bool), inicio de servicio y salida (la salida puede ser > T: se deja
    drenar el sistema sin nuevas llegadas para no truncar los tiempos en el sistema)."""
    rng = rng or np.random.default_rng()
    if lam <= 0:
        return dict(arr=np.empty(0), acc=np.empty(0, bool), start=np.empty(0), dep=np.empty(0), c=c, T=T)
    n0 = int(lam * T + 6 * math.sqrt(lam * T) + 20)
    arr = np.cumsum(rng.exponential(1 / lam, n0))             # proceso de Poisson: suma de exponenciales
    while arr[-1] < T:
        arr = np.concatenate([arr, arr[-1] + np.cumsum(rng.exponential(1 / lam, n0))])
    arr = arr[arr < T]
    n = len(arr)
    S = rng.exponential(1 / mu, n)                            # tiempos de servicio ~ Exp(μ)
    acc = np.zeros(n, bool); start = np.full(n, np.nan); dep = np.full(n, np.nan)
    free = [0.0] * c                                          # instantes en que se libera cada servidor (heap)
    inside = []                                               # salidas pendientes de los que están en el sistema (heap)
    hpop, hpush = heapq.heappop, heapq.heappush
    for i in range(n):
        t = arr[i]
        while inside and inside[0] <= t:                      # los que ya salieron antes de t
            hpop(inside)
        if len(inside) >= Kcap:                               # sistema lleno: rechazo
            continue
        f = hpop(free)
        s = t if f < t else f                                 # FIFO: empieza cuando haya servidor libre
        d = s + S[i]
        hpush(free, d); hpush(inside, d)
        acc[i], start[i], dep[i] = True, s, d
    return dict(arr=arr, acc=acc, start=start, dep=dep, c=c, T=T)

def _overlap(x0, x1, lo, hi):
    """Longitud de [x0, x1] ∩ [lo, hi] (para promedios en el tiempo sobre la parte medida)."""
    return np.clip(np.minimum(x1, hi) - np.maximum(x0, lo), 0, None)

def window_stats(res, warm=WARM):
    """Métricas de la parte medida [warm, T) de una ventana. Cohorte medida = llegadas con t ≥ warm.
    W = tiempo promedio en el sistema de aceptadas; r = fracción rechazada; el resto sirve para verificar contra la teoría."""
    arr, acc, st, dp, c, T = res['arr'], res['acc'], res['start'], res['dep'], res['c'], res['T']
    meas = arr >= warm
    n_meas = int(meas.sum())
    am = meas & acc
    n_acc = int(am.sum())
    span = T - warm
    out = dict(n_gen=n_meas, n_acc=n_acc, n_rej=n_meas - n_acc)
    out['r'] = (n_meas - n_acc) / n_meas if n_meas else 0.0
    out['W'] = float(np.mean(dp[am] - arr[am])) if n_acc else 1 / mu     # sin datos: solo servicio (1/μ)
    out['Wq'] = float(np.mean(st[am] - arr[am])) if n_acc else 0.0
    out['lam_eff'] = n_acc / span                                         # tasa de llegadas aceptadas
    a_ = acc
    out['L'] = float(_overlap(arr[a_], dp[a_], warm, T).sum() / span)     # promedio temporal de # en el sistema
    out['Lq'] = float(_overlap(arr[a_], st[a_], warm, T).sum() / span)    # ... en la cola
    out['busy'] = float(_overlap(st[a_], dp[a_], warm, T).sum() / span)   # ... en servicio (servidores ocupados)
    out['util'] = out['busy'] / c
    # conservación de entidades en toda la ventana [0, T]: llegaron = atendidas + rechazadas + dentro
    completed = int((acc & (dp <= T)).sum())
    inside = int((acc & (dp > T)).sum())
    out['cons'] = dict(arrived=len(arr), completed=completed, rejected=int((~acc).sum()), inside=inside)
    return out

def des_exchange(lam, c, rng):
    """Interfaz SD→DES→SD: recibe λ y c, devuelve (W, r) de la parte medida de una ventana de hora pico."""
    s = window_stats(des_window(lam, c, K, T_WIN, rng))
    return s['W'], s['r']

def mean_ci(x, conf=0.95):
    """Media e intervalo de confianza t de Student a partir de réplicas independientes."""
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, 0.0
    return m, stats.t.ppf(0.5 + conf / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))

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

H_STEP = 0.25   # paso de integración RK4 (días): elegido en el Task 2.2
