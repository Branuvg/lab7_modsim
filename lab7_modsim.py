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

# %% [markdown]
# ## Task 2.1 – Verificación de la capa DES contra la teoría (sin SD)

# %%
NREP = 100                                    # réplicas independientes (≥ 30)
rng = np.random.default_rng(2017)

# a) λ = 6, c = 4, K muy grande: M/M/4 estable (ρ = 0.75)
reps = [window_stats(des_window(6.0, 4, 10**6, T_WIN, rng)) for _ in range(NREP)]
th6 = mmc_theory(6.0, 4)
rows = []
for key, name, teo in [('W', 'Latencia W (s)', th6['W']), ('Wq', 'Espera en cola Wq (s)', th6['Wq']),
                       ('util', 'Utilización ρ', th6['rho'])]:
    m, ci = mean_ci([s[key] for s in reps])
    rows.append((name, teo, m, ci, m - ci <= teo <= m + ci))
df21a = pd.DataFrame(rows, columns=['Métrica', 'Teoría', 'Simulación', 'IC95 ±', 'Teoría dentro del IC'])
print(df21a.to_string(index=False))
RES['t21a'] = df21a.values.tolist()

# b) λ = 10, c = 4, K = 60: sobrecarga
rep_b = [window_stats(des_window(10.0, 4, K, T_WIN, rng)) for _ in range(NREP)]
rb, cib = mean_ci([s['r'] for s in rep_b])
approx = 1 - 4 * mu / 10.0
exact_pk = mmck_blocking(10.0, 4, K)
print(f"r simulada = {rb:.5f} ± {cib:.5f};  1 - c*mu/lam = {approx:.5f};  P_K exacta M/M/4/60 = {exact_pk:.7f}")
print("W de aceptadas en sobrecarga:", mean_ci([s['W'] for s in rep_b]))
RES['t21b'] = dict(r=rb, ci=cib, approx=approx, exact=exact_pk)

# c) Ley de Little con las cantidades de la simulación de (a): L = λ_eff·W, Lq = λ_eff·Wq, servidores ocupados = λ_eff/μ
little = []
for key, wkey, name in [('L', 'W', 'L = λ·W'), ('Lq', 'Wq', 'Lq = λ·Wq'), ('busy', None, 'ocupados = λ/μ')]:
    lhs = np.array([s[key] for s in reps])
    rhs = np.array([s['lam_eff'] * (s[wkey] if wkey else 1 / mu) for s in reps])
    err = np.abs(lhs - rhs) / lhs
    little.append((name, lhs.mean(), rhs.mean(), 100 * (abs(lhs.mean() - rhs.mean()) / lhs.mean()), 100 * err.mean()))
df21c = pd.DataFrame(little, columns=['Relación', 'Lado medido (promedio temporal)', 'Lado Little (λ_eff·tiempo)',
                                      'Error relativo de medias (%)', 'Error relativo medio por réplica (%)'])
print(df21c.to_string(index=False))
RES['t21c'] = df21c.values.tolist()
save_res('task2')

# %% [markdown]
# ## Task 2.2 – Verificación de la capa SD (abandono constante δ0)

# %%
f0 = lambda U: sd_rhs(U, 0.0, 0.0)             # W = r = 0 → δ = δ0
h = H_STEP
t_rk, U_rk = integrate_fixed(f0, U0, 100.0, h, rk4_step)
sol = solve_ivp(lambda t, U: [f0(U[0])], [0, 100], [U0], method='DOP853', rtol=1e-10, atol=1e-10, t_eval=t_rk)
U_ref = sol.y[0]
U_an = sd_exact(t_rk)
e_rk_ivp = np.abs(U_rk - U_ref)
print(f"RK4 (h={h}) vs solve_ivp(1e-10): error max = {e_rk_ivp.max():.3e} usuarios; solve_ivp vs analitica = {np.abs(U_ref - U_an).max():.3e}")
print(f"RK4 vs analitica: {np.abs(U_rk - U_an).max():.3e};  U(100): RK4={U_rk[-1]:.4f}, solve_ivp={U_ref[-1]:.4f}")
RES['t22a'] = dict(h=h, err_rk4_ivp=e_rk_ivp.max(), err_ivp_exact=float(np.abs(U_ref - U_an).max()),
                   U100_rk4=U_rk[-1], U100_ivp=U_ref[-1])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(t_rk, U_ref, 'k-', lw=2, label='solve_ivp (DOP853, tol 1e-10)')
ax[0].plot(t_rk[::8], U_rk[::8], 'o', ms=4, color='tab:red', label=f'RK4 propio (h={h} d)')
ax[0].set_xlabel('t (días)'); ax[0].set_ylabel('U(t)'); ax[0].set_title('Trayectoria con δ = δ0'); ax[0].legend()
ax[1].semilogy(t_rk, np.maximum(e_rk_ivp, 1e-16), color='tab:red')
ax[1].set_xlabel('t (días)'); ax[1].set_ylabel('|U_RK4 − U_solve_ivp|'); ax[1].set_title('Error absoluto del RK4 propio')
plt.tight_layout(); plt.savefig('figs/t22a.png', dpi=150); plt.close()

# b) error máximo contra el paso (referencia: solución analítica)
hs = np.array([4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125])
err_eu, err_rk = [], []
for hh in hs:
    for step, lst in ((euler_step, err_eu), (rk4_step, err_rk)):
        tt, uu = integrate_fixed(f0, U0, 100.0, hh, step)
        lst.append(np.abs(uu - sd_exact(tt)).max())
err_eu, err_rk = np.array(err_eu), np.array(err_rk)
ok = err_rk > 1e-9                                     # evita el piso de redondeo para estimar el orden de RK4
ord_eu = np.polyfit(np.log(hs), np.log(err_eu), 1)[0]
ord_rk = np.polyfit(np.log(hs[ok]), np.log(err_rk[ok]), 1)[0]
print(pd.DataFrame(dict(h=hs, Euler=err_eu, RK4=err_rk)).to_string(index=False))
print(f"orden Euler = {ord_eu:.2f}, orden RK4 = {ord_rk:.2f}")
RES['t22b'] = dict(h=hs.tolist(), euler=err_eu.tolist(), rk4=err_rk.tolist(), ord_euler=ord_eu, ord_rk4=ord_rk)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(hs, err_eu, 'o-', label=f'Euler (pendiente {ord_eu:.2f})')
ax.loglog(hs, err_rk, 's-', label=f'RK4 (pendiente {ord_rk:.2f})')
ax.axvline(H_STEP, color='gray', ls='--', label=f'h elegido = {H_STEP} d')
ax.set_xlabel('paso h (días)'); ax.set_ylabel('error máximo en [0, 100] (usuarios)'); ax.legend(); ax.grid(True, which='both', alpha=.3)
plt.tight_layout(); plt.savefig('figs/t22b.png', dpi=150); plt.close()
save_res('task2')
