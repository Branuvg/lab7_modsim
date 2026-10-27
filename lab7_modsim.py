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

# %% [markdown]
# ## Task 2.3 – Autoescalado como evento de estado localizado
# Se comparan: evento localizado (bisección dentro del paso) y detección ingenua al final de cada Δt (1 y 0.25 días).
# Como el DES solo "ve" c en los instantes de intercambio, se evalúa con H = 1 (base) y con H = 0.25 (ventana fina), y con el
# híbrido completo y la cascada (κW = κR = 0), para separar el efecto de la detección del efecto de la retroalimentación.

# %%
t_cross = brentq(lambda t: sd_exact(t) - theta * 4 * mu / a, 1, 40)   # instante analítico de ρ_pred = θ con c = 4 (sin feedback)
print(f"cruce analitico (SD sin abandono por latencia, W<Wok): t = {t_cross:.5f} d")

versions = [('Evento localizado', dict(detect='event')),
            ('Ingenua Δt = 1 d', dict(detect='naive', dt_naive=1.0)),
            ('Ingenua Δt = 0.25 d', dict(detect='naive', dt_naive=0.25))]
models = [('Híbrido', {}), ('Cascada', dict(kW_=0.0, kR_=0.0))]
NR = 20
cols = ['Modelo', 'H', 'Versión', 'día 1er pedido', 'día llegada', 'Wmax (s)', '±', 'U(100)', '±', 'rmax', '±',
        'días r>5%', '±', 'exceso W (s·d)', '±', 'c final']
rows = []
store = {}
for mname, mkw in models:
    for H_ in (1.0, 0.25):
        for vname, vkw in versions:
            R = [run_hybrid(np.random.default_rng(1000 + i), H=H_, **mkw, **vkw) for i in range(NR)]   # mismas semillas por versión
            store[(mname, H_, vname)] = R
            row = [mname, H_, vname, np.mean([r['orders'][0][0] for r in R]), np.mean([r['arrivals'][0] for r in R])]
            for key in ('Wmax', 'U_end', 'rmax', 'days_r5', 'excess'):
                m, ci = mean_ci([r[key] for r in R]); row += [m, ci]
            row.append(np.mean([r['c_end'] for r in R]))
            rows.append(row)
df23 = pd.DataFrame(rows, columns=cols)
print(df23.round(4).to_string(index=False))
RES['t23'] = dict(t_cross=t_cross, table=df23.values.tolist(), cols=cols)

# trayectoria de una réplica (semilla 1000), cascada con H = 0.25, alrededor del primer pedido
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for (vname, _), col in zip(versions, ['tab:blue', 'tab:orange', 'tab:green']):
    r = store[('Cascada', 0.25, vname)][0]
    ax[0].plot(r['t'], r['U'], color=col, label=vname)
    w = r['wins']
    ax[1].plot(w[:, 0], np.minimum(w[:, 1], 9), color=col, label=vname)
ax[0].axhline(theta * 4 * mu / a, color='gray', ls=':'); ax[0].set_xlim(10, 30); ax[0].set_ylim(8000, 20500)
ax[0].set_xlabel('t (días)'); ax[0].set_ylabel('U(t)'); ax[0].set_title('Cascada, H = 0.25: usuarios'); ax[0].legend(fontsize=8)
ax[1].set_xlim(14, 28); ax[1].set_xlabel('t (días)'); ax[1].set_ylabel('W de la ventana (s)'); ax[1].set_title('Latencia que ve el SD')
plt.tight_layout(); plt.savefig('figs/t23.png', dpi=150); plt.close()

# %% [markdown]
# ### ¿Cuándo hace más daño la detección tardía? (crecimiento q × retraso D)
# Evento localizado vs ingenua Δt = 1 con H = 0.25. T_sat = tiempo libre entre ρ_pred = θ (U = 11 200) y ρ = 1 (U = 16 000) con c = 4:
# si D > T_sat, los servidores llegan cuando la cola ya está saturada.

# %%
def sat_time(q_):
    f = lambda t, lvl: sd_exact(t, q_=q_) - lvl
    return brentq(lambda t: f(t, 16_000), 1, 80) - brentq(lambda t: f(t, 11_200), 1, 80)

grid = []
NG = 10
for q_ in (0.20, 0.30, 0.50):
    for Dl in (3, 5, 8):
        t_c = brentq(lambda t: sd_exact(t, q_=q_) - 11_200, 1, 80)            # instante del cruce de ρ_pred = θ
        row = dict(q=q_, D=Dl, T_sat=sat_time(q_), t_cruce=t_c, retraso_dt1=math.ceil(t_c) - t_c)   # retraso de la ingenua Δt = 1
        for mname, mkw in models:
            for det, dkw in (('ev', dict(detect='event')), ('nv', dict(detect='naive', dt_naive=1.0))):
                R = [run_hybrid(np.random.default_rng(5000 + i), H=0.25, q_=q_, Dl=Dl, **mkw, **dkw) for i in range(NG)]
                for key in ('Wmax', 'rmax', 'days_r5', 'excess', 'U_end'):
                    row[f'{mname}_{det}_{key}'] = np.mean([r[key] for r in R])
        grid.append(row)
dfg = pd.DataFrame(grid)
for mname in ('Híbrido', 'Cascada'):
    for key in ('Wmax', 'rmax', 'days_r5', 'excess'):
        dfg[f'Δ{key} {mname}'] = dfg[f'{mname}_nv_{key}'] - dfg[f'{mname}_ev_{key}']
print(dfg[['q', 'D', 'T_sat', 'Cascada_ev_days_r5', 'Cascada_nv_days_r5', 'Δdays_r5 Cascada', 'Δexcess Cascada',
           'Δexcess Híbrido', 'Δdays_r5 Híbrido']].round(3).to_string(index=False))
RES['t23grid'] = dfg.round(5).to_dict('records')
save_res('task2')
