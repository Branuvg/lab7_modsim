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

# %% [markdown]
# ## Task 2.4 – Verificación de la interfaz SD ↔ DES

# %%
# a) 30 ventanas con λ = 5: número de solicitudes generadas en la parte medida (t ∈ [120, 720) → 600 s)
rng = np.random.default_rng(24)
def counts(n, lam=5.0, c=4):
    out = []
    for _ in range(n):
        res = des_window(lam, c, K, T_WIN, rng)
        out.append(int((res['arr'] >= WARM).sum()))
    return np.array(out)
n30 = counts(30)
mu_hat, var_hat = n30.mean(), n30.var(ddof=1)
exp_n = 5.0 * (T_WIN - WARM)
chi = 29 * var_hat / mu_hat                    # índice de dispersión: (n−1)·s²/x̄ ~ χ²(n−1) si el conteo es Poisson
p_disp = 2 * min(stats.chi2.cdf(chi, 29), stats.chi2.sf(chi, 29))
t_stat, p_mean = stats.ttest_1samp(n30, exp_n)
n1000 = counts(1000)
print(f"30 ventanas: media={mu_hat:.1f} (esperado {exp_n:.0f}), var={var_hat:.1f}, var/media={var_hat / mu_hat:.3f}, "
      f"chi2={chi:.1f}, p_disp={p_disp:.3f}, p_media(t)={p_mean:.3f}")
print(f"1000 ventanas: media={n1000.mean():.2f}, var={n1000.var(ddof=1):.2f}, var/media={n1000.var(ddof=1) / n1000.mean():.3f}")
# tiempos entre llegadas ~ Exp(λ): prueba de Kolmogorov-Smirnov sobre una ventana larga
ia = np.diff(des_window(5.0, 4, K, 20 * T_WIN, rng)['arr'])
ks = stats.kstest(ia, 'expon', args=(0, 1 / 5.0))
print(f"KS tiempos entre llegadas vs Exp(5): D={ks.statistic:.4f}, p={ks.pvalue:.3f}")
RES['t24a'] = dict(n30=n30.tolist(), mean=mu_hat, var=var_hat, expected=exp_n, disp=var_hat / mu_hat, chi2=chi, p_disp=p_disp,
                   p_mean=p_mean, mean1000=n1000.mean(), var1000=n1000.var(ddof=1), ks_D=ks.statistic, ks_p=ks.pvalue)

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.hist(n30, bins=10, density=True, alpha=.6, label='30 ventanas simuladas')
xs = np.arange(n30.min() - 20, n30.max() + 20)
ax.plot(xs, stats.poisson.pmf(xs, exp_n), 'r-', label=f'Poisson(λ·600 = {exp_n:.0f})')
ax.set_xlabel('solicitudes generadas en la parte medida'); ax.legend(); plt.tight_layout(); plt.savefig('figs/t24a.png', dpi=150); plt.close()

# %%
# b) conservación de entidades por ventana: llegaron = atendidas + rechazadas + dentro (al cierre de la ventana, t = 720)
cases = [(2, 4), (4, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (12, 4), (16, 4), (10, 6), (20, 4), (3, 2)]
rows = []
for lam, c in cases:
    res = des_window(float(lam), c, K, T_WIN, rng)
    cs = window_stats(res)['cons']
    suma = cs['completed'] + cs['rejected'] + cs['inside']
    # comprobación independiente de la cota K: replay de eventos para obtener el máximo de solicitudes en el sistema
    ev = np.concatenate([res['arr'][res['acc']], res['dep'][res['acc']]])
    sg = np.concatenate([np.ones(res['acc'].sum()), -np.ones(res['acc'].sum())])
    o = np.lexsort((sg, ev))
    maxN = int(np.cumsum(sg[o]).max())
    rows.append((lam, c, cs['arrived'], cs['completed'], cs['rejected'], cs['inside'], suma, cs['arrived'] - suma, maxN))
df24b = pd.DataFrame(rows, columns=['λ', 'c', 'Llegaron', 'Atendidas', 'Rechazadas', 'Dentro', 'Suma', 'Diferencia', 'Máx. en sistema'])
print(df24b.to_string(index=False))
bad = 0
for _ in range(500):
    lam = rng.uniform(1, 20); c = int(rng.integers(1, 9))
    cs = window_stats(des_window(lam, c, K, T_WIN, rng))['cons']
    bad += cs['arrived'] != cs['completed'] + cs['rejected'] + cs['inside']
print("ventanas aleatorias con balance roto (de 500):", bad)
RES['t24b'] = dict(table=df24b.values.tolist(), broken_of_500=int(bad))

# %%
# c) sesgo de no descartar el calentamiento: sistema vacío al inicio de cada ventana (K = 60, como en el modelo)
NREP2 = 300
rows = []
for lam, c in [(6.0, 4), (7.2, 4), (7.6, 4), (10.0, 4)]:
    th_ = mmck_theory(lam, c, K)
    for warm in (0.0, 120.0, 300.0):
        ss = [window_stats(des_window(lam, c, K, T_WIN, rng), warm=warm) for _ in range(NREP2)]
        mW, cW = mean_ci([s['W'] for s in ss]); mr, cr = mean_ci([s['r'] for s in ss])
        rows.append((lam, c, warm, mW, cW, th_['W'], mr, cr, th_['PK']))
df24c = pd.DataFrame(rows, columns=['λ', 'c', 'calentamiento (s)', 'W sim', '±', 'W teoría M/M/c/K', 'r sim', '± r', 'r teoría'])
print(df24c.round(4).to_string(index=False))
RES['t24c'] = df24c.values.tolist()
save_res('task2')

# %% [markdown]
# ## Task 1.1 – Contraste de las predicciones con los resultados simulados
# Las predicciones se escribieron antes de simular y no se modificaron; aquí solo se comparan con los resultados.
# (a) híbrido contra cascada (κW = κR = 0); (b) día del primer pedido y servidores en el día 100;
# (c) efecto de H = 1 → 7 (comparación preliminar con 20 réplicas; el barrido completo está en el Task 3.1).

# %%
NP = 30
hyb = [run_hybrid(np.random.default_rng(2000 + i)) for i in range(NP)]
cas = [run_hybrid(np.random.default_rng(2000 + i), kW_=0.0, kR_=0.0) for i in range(NP)]
rows = []
for name, R in (('Híbrido', hyb), ('Cascada', cas)):
    row = [name]
    for key in ('U_end', 'Wmax', 'rmax'):
        m, ci = mean_ci([r[key] for r in R]); row += [m, ci]
    row += [np.mean([r['orders'][0][0] for r in R]), np.mean([r['c_end'] for r in R]),
            np.mean([r['U'][80] for r in R]),                     # U en el día 20 (justo antes de que lleguen los servidores)
            np.mean([r['U'][:81].max() for r in R])]              # máximo de U hasta el día 20
    rows.append(row)
df11 = pd.DataFrame(rows, columns=['Modelo', 'U(100)', '±', 'Wmax', '±', 'rmax', '±', 'día 1er pedido', 'c(100)',
                                   'U(día 20)', 'máx U hasta día 20'])
print(df11.round(4).to_string(index=False))
print("c(100) en cada réplica (híbrido):", sorted(set(r['c_end'] for r in hyb)), " segundo pedido (día medio):",
      np.mean([r['orders'][1][0] for r in hyb]))
RES['t11ab'] = df11.values.tolist()

# %%
# (c) H = 1 vs H = 7, con H = 0.25 como referencia; error relativo cuadrático medio de U(t) por réplica pareada (misma semilla)
NH = 20
runsH = {H_: [run_hybrid(np.random.default_rng(3000 + i), H=H_) for i in range(NH)] for H_ in (0.25, 1.0, 7.0)}
ref = runsH[0.25]
rows = []
for H_ in (1.0, 7.0):
    errs = [np.sqrt(np.mean(((r['U'] - rr['U']) / rr['U'])**2)) for r, rr in zip(runsH[H_], ref)]
    row = [H_]
    for vals in ([r['Wmax'] for r in runsH[H_]], [r['rmax'] for r in runsH[H_]], [r['U_end'] for r in runsH[H_]], errs):
        m, ci = mean_ci(vals); row += [m, ci]
    rows.append(row)
df11c = pd.DataFrame(rows, columns=['H', 'Wmax', '±', 'rmax', '±', 'U(100)', '±', 'ECM relativo de U(t) vs H=0.25', '±'])
print(df11c.round(5).to_string(index=False))
RES['t11c'] = df11c.values.tolist()
save_res('task2')
