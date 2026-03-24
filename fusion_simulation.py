"""
Observer-Based Fusion Plant Simulation
Levitated Dipole Control Framework

Author: Wang Pengyu
Location: Wellington, New Zealand
Date: March 2026

Framework:
    dO = -Wc(O - C_ceil) + alpha*sigma(O)*N - beta*||O||^2*O
          + gamma*H*(O* - O) + delta*grad(Pf) + epsilon*(L-Pe)*grad(eta)

Where:
    O     = state vector (T, P, I, B, F) — normalised
    Wc    = constraint weighting matrix (BB-Closed enforcement)
    N(t)  = structured stochastic novelty (BB-Open — levitated dipole turbulence)
    H     = hardware coupling matrix (Branch A)
    O*    = nominal operating target
    Pf    = fusion power functional
    Pe    = extracted electrical power
    eta   = state-dependent conversion efficiency
    L(t)  = grid load demand signal

Key innovation:
    BB-Open novelty is LEVERAGED, not suppressed.
    For a levitated dipole, magnetospheric-like turbulence is
    a source of stability — the observer handles it as structured
    input, not noise to be filtered out.

Dependencies: numpy, scipy, matplotlib
Install:  pip install numpy matplotlib
Run:      python fusion_simulation.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # change to 'TkAgg' or remove for interactive window
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# PARAMETERS — adjust these to match real experimental data
# ============================================================

dt    = 0.005       # time step
T_end = 10.0        # simulation duration
steps = int(T_end / dt)

# Hard constraints (BB-Closed) — physical limits, normalised to 1.0
C      = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # T, P, I, B, F

# Operating targets (well below hard limits)
O_star = np.array([0.82, 0.72, 0.78, 0.85, 0.68])

# Soft constraint ceiling — observer pulls back here
C_ceil = np.array([0.88, 0.85, 0.88, 0.90, 0.85])

# Initial state — cold start
O = np.array([0.30, 0.25, 0.40, 0.60, 0.50])

# Ignition threshold (normalised temperature)
T_ign = 0.55

# Observer gains
alpha   = 0.4    # novelty response amplitude
beta    = 0.15   # norm damping
gamma   = 1.8    # actuation toward target
delta   = 0.9    # fusion power gradient ascent
epsilon = 0.5    # grid load-following

# Constraint and hardware matrices
Wc = np.diag([3.0, 2.0, 2.0, 2.5, 1.5])   # BB-Closed weighting
H  = np.diag([0.9, 0.8, 0.9, 0.7, 0.8])   # hardware coupling

# Physical constants (normalised)
kappa_f = 1.2    # fusion power coefficient
eta0    = 0.42   # peak conversion efficiency
lam_P   = 1.5    # efficiency pressure sensitivity
lam_T   = 1.2    # efficiency temperature sensitivity
P_opt   = 0.72   # optimal pressure
T_opt   = 0.82   # optimal temperature
Nmax    = 0.025  # BB-Open novelty amplitude (levitated dipole turbulence)
S_min   = 0.50   # minimum survival metric before emergency shutdown

# ============================================================
# GRID LOAD PROFILE
# Ramp up → sustain → reduce
# ============================================================
def grid_load(t):
    if t < 2.0:   return t / 2.0 * 0.8
    elif t < 7.0: return 0.8
    elif t < 9.0: return 0.8 - (t - 7.0) / 2.0 * 0.3
    else:         return 0.5

# ============================================================
# SIMULATION LOOP
# ============================================================
log_t=[]; log_O=[]; log_Pf=[]; log_Pe=[]
log_S=[]; log_L=[]; log_eta=[]
milestone_log=[]; ignited=False; ignition_time=None

for step in range(steps):
    t = step * dt

    # ── BB-Open novelty (levitated dipole turbulence) ────────
    N = np.random.uniform(-Nmax, Nmax, 5)
    N[3] *= 0.25   # B-field: dipole geometry naturally stabilises
    N[0] *= 1.15   # Temperature: most sensitive near ignition

    # ── Fusion reactivity ────────────────────────────────────
    if O[0] < T_ign:
        reactivity = 0.0
    else:
        excess = O[0] - T_ign
        reactivity = excess**1.5 * 4.0

    # ── Fusion power ─────────────────────────────────────────
    V_BP = O[3]*0.6 + O[1]*0.4   # confinement volume (B and P)
    Pf = np.clip(kappa_f * O[4] * (O[2]**1.5) * reactivity * V_BP, 0, 3.0)

    # ── Extraction efficiency (state-dependent) ──────────────
    eta = eta0 * np.exp(-lam_P*(O[1]-P_opt)**2 - lam_T*(O[0]-T_opt)**2)

    # ── Grid load and electrical output ──────────────────────
    L  = grid_load(t)
    Pe = eta * Pf

    # ── Survival metric (BB-Closed) ──────────────────────────
    S = float(np.prod([np.exp(-max(0.0, O[i]-C[i])/C[i]) for i in range(5)]))

    # ── Gradients ────────────────────────────────────────────
    grad_Pf = np.zeros(5)
    if reactivity > 0:
        grad_Pf[0] = kappa_f*O[4]*(O[2]**1.5)*1.5*excess**0.5*4.0*V_BP
        grad_Pf[2] = kappa_f*O[4]*1.5*(O[2]**0.5)*reactivity*V_BP
        grad_Pf[3] = kappa_f*O[4]*(O[2]**1.5)*reactivity*0.6
        grad_Pf[4] = kappa_f*(O[2]**1.5)*reactivity*V_BP

    grad_eta = np.zeros(5)
    grad_eta[0] = -2*lam_T*(O[0]-T_opt)*eta
    grad_eta[1] = -2*lam_P*(O[1]-P_opt)*eta

    # ── Observer dynamics (core equation) ────────────────────
    constraint_excess = np.maximum(0, O - C_ceil)
    constraint_pull   = -Wc @ constraint_excess

    # Hard stop near limit
    for i in range(5):
        if O[i] > C[i]*0.97:
            constraint_pull[i] -= 8.0*(O[i] - C[i]*0.95)

    dO = (constraint_pull
          + alpha * np.tanh(O) * N
          - beta  * np.linalg.norm(O)**2 * O
          + gamma * H @ (O_star - O)
          + delta * grad_Pf
          + epsilon * (L - Pe) * grad_eta)

    # Integrate and apply hard clip
    O = np.clip(O + dt*dO, 0.01, C*0.99)

    # ── Milestone detection ──────────────────────────────────
    if not ignited and O[0] >= T_ign and reactivity > 0:
        ignited = True; ignition_time = t
        milestone_log.append(f"M1 IGNITION         t={t:.2f}s  T={O[0]:.3f}")
    if ignited and Pf > 0.10 and not any('M2' in m for m in milestone_log):
        milestone_log.append(f"M2 SUSTAINED BURN   t={t:.2f}s  Pf={Pf:.3f}")
    if Pe > 0.08 and not any('M3' in m for m in milestone_log):
        milestone_log.append(f"M3 POWER EXTRACTION t={t:.2f}s  Pe={Pe:.3f}  eta={eta*100:.1f}%")
    if abs(Pe-L)<0.06 and Pe>0.3 and not any('M4' in m for m in milestone_log):
        milestone_log.append(f"M4 GRID FOLLOWING   t={t:.2f}s  Pe={Pe:.3f}  L={L:.3f}")

    # Emergency shutdown
    if S < S_min:
        milestone_log.append(f"EMERGENCY SHUTDOWN  t={t:.2f}s  S={S:.3f}")
        break

    if step % 10 == 0:
        log_t.append(t);   log_O.append(O.copy())
        log_Pf.append(Pf); log_Pe.append(Pe)
        log_S.append(S);   log_L.append(L)
        log_eta.append(eta)

log_t  = np.array(log_t);  log_O   = np.array(log_O)
log_Pf = np.array(log_Pf); log_Pe  = np.array(log_Pe)
log_S  = np.array(log_S);  log_L   = np.array(log_L)
log_eta= np.array(log_eta)

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n=== MILESTONES ===")
for m in milestone_log: print(f"  {m}")
if not any('SHUTDOWN' in m for m in milestone_log):
    print("  No emergency shutdown — fully stable ✅")

burn    = log_t > (ignition_time + 0.5) if ignition_time else log_t > 2.0
sustain = (log_t > 2.0) & (log_t < 7.0)
reduce  = log_t > 9.0

print(f"\n=== PERFORMANCE ===")
print(f"  Peak fusion power:          {log_Pf[burn].max():.3f}")
print(f"  Mean fusion power:          {log_Pf[burn].mean():.3f}")
print(f"  Peak electrical output:     {log_Pe[burn].max():.3f}")
print(f"  Mean conversion efficiency: {log_eta[burn].mean()*100:.1f}%")
print(f"  Min survival metric S:      {log_S.min():.4f}  (limit={S_min}) ✅")
print(f"  Load error (sustained):     {np.mean(np.abs(log_Pe[sustain]-0.80)):.3f}")

# ============================================================
# PLOTS
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(15, 14))
fig.suptitle(
    'Observer-Based Fusion Control — Levitated Dipole Simulation\n'
    'Wang Pengyu  |  Wellington, New Zealand  |  March 2026',
    fontsize=12, fontweight='bold')

ax = axes[0,0]
ax.plot(log_t, log_O[:,0], color='#E63946', lw=2.5, label='Temperature T')
ax.axhline(T_ign, color='orange', ls='--', lw=2, label=f'Ignition threshold ({T_ign})')
ax.axhline(C_ceil[0], color='#B45309', ls=':', lw=1.5, alpha=0.8, label='Soft constraint')
ax.axhline(0.99, color='red', ls=':', lw=1, alpha=0.5, label='Hard limit')
if ignition_time:
    ax.axvline(ignition_time, color='gold', lw=2, alpha=0.9,
               label=f'Ignition t={ignition_time:.2f}s')
ax.fill_between(log_t, log_O[:,0], T_ign,
                where=(log_O[:,0]>=T_ign), alpha=0.12, color='red')
ax.set_title('Plasma Temperature — Cold Start to Sustained Burn', fontweight='bold')
ax.set_xlabel('Time (s)'); ax.set_ylabel('T (normalised)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

ax = axes[0,1]
ax.plot(log_t, log_Pf, color='#FF6B35', lw=2.5, label='Fusion power Pf')
ax.plot(log_t, log_Pe, color='#2E86AB', lw=2.5, label='Electrical output Pe')
ax.plot(log_t, log_L,  color='gray',    lw=2, ls='--', label='Grid demand L(t)')
ax.fill_between(log_t, log_Pe, log_L, alpha=0.1, color='blue', label='Load error')
ax.set_title('Power Production & Grid Load-Following', fontweight='bold')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Power (normalised)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[1,0]
for i,(l,c) in enumerate(zip(['T','P','I','B','F'],
    ['#E63946','#2E86AB','#1A5E36','#B45309','#6B21A8'])):
    ax.plot(log_t, log_O[:,i], color=c, lw=1.8, label=l, alpha=0.9)
ax.axhline(0.99, color='red', ls='--', lw=1.5, alpha=0.7, label='Hard limit')
ax.axhline(0.88, color='orange', ls=':', lw=1, alpha=0.5, label='Soft ceiling')
ax.set_ylim(0, 1.08)
ax.set_title('All State Variables — Within Constraints Throughout', fontweight='bold')
ax.set_xlabel('Time (s)'); ax.set_ylabel('State (normalised)')
ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.plot(log_t, log_S, color='#1A5E36', lw=2.5)
ax.axhline(S_min, color='red', ls='--', lw=2, label=f'Shutdown threshold ({S_min})')
ax.fill_between(log_t, log_S, S_min,
                where=(log_S>=S_min), alpha=0.15, color='green',
                label='Safe operating zone')
ax.set_ylim(0.4, 1.05)
ax.set_title('Survival Metric S(t) — Safety Maintained', fontweight='bold')
ax.set_xlabel('Time (s)'); ax.set_ylabel('S(t)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2,0]
ax.plot(log_t, log_eta*100, color='#B45309', lw=2.5)
ax.axhline(eta0*100, color='gray', ls='--', lw=1.5,
           alpha=0.7, label=f'Peak efficiency {eta0*100:.0f}%')
ax.fill_between(log_t, log_eta*100, 0, alpha=0.15, color='#B45309')
ax.set_ylim(0, 50)
ax.set_title('Conversion Efficiency η(t)', fontweight='bold')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Efficiency (%)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2,1]
sc = ax.scatter(log_O[:,0], log_Pf, c=log_t, cmap='plasma', s=10, alpha=0.8)
ax.axvline(T_ign, color='orange', ls='--', lw=2, label='Ignition threshold')
ax.axvline(O_star[0], color='green', ls=':', lw=1.5, alpha=0.7, label='Target T')
plt.colorbar(sc, ax=ax, label='Time (s)')
if ignition_time:
    idx = np.argmin(np.abs(log_t - ignition_time))
    ax.scatter([log_O[idx,0]], [log_Pf[idx]], s=150, color='gold',
               zorder=10, marker='*', label='Ignition point')
ax.set_title('Phase Portrait: T vs Fusion Power\n'
             'Cold start → ignition → sustained burn attractor', fontweight='bold')
ax.set_xlabel('Temperature T'); ax.set_ylabel('Fusion Power Pf')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
plt.savefig('fusion_simulation_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as: fusion_simulation_results.png")

# Show interactive window (comment out if running headless)
# plt.show()
