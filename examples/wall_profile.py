"""Hard rods in a slit: the density profile at a hard wall, against exact Tonks.

Produces ``docs/wall_profile.png``, the canonical mcax picture. Hard rods are
the case worth plotting first because everything in it is known exactly: the
bulk density the interior must return to, and the contact value rho(0+) = beta P
the profile must extrapolate to at the wall.

    JAX_PLATFORMS=cpu python examples/wall_profile.py

Needs matplotlib (``pip install -e ".[examples]"``).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# nord ink, so the figure reads as part of the page rather than pasted onto it
plt.rcParams.update({"text.color": "#2E3440", "axes.labelcolor": "#2E3440",
                     "xtick.color": "#2E3440", "ytick.color": "#2E3440",
                     "axes.edgecolor": "#4C566A", "font.size": 10})

from mcax import make_spec, burn_and_sample, summary, format_summary, eos

RHO_B = 0.5                       # target bulk density of the reservoir
H = 8.0                           # centre-accessible wall separation

spec = make_spec(d=1, H=H, z_act=float(eos.z_of_rho(1, RHO_B)), slit=True)
print(f"device {jax.devices()[0].platform}: z = {spec.z_act:.4f}, "
      f"Nmax = {spec.Nmax}")

res = burn_and_sample(spec, C=48, seed=7, n_burn=100_000, n_run=600_000,
                      thin=500, nbins=160)

print(format_summary(summary(res.Ns)))
if res.capacity_warning:
    raise SystemExit(f"run is not trustworthy: {res.capacity_warning}")

# Both walls are equivalent, so mirror-averaging halves the counting variance.
rho_sym = 0.5 * (res.rho + res.rho[::-1])
contact = float(onp.polyval(onp.polyfit(res.z[:3], rho_sym[:3], 2), 0.0))
P = eos.p_of_rho(1, RHO_B)
print(f"contact rho(0+) = {contact:.4f}   exact beta P = {P:.4f}   "
      f"rel {abs(contact - P) / P:.4f}")

fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
ax.plot(res.z, res.rho, lw=1.0, alpha=0.45, color="#81A1C1",
        label="mcax, raw")
ax.plot(res.z, rho_sym, lw=1.8, color="#5E81AC", label="mcax, mirror-averaged")
ax.axhline(RHO_B, ls="--", lw=1.0, color="#4C566A",
           label=rf"bulk $\rho_b = {RHO_B}$")
ax.axhline(P, ls=":", lw=1.4, color="#BF616A",
           label=rf"exact contact $\beta P = {P:.3f}$")
ax.plot([0.0], [contact], "o", ms=6, color="#BF616A",
        label=rf"extrapolated $\rho(0^+) = {contact:.3f}$")

ax.set_xlabel(r"$z / \sigma$   (centre coordinate)")
ax.set_ylabel(r"$\rho(z)\,\sigma$")
ax.set_title(r"Hard rods in a slit, $\mu VT$: 48 chains, $6\times10^5$ steps")
ax.set_xlim(0.0, H)
# Headroom above the contact value so the legend has somewhere to sit that is
# not on top of the wall peaks.
ax.set_ylim(0.30, 1.34)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)

out = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "wall_profile.png")
fig.savefig(path, dpi=150)
print(f"wrote {os.path.normpath(path)}")
