"""Whole root system conductance (Krs) for different root architectures

Radial and axial conductivities follow six segment classes (example6b / M3.2 benchmark):
  Main axis:  proximal (P), middle (M), tip (T)
  Lateral:    proximal (P), middle (M), tip (T)

Zones are assigned per root organ from distance to the tip (thirds along each organ).
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import plantbox as pb
from plantbox.functional.PlantHydraulicModel import HydraulicModel_Meunier
from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters
from plantbox.visualisation import figure_style

# Simulation parameters
sim_time = 21  # simulate from day 0 to sim_time - 1
dt = 1

architectures = [
    "christmas_tree",
    "droopy_telephone_pole",
    "telephone_pole",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.normpath(os.path.join(script_dir, "../../modelparameter/structural/rootsystem/"))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Conductivities [kr: 1/day], [kx: cm3/day] — from example6b_xylemflux.py / M3.2 benchmark
KR_MAIN = {"p": 8.83e-05, "m": 8.76e-05, "t": 9.69e-05}  # MP, MM, MT
KR_LAT = {"p": 8.67e-05, "m": 8.51e-05, "t": 8.65e-05}  # LP, LM, LT
KX_MAIN = {"p": 2.2e-06, "m": 3.44e-07, "t": 3.85e-07}  # MP, MM, MT
KX_LAT = {"p": 4.18e-07, "m": 4.16e-07, "t": 4.20e-07}  # LP, LM, LT

# subType groups (adjust if your XML uses different numbering)
MAIN_SUBTYPES = {1, 4}   # taproot / main axes
LATERAL_SUBTYPES = {2, 3}  # 1st- and 2nd-order laterals


def zone_from_tip_fraction(rel_dist_from_tip):
    """rel_dist_from_tip in [0, 1]: 0 = tip, 1 = base of organ."""
    if rel_dist_from_tip < 1.0 / 3.0:
        return "t"
    if rel_dist_from_tip < 2.0 / 3.0:
        return "m"
    return "p"


def apply_six_zone_conductivities(hm, plant, main_subtypes=None, lateral_subtypes=None):
    """Set per-segment kr/kx on hm.params from six-zone conductivities."""
    if main_subtypes is None:
        main_subtypes = MAIN_SUBTYPES
    if lateral_subtypes is None:
        lateral_subtypes = LATERAL_SUBTYPES

    ms = hm.ms
    ms.calcExchangeZoneCoefs()

    n = ms.getNumberOfMappedSegments()
    origins = plant.getSegmentOrigins(-1)
    # MappedPlant may insert one extra collar segment (extraNode) before real organs
    shift = n - len(origins)
    if shift < 0:
        raise RuntimeError(
            f"More segment origins ({len(origins)}) than mapped segments ({n})"
        )

    ot_root = int(pb.OrganTypes.root)
    kr = np.zeros(n)
    kx = np.zeros(n)

    # Collar / seed link segment(s) — not in getSegmentOrigins()
    for si in range(shift):
        kr[si] = KR_MAIN["p"]
        kx[si] = KX_MAIN["p"]

    organ_segs = defaultdict(list)
    for si in range(shift, n):
        organ_segs[origins[si - shift].getId()].append(si)

    for seg_indices in organ_segs.values():
        if ms.organTypes[seg_indices[0]] != ot_root:
            continue

        st = ms.subTypes[seg_indices[0]]
        if st in main_subtypes:
            kr_map, kx_map = KR_MAIN, KX_MAIN
        elif st in lateral_subtypes:
            kr_map, kx_map = KR_LAT, KX_LAT
        else:
            # other root subTypes: treat as lateral
            kr_map, kx_map = KR_LAT, KX_LAT

        max_d = max(ms.distanceTip[si] for si in seg_indices)
        if max_d <= 0:
            max_d = 1.0

        for si in seg_indices:
            rel = ms.distanceTip[si] / max_d
            zone = zone_from_tip_fraction(rel)
            kr[si] = kr_map[zone]
            kx[si] = kx_map[zone]

    hm.params.setKrValues(kr.tolist())
    hm.params.setKxValues(kx.tolist())


# Simulation loop
krs_all = []
lengths = []
surfaces = []
csv_data = []

for name in architectures:
    print(f"\nSimulating: {name}")

    plant = pb.MappedPlant()
    plant.readParameters(os.path.join(path, name + ".xml"))
    plant.initialize()

    params = PlantHydraulicParameters()
    hm = HydraulicModel_Meunier(plant, params)

    krs_values = []
    arch_lengths = []
    arch_surfaces = []

    for t in range(0, sim_time):
        plant.simulate(dt)
        apply_six_zone_conductivities(hm, plant)
        hm.update(t)
        krs, _ = hm.get_krs(t)
        krs_values.append(krs)

        total_length = np.sum(np.array(plant.getParameter("length")))
        total_surface = np.sum(np.array(plant.getParameter("surface")))
        arch_lengths.append(total_length)
        arch_surfaces.append(total_surface)

        csv_data.append([name, t, krs, total_length, total_surface])

    krs_all.append(krs_values)
    lengths.append(arch_lengths[-1])
    surfaces.append(arch_surfaces[-1])

# Plotting
n_arch = len(architectures)
fig, axes = figure_style.subplots12(1, n_arch, sharey=True)

if n_arch == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    ax.plot(range(0, sim_time), krs_all[i])
    ax.set_title(architectures[i])
    ax.set_xlabel("Root system age (day)")
    ax.set_yscale("log")
    if i == 0:
        ax.set_ylabel("Krs (cm$^2$ day$^{-1}$)")
    ax.grid(True)

plt.tight_layout()
plt.show()

print("\nSummary:")
for i, name in enumerate(architectures):
    print(f"{name:20s} | Total root length: {lengths[i]:8.2f} cm | Surface area: {surfaces[i]:8.2f} cm2")

csv_file = os.path.join(results_dir, "krs_length_surface.csv")
with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["architecture", "day", "krs", "length", "surface"])
    writer.writerows(csv_data)

print(f"\nSaved results to: {csv_file}")
