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
sim_time = 14  # days of growth; length and surface are matched at this age
dt = 1

# Shared size at sim_time. Same targets as tutorial/chapter2_structure/day14_match.py.
TARGET_LENGTH_CM = 100.0
TARGET_SURFACE_CM2 = 20.0
_LENGTH_FIELDS = ("lmax", "lmaxs", "la", "las", "lb", "lbs", "ln", "lns", "r", "rs")

architectures = [
    "christmas_tree",
    "droopy_telephone_pole",
    "broomstick_2",
    "telephone_pole",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.normpath(os.path.join(script_dir, "../../modelparameter/structural/rootsystem/"))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Conductivities [kr: 1/day], [kx: cm3/day] — from example6b_xylemflux.py / M3.2 benchmark

KR_MAIN = {"p": 2.36e-05, "m": 2.42e-05, "t": 2.4e-04}  # MP, MM, MT
KR_LAT = {"p": 1.5e-04, "m": 1.52e-04, "t": 2.37e-04}  # LP, LM, LT
KX_MAIN = {"p": 1.85e-06, "m": 2.81e-07, "t": 3.18e-07}  # MP, MM, MT
KX_LAT = {"p": 3.34e-07, "m": 3.73e-07, "t": 7.66e-07}  # LP, LM, LT

#OLD: KR_MAIN = {"p": 8.83e-05, "m": 8.76e-05, "t": 9.69e-05}  # MP, MM, MT
#KR_LAT = {"p": 8.67e-05, "m": 8.51e-05, "t": 8.65e-05}  # LP, LM, LT
#KX_MAIN = {"p": 2.2e-06, "m": 3.44e-07, "t": 3.85e-07}  # MP, MM, MT
#KX_LAT = {"p": 4.18e-07, "m": 4.16e-07, "t": 4.20e-07}  # LP, LM, LT

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
    kr = np.full(n, KR_LAT["p"])
    kx = np.full(n, KX_LAT["p"])

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


def _total_length_and_surface(plant):
    """Same totals the summary prints: sum of segment length and surface."""
    total_length = float(np.sum(np.array(plant.getParameter("length"))))
    total_surface = float(np.sum(np.array(plant.getParameter("surface"))))
    return total_length, total_surface


def _scale_root_parameters(plant, length_scale, radius_scale):
    """Scale elongation together so emergence times stay put, then scale radii."""
    for param in plant.getOrganRandomParameter(pb.root):
        if param.subType == 0 or not hasattr(param, "lmax"):
            continue
        for name in _LENGTH_FIELDS:
            setattr(param, name, getattr(param, name) * length_scale)
        param.a *= radius_scale
        param.a_s *= radius_scale


def match_length_and_surface(plant, name):
    """Grow to sim_time, then scale this architecture onto the shared targets.

    Leaves the plant at the end of the calibration run. Call initialize()
    again before the conductance loop.
    """
    for round_index in range(3):
        plant.setSeed(1)
        plant.initialize(False)
        for _ in range(sim_time):
            plant.simulate(dt)
        length, surface = _total_length_and_surface(plant)
        print(
            f"  {name} calibration {round_index + 1}: "
            f"length {length:.4f} cm, surface {surface:.4f} cm2"
        )
        if length <= 0.0 or surface <= 0.0:
            raise RuntimeError(f"{name} has no root length or surface to match.")
        length_scale = TARGET_LENGTH_CM / length
        radius_scale = TARGET_SURFACE_CM2 / (length_scale * surface)
        if abs(length_scale - 1.0) < 1e-3 and abs(radius_scale - 1.0) < 1e-3:
            break
        _scale_root_parameters(plant, length_scale, radius_scale)
        print(
            f"  applied length scale {length_scale:.6f}, "
            f"radius scale {radius_scale:.6f}"
        )


def print_krs_table(architectures, krs_all, sim_time):
    """Print Krs vs day — same values as plotted in the graphs."""
    col_width = max(12, max(len(name) for name in architectures))
    header = f"{'day':>5}  " + "  ".join(f"{name:>{col_width}s}" for name in architectures)
    sep = "-" * len(header)

    print("\nKrs (cm2 day-1) vs root system age:")
    print(header)
    print(sep)
    for t in range(sim_time):
        values = "  ".join(f"{krs_all[i][t]:>{col_width}.6e}" for i in range(len(architectures)))
        print(f"{t:5d}  {values}")


# Simulation loop
krs_all = []
lengths = []
surfaces = []
csv_data = []

for name in architectures:
    print(f"\nSimulating: {name}")

    plant = pb.MappedPlant()
    plant.readParameters(os.path.join(path, name + ".xml"))
    match_length_and_surface(plant, name)
    plant.setSeed(1)
    plant.initialize()

    params = PlantHydraulicParameters()
    hm = HydraulicModel_Meunier(plant, params)

    krs_values = []
    arch_lengths = []
    arch_surfaces = []

    for t in range(0, sim_time):
        plant.simulate(dt)
        apply_six_zone_conductivities(hm, plant)
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

print_krs_table(architectures, krs_all, sim_time)

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
