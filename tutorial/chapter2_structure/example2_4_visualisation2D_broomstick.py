"""2D representation of a plant using Matplotlib"""

import os

import matplotlib.pyplot as plt  # |\label{l2_1g:importStart}|
import numpy as np
import pandas as pd

import plantbox as pb
from plantbox.structural.MappedOrganism import MappedPlantPython  # |\label{l2_1g:importEnd}|
from plantbox.visualisation import figure_style

sim_time = 28  # [day]; taproot must pass ~24 cm before laterals form  # |\label{l2_1g:defineStart}|
plant = MappedPlantPython()  # |\label{l2_1g:MappedPlantPython}|
script_dir = os.path.dirname(os.path.abspath(__file__))
param_path = os.path.normpath(
    os.path.join(script_dir, "../../modelparameter/structural/plant/broomstick.xml")
)
print(f"Loading parameters: {param_path}")
plant.readParameters(param_path)

# Sanity check: successor rule must define probability or laterals are never created
tap = plant.getOrganRandomParameter(pb.Organism.ot_root, 1)
if not tap.successorST or not tap.successorP:
    raise RuntimeError(
        "Taproot has no successor rule (successorST/successorP empty). "
        "Check <parameter name='successor' ... percentage='1'/> in try1.xml"
    )
print(f"Taproot successor: ST={tap.successorST}, P={tap.successorP}, where={tap.successorWhere}")

soil_domain = pb.SDF_PlantContainer(500, 500, 500, True)  # to avoid root growing aboveground
plant.setGeometry(soil_domain)  # creates soil space to stop roots from growing out of the soil

verbose = False
plant.initialize(verbose)
plant.simulate(sim_time, verbose)  # |\label{l2_1g:defineEnd}|

subtypes = plant.getParameter("subType")
lengths = plant.getParameter("length")
print(f"Root organs: {len(subtypes)} (subTypes={subtypes}, lengths={[round(l, 1) for l in lengths]})")

n_root_polylines = len(plant.getPolylines(pb.root))
print(f"Root polylines (one per root organ): {n_root_polylines}")
if n_root_polylines < 2:
    print(
        "Warning: only taproot found. Increase sim_time or check try1.xml "
        "(la, ln, where, percentage on successor)."
    )

# Export root x-y coordinates (cm) to Excel
root_rows = []
for organ_id, poly in enumerate(plant.getPolylines(pb.root)):
    nodes = plant.toNumpy(poly)
    for node_index, (x, y, _z) in enumerate(nodes):
        root_rows.append(
            {"organ_id": organ_id, "node_index": node_index, "x": x, "y": y}
        )
os.makedirs(os.path.join(script_dir, "results"), exist_ok=True)
pd.DataFrame(root_rows).to_excel(
    os.path.join(script_dir, "results/example_2_4_root_xy.xlsx"), index=False
)

fig, ax = figure_style.subplots11()  # |\label{l2_1g:plotStart}|
organ_name = ["root", "stem", "leaf", "root tips"]
color = ["tab:red", "tab:orange", "tab:green", "tab:blue"]

for idx, ot in enumerate([pb.root, pb.stem, pb.leaf]):
    pl = plant.getPolylines(ot)  # 3D vectors with coordinates of nodes regrouped per organs  |\label{l2_1g:getPolylines}|
    label_added = False
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(pl), 1)))
    for i, node in enumerate(pl):
        node = plant.toNumpy(node)  # 'plantbox.Vector3d' to 2D python array  |\label{l2_1g:toNumpy}|
        c = cmap[i % len(cmap)] if ot == pb.root else color[idx]
        lbl = organ_name[idx] if (not label_added and ot != pb.root) else None
        if ot == pb.root and i == 0:
            lbl = "taproot"
        elif ot == pb.root:
            lbl = f"lateral {i}"
        ax.plot(
            node[:, 0],  # x-axis                                         |\label{l2_1g:plot}|
            node[:, 2],  # z-axis
            c=c,
            linewidth=2 if ot == pb.root else 1.5,
            label=lbl,
        )
        label_added = True

root_tips = plant.get_root_tips()
ax.scatter(root_tips[:, 0], root_tips[:, 2], c=color[3], label=organ_name[3])

ax.legend(bbox_to_anchor=(1, 0.5))
ax.grid(True)
plt.xlabel("X-axis (cm)")
plt.ylabel("Depth (cm)")
ax.relim()
ax.set_aspect("equal", "box")
plt.tight_layout()
out_png = os.path.join(script_dir, "results/example_2_4_2DVisualisation.png")
plt.savefig(out_png)
print(f"Saved {out_png}")
plt.show()  # |\label{l2_1g:plotEnd}|
