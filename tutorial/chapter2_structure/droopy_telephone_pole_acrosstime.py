"""2D representation of a plant using Matplotlib"""

import os

import matplotlib.pyplot as plt  # |\label{l2_1g:importStart}|
import pandas as pd

import plantbox as pb
from plantbox.structural.MappedOrganism import MappedPlantPython  # |\label{l2_1g:importEnd}|
from plantbox.visualisation import figure_style

snapshot_days = [7, 14, 21]  # cumulative simulation times [day]  # |\label{l2_1g:defineStart}|
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Populated during the run — available after the script finishes (or on import)
figures = {}
root_coords = {}


def collect_root_xz(plant, sim_day):
    rows = []
    for organ_id, poly in enumerate(plant.getPolylines(pb.root)):
        nodes = plant.toNumpy(poly)
        for node_index, (x, _y, z) in enumerate(nodes):
            rows.append(
                {
                    "sim_day": sim_day,
                    "organ_id": organ_id,
                    "node_index": node_index,
                    "x": x,
                    "z": z,
                }
            )
    return pd.DataFrame(rows)


def print_root_table(root_df, sim_day):
    print(f"\n2D root coordinates (cm) at day {sim_day} — x and z:\n")
    with pd.option_context(
        "display.max_rows", None, "display.width", 100, "display.float_format", "{:.4f}".format
    ):
        print(root_df[["organ_id", "node_index", "x", "z"]].to_string(index=False))
    print()


def plot_roots_2d(plant, sim_day):
    fig, ax = figure_style.subplots11()  # |\label{l2_1g:plotStart}|
    organ_name = ["root", "stem", "leaf", "root tips"]
    color = ["tab:red", "tab:orange", "tab:green", "tab:blue"]

    for idx, ot in enumerate([pb.root, pb.stem, pb.leaf]):
        pl = plant.getPolylines(ot)  # |\label{l2_1g:getPolylines}|
        label_added = False
        for node in pl:
            node = plant.toNumpy(node)  # |\label{l2_1g:toNumpy}|
            ax.plot(
                node[:, 0],  # x-axis  |\label{l2_1g:plot}|
                node[:, 2],  # z-axis
                c=color[idx],
                label=organ_name[idx] if not label_added else None,
            )
            label_added = True

    root_tips = plant.get_root_tips()
    ax.scatter(root_tips[:, 0], root_tips[:, 2], c=color[3], label=organ_name[3])

    ax.legend(bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    ax.set_title(f"Day {sim_day}")
    plt.xlabel("X-axis (cm)")
    plt.ylabel("z (cm)")
    ax.relim()
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    return fig, ax


plant = MappedPlantPython()  # |\label{l2_1g:MappedPlantPython}|
path = os.path.join(script_dir, "../../modelparameter/structural/plant/")
filename = "droopy_telephone_pole"
plant.readParameters(os.path.normpath(os.path.join(path, filename + ".xml")))

soil_domain = pb.SDF_PlantContainer(500, 500, 500, True)
plant.setGeometry(soil_domain)

verbose = False
plant.initialize(verbose)

prev_day = 0
all_root_rows = []
for sim_day in snapshot_days:
    dt = sim_day - prev_day
    plant.simulate(dt, verbose)  # |\label{l2_1g:defineEnd}|
    prev_day = sim_day

    root_df = collect_root_xz(plant, sim_day)
    root_coords[sim_day] = root_df
    all_root_rows.append(root_df)
    print_root_table(root_df, sim_day)

    fig, _ax = plot_roots_2d(plant, sim_day)
    figures[sim_day] = fig
    out_png = os.path.join(results_dir, f"example_2_4_2DVisualisation_day{sim_day}.png")
    fig.savefig(out_png)
    print(f"Saved {out_png}")

combined_df = pd.concat(all_root_rows, ignore_index=True)
excel_path = os.path.join(results_dir, "example_2_4_root_xz.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    combined_df.to_excel(writer, sheet_name="all_days", index=False)
    for sim_day in snapshot_days:
        root_coords[sim_day].to_excel(writer, sheet_name=f"day_{sim_day}", index=False)
print(f"\nSaved coordinates to {excel_path}")

plt.show()  # |\label{l2_1g:plotEnd}|
