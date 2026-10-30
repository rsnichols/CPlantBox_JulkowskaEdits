import plantbox as pb

TARGET_LENGTH_CM = 100.0
TARGET_SURFACE_CM2 = 20.0
MATCH_DAY = 14

_LENGTH_FIELDS = ("lmax", "lmaxs", "la", "las", "lb", "lbs", "ln", "lns", "r", "rs")


def measure_roots(plant):
    """Realized root length [cm] and cylinder surface area [cm^2]."""
    length = 0.0
    surface = 0.0
    for organ in plant.getOrgans(pb.root):
        length += organ.getParameter("length")
        surface += organ.getParameter("surface")
    return length, surface


def _scale_parameters(plant, length_scale, radius_scale):
    for param in plant.getOrganRandomParameter(pb.root):
        if param.subType == 0 or not hasattr(param, "lmax"):
            continue
        for name in _LENGTH_FIELDS:
            setattr(param, name, getattr(param, name) * length_scale)
        param.a *= radius_scale
        param.a_s *= radius_scale


def apply_day14_targets(
    plant,
    day=MATCH_DAY,
    target_length=TARGET_LENGTH_CM,
    target_surface=TARGET_SURFACE_CM2,
):
    """Simulate to `day`, then scale parameters so that day matches the targets.

    Call this after readParameters and setGeometry, and before the figure
    simulation. The plant is left at the end of a calibration run; start a
    fresh initialize() afterwards.
    """
    for round_index in range(3):
        plant.setSeed(1)
        plant.initialize(False)
        plant.simulate(day, False)
        length, surface = measure_roots(plant)
        print(
            f"Calibration round {round_index + 1} at day {day}: "
            f"length {length:.4f} cm, surface {surface:.4f} cm^2"
        )
        if length <= 0.0 or surface <= 0.0:
            raise RuntimeError("Root system has no length or surface to calibrate.")
        length_scale = target_length / length
        radius_scale = target_surface / (length_scale * surface)
        if abs(length_scale - 1.0) < 1e-3 and abs(radius_scale - 1.0) < 1e-3:
            break
        _scale_parameters(plant, length_scale, radius_scale)
        print(
            f"  applied length scale {length_scale:.6f}, "
            f"radius scale {radius_scale:.6f}"
        )
    print(
        f"Day {day} targets: {target_length:.1f} cm length, "
        f"{target_surface:.1f} cm^2 surface."
    )
