""" water movement within the root (static soil) """
import sys; sys.path.append("../.."); sys.path.append("../../src/")

import plantbox as pb
import plantbox.visualisation.vtk_plot as vp
#from plantbox.functional.xylem_flux import XylemFluxPython
from plantbox.functional.PlantHydraulicModel import HydraulicModel_Meunier
from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

import numpy as np
import matplotlib.pyplot as plt

""" Parameters """
#kz = 4.32e-2  # axial conductivity [cm3/day]
#kr = 1.728e-4  # radial conductivity [1/day]
#do conductivity by segment (same conductivities for each topology):
#radial
kr_mp = 8.83e-05
kr_mm = 8.76e-05
kr_mt = 9.69e-05
kr_lp = 8.67e-05
kr_lm = 8.51e-05
kr_lt = 8.65e-05
#axial:
kx_mp = 2.2e-06
kx_mm = 3.44e-07
kx_mt = 3.85e-07
kx_lp = 4.18e-07
kx_lm = 4.16e-07
kx_lt = 4.20e-07


p_s = -200  # constant soil potential [cm]
p0 = -500  # dirichlet bc at top [cm]
simtime = 14  # [day]

""" root system """
rs = pb.MappedPlant()
path = "../../modelparameter/structural/rootsystem/"
#name = "Anagallis_femina_Leitner_2010"  # Zea_mays_1_Leitner_2010
name = "test"
rs.readParameters(path + name + ".xml")
rs.initialize()
rs.simulate(simtime, False)

""" root problem """
params = PlantHydraulicParameters()
params.setKxValues(kx_mm,kx_mp,kx_mt,kx_lm,kx_lm,kx_lt)
params.setKrValues(kr_mm,kr_mp,kr_mt,kr_lm,kr_lm,kr_lt)
r = HydraulicModel_Meunier(rs, params, cached=False)
soil_index = lambda x, y, z: 0
r.ms.setSoilGrid(soil_index)

""" Numerical solution """
soil = [p_s]
rx = r.solve_dirichlet(simtime, p0, soil, cells=True)
fluxes = r.radial_fluxes(simtime, rx, soil, cells=True)
print("Transpiration", r.get_transpiration(simtime, rx, soil, cells=True), "cm3/day")

""" Macroscopic root system parameter """
suf = r.get_suf(simtime)
krs, _ = r.get_krs(simtime)
print("Krs: ", krs, "cm2/day")

""" plot results """
nodes = r.get_nodes()
plt.plot(rx, nodes[:, 2] , "r*")
plt.xlabel("Xylem potentials (cm)")
plt.ylabel("Depth (m)")
plt.show()

""" Additional vtk plot """
ana = pb.SegmentAnalyser(r.rs.mappedSegments())
ana.addData("rx", rx)  # xylem potentials [cm]
ana.addData("SUF", suf)  # standard uptake fraction [1]
ana.addAge(simtime)  # age [day]
ana.addConductivities(r, simtime)  # kr [1/day], kx [cm3/day]
ana.addFluxes(r, rx, p_s * np.ones(rx.shape), simtime)  # "axial_flux" [cm3/day], "radial_flux" [ (cm3/cm2) / day]
vp.plot_roots(ana, "subType")  # "rx", "SUF", "age", kr, "axial_flux", "radial_flux"
