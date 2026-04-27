#!/usr/bin/python3
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse


################################################################################
# Notes
################################################################################
### Part 1: Modelling the carbon cycle
# - Model biosphere uptake by a box model
# - Model ocean uptake by impulse response functions
#
### Part 2: Energy balance modelling
# - Goal is to analyze: Human activities -> change in radiatie balance -> change in mean global temperature
# - Goal: Analyze temperature(mean global temperature) response caused by human activities.
# - Radiative forcing: Warminng due to greenhouse gas emissions and other substances
# - Energy balance model: The change in earths energy is calculated from radiative forcing, outgoing thermal radiation and energy to heat the oceans from pre-industrial levels to current temperatures.
# - Due to the large thermal capacity of the ocean there is a lag betwee radiative forcig and thermal radiation.
#
# Box Model:
# - Upper box: Combined effective heat capacity of upper ocean, atmosphere and land surfaces
# - Lower box: Deep ocean
# - Climate-forcing agents affect the upper box through radiative forcing
# - The upper and lower box conduct heat to each other
#
# ## Radiative forcing
# - defined as the change in net radiation in the tropopause.
#
# # The effect of CO2 on radiative forcing:
#   RF_co2 = 5.35 * ln((P_co2/P_co2_0))
# where P_co2 is the co2 concentration now, ad P_co2_0 is the concentration before industrialization.
#
#


################################################################################
# Biosphere uptake of CO2 (Part 1):
################################################################################
# Using a box model we have:
# - Box 0: Carbon in atmosphere
# - Box 1: Carbon in biomass above ground
# - Box 2: Carbon in biomass below ground
# Note that we name first box 0 instead of 1.
# B[i] - Box i: Carbon content (GtC)
# F[i, j] - Annual flow from box i to box j ()
# B0 - Sizes of boxes pre-industrial time
# F0 - Annual flow pre-industrial time

B0 = np.array([600, 600, 1500]) # Pre-industrial B (GtC)
F0 = np.array([ # Pre-industrial F (GtC / yr)
  [0, 60, 0],
  [15, 0, 45],
  [45, 0, 0],
])
NPP0 = F0[0, 1] # Pre-industrial NPP (GtC / yr)
print("NPP0", NPP0)

alpha = np.zeros((3, 3))
for i in range(3):
  for j in range(3):
    alpha[i, j] = F0[i, j] / B0[i]

# NPP(Net Primary Production) - Flow from atmosphere to above-ground biomass (GtC/yr)
# beta - Fertilization factor (???)
def NPP(NPP0, B, B0, beta):
  return NPP0*(1 + beta * np.log(B[0] / B0[0]))

# dB/dt for the basic energy balance model (GtC/yr)
# t_i - Time at discrete time point i
# i   - Discrete time point (0, 1, 2, ...)
# U   - Emission indexed by i
# alpha[i,j[i,j]] = F0[i,j] / B0[i]
# beta - Fertilization factor
def dBdt_basic(B, U, alpha, beta):
  NPP_val = NPP(NPP0, B, B0, beta)
  # "net flow" means inflow - outflow

  # # New code that doesn't work:
  # # Inflow modelled by a linear function of B
  # alpha_linear = alpha * np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]])
  # outflow_linear = B @ alpha_linear # Inflow modelled by a linear function of B
  # inflow_linear = alpha_linear @ B # Inflow modelled by a linear function of B
  # net_flow_linear = inflow_linear - outflow_linear
  # print(alpha)
  # print(alpha_linear)
  # print(B)
  # print("inflow linear", inflow_linear)
  # print("expected for [0]", alpha[2,0]*B[2] + alpha[1,0]*B[1])
  # print("outflow linear", outflow_linear)
  # print("net flow linear", net_flow_linear)
  # print("Expected net flow linear:")
  # print(np.array([
  #   alpha[2, 0]*B[2] + alpha[1, 0]*B[1],
  #   alpha[1, 2]*B[1] - alpha[1, 0]*B[1],
  #   alpha[1, 2]*B[1] - alpha[2, 0]*B[2]
  # ]))
  
  # assert math.isclose(inflow_linear[0], alpha[2,0]*B[2] + alpha[1,0]*B[1])
  # assert math.isclose(inflow_linear[1], 0)
  # assert math.isclose(outflow_linear[0], 0)
  # assert math.isclose(outflow_linear[1], alpha[2,3]*B[2] + alpha[2,1]*B[2])
  # assert math.isclose(net_flow_linear[0], alpha[2,0]*B[2] + alpha[1,0]*B[1])

  # net_flow_NPP = np.array([-NPP_val, NPP_val, 0]) # Inflow due to photosynthesis
  # net_flow_U = np.array([U, 0, 0]) # Inflow due to emisssions
  # return net_flow_linear + net_flow_NPP + net_flow_U

  
  return np.array([
    alpha[2, 0]*B[2] + alpha[1, 0]*B[1] - NPP_val + U,
    NPP_val - alpha[1, 2]*B[1] - alpha[1, 0]*B[1],
    alpha[1, 2]*B[1] - alpha[2, 0]*B[2]
  ])

################################################################################
# Impulse response function (Part 1)
################################################################################
# Model oceanic uptake of atmospheric excess carbon(due to emissions) by convolving emissions with an impulse response function with the atmospheric excess carbon. By "excess carbon" I mean the additional carbon in the atmosphere relative to pre-industrial times.
# Why this impulse function? Upper layers of the ocean absorb carbon relatively quickly, but carbon exchange between deeper and deeper layers of the ocean takes time.

# Consts
k = 3.06e-3
A = [0.113, 0.213, 0.258, 0.273, 0.1430]
tau0 = [2.0, 12.2, 50.4, 243.3, 1e12]
n = 5
M0 = B0[0] # Carbon stock in atmosphere before industrialization

def tau(i, s_idx, cumulative_U):
  return tau0[i] * (1 + k * cumulative_U[s_idx-1])

# Impulse control
def I(t_val, s_idx, cumulative_U):
  return sum(A[i] * np.exp(-t_val / tau(i, s_idx, cumulative_U)) for i in range(n))

# Amount of CO2 in atmosphere
def M(t, s, t_idx, cumulative_U):
  return M0 + sum(
    I(t[t_idx] - s[s_idx], t_idx, cumulative_U) * U[s_idx]
      for s_idx in range(t_idx + 1)
  )

################################################################################
# Load data from .csv-files
################################################################################
t = np.arange(1765, 2500+1) # years

# Emission data
emission_data = np.genfromtxt("utslappRCP45.csv", delimiter=",")
# for i in range(len(t)):
#   print(f"{t} - {emission_data[1:, 0]}")
# print(t)
# print(emission_data)
assert np.equal(emission_data[1:, 0] , t).all()
U = emission_data[1:, 1]
# Co2 concentration data
co2_concentration_data = np.genfromtxt("koncentrationerRCP45.csv", delimiter=",")
assert np.equal(co2_concentration_data[1:, 0], t).all()
co2_concentration = co2_concentration_data[1:, 1]
#
rf_data = np.genfromtxt("radiativeForcingRCP45.csv", delimiter=",")
np.equal(rf_data[1:, 0] , t).all()
rf_data_co2      = rf_data[1:, 1] # W/m²
rf_data_aerosols = rf_data[1:, 2] # W/m²
rf_data_other    = rf_data[1:, 3] # W/m²



print("emission_data")
print(emission_data)
print("co2_concentration_data")
print(co2_concentration_data)

################################################################################
# Tasks
################################################################################
# Constants
co2_per_gtc = 0.469 # (ppm CO₂/GtC)

## Parse arguments
parser = argparse.ArgumentParser();
# Question/task
parser.add_argument(
  "-q",
  default=0, # 0 means all
  type=int
)
# Parts
parser.add_argument(
  "-p",
  default=0, # 0 means all
  type=int
)

args = parser.parse_args()

# # Only used by old implementations, so can be removed
# def solve_euler(dfdt, t, f0, args=None):
#   f = np.zeros((len(t), *np.shape(f0)))
#   f[0] = f0
#   for i, t_i in enumerate(t):
#     if i == 0: continue
#     dt = t[i] - t[i-1]
#     dfdt_val = dfdt(t_i, i, f[i-1], f0, *args)
#     f[i] = f[i-1] + dfdt_val * dt
#   return f


## Tasks (in parts)
def part1():
  # Task 1:
  # - Construct a model for the carbon cycle
  # Q: Analyze how flows between differnt boxes are affected by emission_data:
  # A: The emissions cause all the flows to increase which is not suprising, but even if the emissions stop the flows continue. This is because the emissions cause a CO2 imbalance, and that imbalance will stay for very long times even if emissions stop.
  # Q: Compare to utslappRCP45.csv and answer "Why do you think your calculated concentrations differ?":
  # A: The atmospheric CO2 grow slightly faster with our model. One reason could be that we don't model the oceean absorbing CO2.
  def task1():
    beta = 0.35 # Fertilization factor

    # Calculate B by euler method
    # B = solve_euler(dBdt, t, B0, args=(NPP0, U, alpha, beta)) # Old way of calculating B
    NPP_vals = np.zeros((len(t),))
    B = np.zeros((len(t), *np.shape(B0)))
    B[0] = B0 # Initial condition
    # Loop for euler method:
    for t_idx in range(len(t)-1):
      dt = 1
      dBdt_val = dBdt_basic(B[t_idx], U[t_idx], alpha, beta)
      NPP_val = NPP(NPP0, B[t_idx], B0, beta)
      NPP_vals[t_idx] = NPP_val
      dBdt_val2= np.array([
        alpha[2, 0]*B[t_idx, 2] + alpha[1, 0]*B[t_idx, 1] - NPP_val + U[t_idx],
        NPP_val - alpha[1, 2]*B[t_idx, 1] - alpha[1, 0]*B[t_idx, 1],
        alpha[1, 2]*B[t_idx, 1] - alpha[2, 0]*B[t_idx, 2]
      ])
      assert np.isclose(dBdt_val, dBdt_val2).all()
      B[t_idx+1] = B[t_idx] + dBdt_val * dt

    # Plot flows
    plt.plot(t, NPP_vals, label="0 to 1")
    plt.plot(t, alpha[1,0]*B[:,1], label="1 to 0")
    plt.plot(t, alpha[2,0]*B[:,2], label="2 to 0")
    plt.plot(t, alpha[1,2]*B[:,1], label="1 to 2")
    plt.plot(t, U, label="Emissions")
    plt.title("Box flows")
    plt.xlabel("Year")
    plt.ylabel("CO2/yr")
    plt.legend()
    plt.show()

    # Calculate atomstpheric CO2 concentrations
    co2 = co2_per_gtc * B[:,0]
    # Plot CO2 according to model and data
    plt.plot(t, co2, label=f"beta={beta}")
    plt.plot(t, co2_concentration, "black", linestyle=":", label="koncentrationerRCP4.csv")
    plt.xlabel("Year")
    plt.ylabel("CO²")
    plt.title("Task 1 - Atmospheric CO2 by box model")
    plt.legend()
    plt.show()

  # Task 2:
  # - Same as Task 1, but we vary beta(CO2 fertilization factor)
  # Q: Describe what happens to the CO2 and carbon biomass
  # A: Both of them increase. The increase continues on even after emissions stop, which is expected as we know the flows do from task 1. Larger fertilization factor leads to faaster increase which is quite expected.
  # Q: "Explain the results by considering how an increased or decreased fertilization effect influences net primary production (NPP), carbon uptake by vegetation, and overall carbon cycling between the atmosphere, biosphere, and soil."
  # A: Increase in fertilization factor means faster photosynthesis which leads to a higher NPP, which is the flow from box 0 to 1.
  def task2():
    # Solve B for different values of beta(fertilization factor), and plot atmospheric CO2 concentrations
    for beta in np.linspace(0.1, 0.8, 3):
      # Calculate B by euler method
      # B = solve_euler(dBdt, t, B0, args=(NPP0, U, alpha, beta)) # Old method
      B = np.zeros((len(t), *np.shape(B0)))
      B[0] = B0
      # Loop for euler method:
      for t_idx in range(len(t)-1):
        dt = t[t_idx+1] - t[t_idx]
        dBdt_val = dBdt_basic(B[t_idx], U[t_idx], alpha, beta)
        B[t_idx+1] = B[t_idx] + dBdt_val * dt

      # Plot atmospheric CO2
      co2 = co2_per_gtc * B[:,0]
      plt.plot(t, co2, label=f"beta={beta}")
    # Plot CO2 concentration data for comparison
    plt.plot(t, co2_concentration, "black", linestyle=":", label="koncentrationerRCP4.csv")
    plt.title("Atmospheric CO²")
    plt.xlabel("Year")
    plt.ylabel("CO2")
    plt.legend()
    plt.show()

    for beta in np.linspace(0.1, 0.8, 3):
      B = np.zeros((len(t), *np.shape(B0)))
      B[0] = B0
      for t_idx in range(len(t)-1):
        dt = t[t_idx+1] - t[t_idx]
        dBdt_val = dBdt_basic(B[t_idx], U[t_idx], alpha, beta)
        B[t_idx+1] = B[t_idx] + dBdt_val * dt
      plt.plot(t, B[:, 1], label=f"box 1, beta={beta}")
      plt.plot(t, B[:, 2], label=f"box 2, beta={beta}")
    plt.title("Carbon: Box 1(biomass+upper soil) and Box 2(below ground)")
    plt.xlabel("Year")
    plt.ylabel("GtC")
    plt.legend()
    plt.show()
    

  # Task 3 
  def task3():
    for cumulative_U_val in (0, 140, 560, 1680):
      cumulative_U = [cumulative_U_val]
      t = np.arange(501)
      print(t)
      I_vals = np.zeros_like(t, dtype=np.float64)

      for t_idx, t_val in enumerate(t):
        I_vals[i] = I(t_val, 0, cumulative_U)

      plt.plot(t, I_vals, label=cumulative_U)
    plt.show()

  def task4():
    M_vals = np.zeros_like(t, dtype=np.float64)
    cumulative_U = np.zeros_like(t, dtype=np.float64)
    # Set initial values
    M_vals[0] = M0
    cumulative_U[0] = U[0]

    # Simulate
    for t_idx, t_val in enumerate(t[1:], 1): # Note that we simulate for t=0
      cumulative_U[t_idx] = cumulative_U[t_idx - 1] + U[t_idx]
      M_vals[t_idx] = M(t, t, t_idx, cumulative_U)

    co2 = co2_per_gtc * M_vals
    plt.plot(t, co2, label="A")
    plt.plot(t, co2_concentration, "black", label="koncentrationerRCP4.csv")
    plt.title("CO²")
    plt.show()

  # TODO: Adjust beta so that model-calculated concentrationsn align with kocntrationerRCP45.csv
  # TODO: Task 7
  def task6_and_7():
    ## Task 6
    # global t
    # global t
    # global co2_concentration
    #  Use a subset of t
    # t = t[0:300]
    # t = t[0:300]
    # co2_concentration = t[0:300]
    #Consts
    beta = 0.35

    # Simulation outputs  
    cumulative_U = np.zeros_like(t, dtype=np.float64)
    B = np.zeros((len(t), *np.shape(B0)))
    # Set initial values
    cumulative_U[0] = U[0]
    B[0] = B0
    # Simulate
    for t_idx in range(len(t)-1):
      dt = t[t_idx+1] - t[t_idx]
      NPP_val = NPP(NPP0, B[t_idx], B0, beta)

      # "emissions" include non athropogenic emissions, but not co2 uptake by oceans
      emissions = alpha[2, 0]*B[t_idx,2] + alpha[1, 0]*B[t_idx,1] - NPP_val + U[i]
      dB1dt = NPP_val - alpha[1, 2]*B[t_idx, 1] - alpha[1, 0]*B[t_idx, 1]
      dB2dt = alpha[1, 2]*B[t_idx, 1] - alpha[2, 0]*B[t_idx, 2]
      cumulative_U[t_idx+1] = cumulative_U[t_idx] + emissions
      B[t_idx+1, 0] = M(t, t, t_idx+1, cumulative_U)
      B[t_idx+1, 1] = B[t_idx, 1] + dB1dt * dt
      B[t_idx+1, 2] = B[t_idx, 2] + dB2dt * dt
    
    co2 = co2_per_gtc * B[:, 0]
    plt.plot(t, co2, label="model")
    plt.plot(t, co2_concentration, "black", label="koncentrationerRCP4.csv")
    plt.title("CO²")
    plt.legend()
    plt.show()

    ## Task 7
    plt.plot(t, B[:,0] / B0[0], label="1")
    plt.plot(t, B[:,1] / B0[1], label="2")
    plt.plot(t, B[:,2] / B0[2], label="3 / 10")
    plt.plot(t, (co2_concentration/co2_concentration[0]), "black", label="koncentrationerRCP45.csv")
    plt.title("carbon")
    plt.legend()
    plt.show()

  if args.q in (0, 1): task1()
  if args.q in (0, 2): task2()
  if args.q in (0, 3): task3()
  if args.q in (0, 4): task4()
  if args.q in (0, 6, 7): task6_and_7()


def part2():
  def task8():
    P_co2 = co2_concentration
    P_co2_0 = P_co2[0]
    rf_co2 = np.zeros_like(t, dtype=np.float64)
    for t_idx, t_val in enumerate(t):
      rf_co2     [t_idx] = 5.35 * np.log((P_co2     [t_idx]/P_co2_0))
      print(t_idx, t_val, rf_co2[t_idx], (5.35 * np.log(P_co2/P_co2_0))[t_idx])
    plt.plot(t, rf_co2)
    plt.plot(t, 5.35 * np.log(P_co2/P_co2_0), label="model")
    plt.plot(t, rf_data_co2, label="koncentrationerRCP45.csv")
    plt.title("Radiative forcing")
    plt.legend()
    plt.show()

  def task9():
    s = 1
    rf_total = rf_data_co2 + s * rf_data_aerosols + rf_data_other
    plt.plot(t, rf_data_co2     , label="co2")
    plt.plot(t, s * rf_data_aerosols, label="s * aerosols")
    plt.plot(t, rf_data_other   , label="other")
    plt.plot(t, rf_total   , label="all")
    plt.ylabel("W/m²")
    plt.xlabel("Year")
    plt.legend()
    plt.title("Radiative forcing")
    plt.show()
  


  def task10():
    s = 1
    rf_total = rf_data_co2 + s * rf_data_aerosols + rf_data_other

    lambda_param = None # (K W⁻¹ m²), climate sensitivity parameter 
    kappa = None # (W K⁻¹ m⁻²), exchange coefficition between box 1 and box 2
    c_jl = 4186 # (J/kg)/K Specific heat capacity of water using Joule units.
    c = c_jl * 3600 * 24 * 365.25 # ((W yr / kg)/K) Same as c_jl, but with the prefered units.
    rho = 1020 # (kg/m³) Density of water
    h = 50 # (m) the effective height of box 0 (Surface)
    d = 2000 # (m) The effective depth of box 1 (Deep ocean)
    C = np.array([c * h * rho, c * d * rho]) # (W yr / kg / K * m * kg / m³ =
    # = W yr K⁻¹ m⁻²) Effective heat capacity for box 0 and 1
    # T_diff = temperature response 

    # t - Time in yrs
    # rf_net - Net radiative forcing in (W/m²)
    def simulate(t, rf_net, lambda_param=0.8, kappa=0.5):
      # T_diff - Temperature difference since preindustrial times in Kelvin
      T_diff = np.zeros((len(t), 2), dtype=np.float64)
      T_diff[0] = [0, 0] # Initial conditions

      # dTdt_arr = np.zeros((len(t), 2), dtype=np.float64)
      for t_idx in range(len(t) - 1):
        dt = t[t_idx+1] - t[t_idx]
        temp_exchange = kappa * (T_diff[t_idx, 0] - T_diff[t_idx, 1]) # (W m⁻²)
        dTdt = np.array([
          rf_net[t_idx] - T_diff[t_idx, 0] / lambda_param - temp_exchange,
          temp_exchange
        ]) / C * 1e15 # (K/yr)
        T_diff[t_idx+1] = T_diff[t_idx] + dTdt * dt # (Kelvin)
        # dTdt_arr[t_idx+1] = dTdt
      return T_diff    
    
    
    

    def plot_T_diff(t, T_diff, lambda_param=0.8, kappa=0.5, label_suffix="", color=None):
      equilibrium = 1 * lambda_param
      e_folding_time = None
      for t_idx, t_val in enumerate(t):
        if T_diff[t_idx, 0] > (1 - 1/math.e) * equilibrium:
          e_folding_time = t_val
          print("Folding time:", e_folding_time)
          break

      plt.plot(t, T_diff[:,0], c=color, label = "Box 0(atomsphere + upper ocean)" + label_suffix)
      plt.plot(t, T_diff[:,1], c=color, linestyle="--", label = "Box 1(deep ocean)" + label_suffix)
      # plt.plot(t, dTdt_arr[:,0], label = "derivate of Box 0")
      # plt.plot(t, dTdt_arr[:,1], label = "derivate of Box 1")
      plt.hlines(equilibrium, t[0], t[-1], "black", linestyle=":")#, label="Equilibrium temperature difference" + label_suffix)
      plt.vlines(e_folding_time, 0, equilibrium, "black", linestyle=":")
      plt.xlabel("Year")
      plt.ylabel("Kelvin")
      plt.text(0.5 * kappa*t[-1], equilibrium+0.01, f"Folding time: {e_folding_time}") # Weird x coordinate because we don't want text to overlap
      plt.legend()
    
    
    # Task 10a
    t_test = np.arange(0, 10**4)
    rf_test = np.zeros_like(t_test, dtype=np.float64) + 1
    plt.title("Task 10a - Temperature response")
    T_diff = simulate(t_test, rf_test)
    plot_T_diff(t_test, T_diff)
    plt.show()

    # Task 10b
    plt.title("Task 10b - Temperature response for different lambdas")
    lambda_params = np.linspace(0.5, 1.3, 3)
    colors = plt.cm.jet(np.linspace(0, 1, len(lambda_params)))
    for idx in range(len(lambda_params)):
      lambda_param = lambda_params[idx]
      color = colors[idx]
      T_diff = simulate(t_test, rf_test, lambda_param=lambda_param)
      plot_T_diff(t_test, T_diff, lambda_param=lambda_param, color=color, label_suffix=f", lambda={lambda_param}")
    plt.show()

    plt.title("Task 10b - Temperature response for different kappas")
    lambda_param = 0.8
    kappas = np.linspace(0.2, 1.0, 3)
    colors = plt.cm.jet(np.linspace(0, 1, len(kappas)))
    for idx in range(len(kappas)):
      kappa = kappas[idx]
      color = colors[idx]
      T_diff = simulate(t_test, rf_test, kappa=kappa)
      plot_T_diff(t_test, T_diff, kappa=kappa, color=color, label_suffix=f", kappa={kappa}")
    plt.show()

    # TODO: Task 10c

    # Not the exercise:
    T_diff = simulate(t, rf_total, lambda_param)
    plot_T_diff(t, T_diff)
    plt.show()
  

  #   # Simulate
  #   M_vals = np.zeros_like(t)
  #   for t_idx, t_val in enumerate(t):
  #     M_vals[t_idx] = M(t, t, t_idx, cumulative_U)

  #   # Simulate
  #   for t_idx, t_val in enumerate(t):
    
  if args.q in (0, 8): task8()
  if args.q in (0, 9): task9()
  if args.q in (0, 10): task10()

def part3():
  print("TODO: part 3")

## Do tasks
if args.p in (0,1): part1()
if args.p in (0,2): part2()
if args.p in (0,3): part3()
  

