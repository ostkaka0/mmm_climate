#!/usr/bin/python3
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import os

# Helper function to save plots to .pdf-files
def show_plot(name):
  filepath = f"plots/{name}.pdf"
  plt.savefig(filepath, format="pdf")
  plt.show()

# We always use fixed time steps of 1 year in simulations
dt = 1

# Color and style sets
colors = [plt.cm.tab10(i) for i in range(5)]
styles = [":", "--", "-.", (5, (10, 3))]

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
# - Radiative forcing: Warminng due to greenhouse gas emissions and other substances.
# - Energy balance model: The change in earths energy is calculated from radiative forcing, outgoing thermal radiation and energy to heat the oceans from pre-industrial levels to current temperatures.
# - Due to the large thermal capacity of the ocean there is a lag betwee radiative forcing and thermal radiation.
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

box_names = ["Atmosphere", "Biosphere", "Below Ground", "Ocean"]

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

beta_default = 0.35 # Default value for beta(Fertilization factor)

# NPP(Net Primary Production) - Flow from atmosphere to above-ground biomass (GtC/yr)
# beta - Fertilization factor
def NPP(NPP0, B, B0, beta):
  return NPP0*(1 + beta * np.log(B[0] / B0[0]))

# dB/dt for the basic energy balance model (GtC/yr)
# t_i - Time at discrete time point i
# i   - Discrete time point (0, 1, 2, ...)
# U   - Emission indexed by i
# alpha[i,j[i,j]] = F0[i,j] / B0[i]
# beta - Fertilization factor
def dBdt_basic(B, U, beta):
  NPP_val = NPP(NPP0, B, B0, beta)

  # # Alternative code that might not be more readable:
  # # Inflow modelled by a linear function of B
  # alpha_linear = alpha * np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]])
  # inflow_linear = alpha_linear.T @ B # Inflow modelled by a linear function of B
  # outflow_linear = np.sum(alpha_linear, axis=1) * B # Inflow modelled by a linear function of B
  # net_flow_linear = inflow_linear - outflow_linear
  # return net_flow_linear + net_flow_NPP + net_flow_U

  return np.array([
    alpha[2,0]*B[2] + alpha[1,0]*B[1] - NPP_val + U,
    NPP_val - alpha[1, 2]*B[1] - alpha[1, 0]*B[1],
    alpha[1,2]*B[1] - alpha[2,0]*B[2]
  ])

################################################################################
# Impulse response function (Part 1)
################################################################################
# Model oceanic uptake of atmospheric excess carbon(due to emissions) by convolving emissions with an impulse response function with the atmospheric excess carbon. By "excess carbon" I mean the additional carbon in the atmosphere relative to pre-industrial times.
# Why this impulse function? Upper layers of the ocean absorb carbon relatively quickly, but carbon exchange between deeper and deeper layers of the ocean takes time. Also as the oceans gets more saturated with carbon, the absoption slows down.

# Consts
k_default = 3.06e-3 # Constant for speed of oceanic saturation(and absorption)
A = [0.113, 0.213, 0.258, 0.273, 0.1430] # Fraction of CO2 emissions that decay with a time constant tau_i.
tau0 = [2.0, 12.2, 50.4, 243.3, math.inf] # Tau before industrialization
M0 = B0[0] # Carbon stock in atmosphere before industrialization
O0 = 900 + 3 + 37100 + 700 + 1750

# Time constant
def tau(i, s, cumulative_Us, k=k_default):
  return tau0[i] * (1 + k * cumulative_Us[s])

# Impulse response (Fraction of CO2 remaining)
# t, may be np-array or single float
# cumulative_U - cumulative *prior* emissions.
def I(t, s, cumulative_Us, k=k_default):
  return sum(
    A[i] * np.exp(-t / tau(i, s, cumulative_Us, k=k))
      for i in range(len(A)-1)
  ) + A[4]

# Amount of GtC in atmosphere
# cumulative_Us - cumulative *prior* emissions.
def M(t, U, cumulative_Us, k=k_default):
  return M0 + sum(
    I(t - s, t, cumulative_Us, k=k) * U[s]
      for s in range(t+1)
  )
# Amount of GtC in ocean
# Calculated similarly to M
def O(t, U, cumulative_Us, k=k_default):
  return O0 + sum(
    (1 - I(t-s, t, cumulative_Us, k=k)) * U[s]
      for s in range(t+1)
  )


################################################################################
# Combined model(box and impulse model)
################################################################################
# The final carbon balance model where we extend the basic box-model with the impulse-response model for oceanic absorption of carbon(co2).
def simulate_carbon_balance(k=k_default, beta=beta_default):
  cumulative_Us = np.zeros_like(time_steps, dtype=np.float64)
  B = np.zeros((len(time_steps), 4))
  net_emissions = np.zeros(len(time_steps))
  # Set initial values
  cumulative_Us[0] = 0
  net_emissions[0] = U[0]
  B[0] = [*B0, 1750 + 37100 + 700 + 900 + 3]
  # Simulate
  for t in range(len(time_steps)-1):
    NPP_val = NPP(NPP0, B[t], B0, beta)

    # "emissions" include non athropogenic emissions, but not co2 uptake by oceans
    net_emissions[t+1] = alpha[2,0]*B[t,2] + alpha[1, 0]*B[t,1] - NPP_val + U[t+1]
    dB1dt = NPP_val - alpha[1,2]*B[t,1] - alpha[1,0]*B[t,1]
    dB2dt = alpha[1,2]*B[t,1] - alpha[2,0]*B[t,2]
    cumulative_Us[t+1] = cumulative_Us[t] + net_emissions[t]
    # Euler step
    B[t+1,0] = M(t+1, net_emissions, cumulative_Us, k=k)
    B[t+1,1] = B[t,1] + dB1dt * dt
    B[t+1,2] = B[t,2] + dB2dt * dt
    B[t+1,3] = O(t+1, net_emissions, cumulative_Us, k=k)
  return B

################################################################################
# Radiative forcing
################################################################################
def calc_rf_co2(P_co2):
  rf_co2 = 5.35 * np.log((P_co2/P_co2[0]))
  # rf_co2 = np.zeros_like(time_steps, dtype=np.float64)
  # for t in time_steps:
  #   rf_co2     [t] = 5.35 * np.log((P_co2     [t]/P_co2_0))
  #   # print(t, t, rf_co2[t], (5.35 * np.log(P_co2/P_co2_0))[t])
  return rf_co2

## The energy balance model
c_jl = 4186 # (J/kg)/K Specific heat capacity of water using Joule units.
c = c_jl / (3600 * 24 * 365.25) # ((W yr / kg)/K) Same as c_jl, but with the prefered units.
rho = 1020 # (kg/m³) Density of water
h = 50 # (m) the effective height of box 0 (Surface)
d = 2000 # (m) The effective depth of box 1 (Deep ocean)
C = np.array([c * h * rho, c * d * rho]) # (W yr / kg / K * m * kg / m³ =
# = W yr K⁻¹ m⁻²) Effective heat capacity for box 0 and 1

# time_steps - Time in yrs
# rf_net - Net radiative forcing in (W/m²)
def simulate_temp(time_steps, rf_net, lambda_param=0.8, kappa=0.5):
  # T_diff - Temperature difference since preindustrial times in Kelvin
  dTdt = np.zeros((len(time_steps), 2), dtype=np.float64)
  T_diff = np.zeros((len(time_steps), 2), dtype=np.float64)
  # Initial conditions
  T_diff[0] = [0,0]
  dTdt[0] = np.array([
    rf_net[0],
    0
  ]) / C # (K/yr)
  
  for t in range(1, len(time_steps)):
    print("t =", t)
    T_diff[t] = T_diff[t-1] + dTdt[t-1] * dt # (Kelvin)
    temp_exchange = kappa * (T_diff[t,0] - T_diff[t,1]) # (W m⁻²)
    dTdt[t] = np.array([
      rf_net[t] - T_diff[t,0] / lambda_param - temp_exchange,
      temp_exchange
    ]) / C # (K/yr)
    
  return T_diff, dTdt

################################################################################
# Time
################################################################################
# Time: Time-values are always integers. The generally use relative time that start at t=0 when simulation begins. For plotting we simply add start_year to get the absolute time, or simply use absolute_time_steps.
# The variables `t` or `s` generally refer to specific time points. 
start_year = 1765
end_year = 2500
absolute_time_steps = np.arange(start_year, end_year+1)
time_steps = np.arange(0, end_year-start_year+1)

################################################################################
# Load data from .csv-files
################################################################################

# Emission data
U_csv = np.genfromtxt("utslappRCP45.csv", delimiter=",")
assert np.equal(U_csv[1:, 0] , absolute_time_steps).all()
U = U_csv[1:, 1] # GtC/yr
# Co2 concentration data
P_co2_data_csv = np.genfromtxt("koncentrationerRCP45.csv", delimiter=",")
assert np.equal(P_co2_data_csv[1:, 0], absolute_time_steps).all()
P_co2_data = P_co2_data_csv[1:, 1] # ppm CO2

#
rf_data_csv = np.genfromtxt("radiativeForcingRCP45.csv", delimiter=",")
np.equal(rf_data_csv[1:, 0] , absolute_time_steps).all()
rf_data_co2      = rf_data_csv[1:, 1] # W/m²
rf_data_aerosols = rf_data_csv[1:, 2] # W/m²
rf_data_other    = rf_data_csv[1:, 3] # W/m²

T_diff_data_csv = np.genfromtxt("NASA_GISS.csv", delimiter=",")
T_diff_data_t = T_diff_data_csv[2:, 0]
# assert np.equal(T_diff_data_csv[2:, 0], time_steps).all()
T_diff_data = T_diff_data_csv[2:, 1] # Kelvin

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
# def solve_euler(dfdt, time_steps, f0, args=None):
#   f = np.zeros((len(time_steps), *np.shape(f0)))
#   f[0] = f0
#   for i, t_i in enumerate(time_steps):
#     if i == 0: continue
#     dt = time_steps[i] - time_steps[i-1]
#     dfdt_val = dfdt(t_i, i, f[i-1], f0, *args)
#     f[i] = f[i-1] + dfdt_val * dt
#   return f


## Tasks (in parts)
def part1():
  # Task 1:
  # - Construct a model for the carbon cycle
  # Q: Analyze how flows between differnt boxes are affected by U_csv:
  # A: The emissions cause all the flows to increase which is not suprising, but even if the emissions stop the flows continue. This is because the emissions cause a CO2 imbalance, and that imbalance will stay for very long times even if emissions stop.
  # Q: Compare to utslappRCP45.csv and answer "Why do you think your calculated concentrations differ?":
  # A: The atmospheric CO2 grow slightly faster with our model. One reason could be that we don'time_steps model the oceean absorbing CO2.
  def task1():
    beta = 0.35 # Fertilization factor

    # Calculate B by euler method
    # B = solve_euler(dBdt, time_steps, B0, args=(NPP0, U, alpha, beta)) # Old way of calculating B
    NPP_vals = np.zeros((len(time_steps),))
    B = np.zeros((len(time_steps), *np.shape(B0)))
    # Inital conditions
    B[0] = B0
    NPP_vals[0] = NPP0
    # Simulate
    for t in range(len(time_steps)-1):
      dBdt_val = dBdt_basic(B[t], U[t], beta)
      NPP_val = NPP(NPP0, B[t], B0, beta)
      NPP_vals[t+1] = NPP_val
      B[t+1] = B[t] + dBdt_val * dt # Euler step

    # Plot flows
    plt.plot(absolute_time_steps, NPP_vals, label="0 to 1")
    plt.plot(absolute_time_steps, alpha[1,0]*B[:,1], label="1 to 0")
    plt.plot(absolute_time_steps, alpha[2,0]*B[:,2], label="2 to 0")
    plt.plot(absolute_time_steps, alpha[1,2]*B[:,1], label="1 to 2")
    plt.plot(absolute_time_steps, U, label="Emissions")
    plt.title("Box flows (Task 1)")
    plt.xlabel("Year")
    plt.ylabel("GtC/yr")
    plt.legend()
    show_plot("t1a")

    # Calculate atomstpheric CO2 concentrations
    P_co2 = co2_per_gtc * B[:,0] # ppm
    # Plot CO2 according to model and data
    plt.plot(absolute_time_steps, P_co2, label=f"beta={beta}")
    plt.plot(absolute_time_steps, P_co2_data, "black", linestyle=":", label="koncentrationerRCP4.csv")
    plt.xlabel("Year")
    plt.ylabel("ppm CO2")
    plt.title("Atmospheric CO2 concentration by box model (Task 1)")
    plt.legend()
    show_plot("t1b")

  # Task 2:
  # - Same as Task 1, but we vary beta(fertilization factor)
  # Q: Describe what happens to the CO2 and carbon biomass
  # A: Both of them increase. The increase continues on even after emissions stop, which is expected as we know the flows do from task 1. Larger fertilization factor leads to faaster increase which is quite expected.
  # Q: "Explain the results by considering how an increased or decreased fertilization effect influences net primary production (NPP), carbon uptake by vegetation, and overall carbon cycling between the atmosphere, biosphere, and soil."
  # A: Increase in fertilization factor means faster photosynthesis which leads to a higher NPP, which is the flow from box 0 to 1.
  def task2():
    # Solve B for different values of beta(fertilization factor), and plot atmospheric CO2 concentrations
    for beta in np.linspace(0.1, 0.8, 3):
      B = np.zeros((len(time_steps), *np.shape(B0)))
      B[0] = B0 # Initial condition
      for t in range(len(time_steps)-1):
        dBdt_val = dBdt_basic(B[t], U[t], beta)
        B[t+1] = B[t] + dBdt_val * dt # By euler method

      # Plot atmospheric CO2
      co2 = co2_per_gtc * B[:,0]
      plt.plot(absolute_time_steps, co2, label=f"beta={beta}")
    # Plot CO2 concentration data for comparison
    plt.plot(absolute_time_steps, P_co2_data, "black", linestyle=":", label="koncentrationerRCP4.csv")
    plt.title("Atmospheric CO² (Task 2)")
    plt.xlabel("Year")
    plt.ylabel("CO2")
    plt.legend()
    show_plot("t2a")

    for beta_idx, beta in enumerate(np.linspace(0.1, 0.8, 3)):
      B = np.zeros((len(time_steps), *np.shape(B0)))
      B[0] = B0
      for t in range(len(time_steps)-1):
        dBdt_val = dBdt_basic(B[t], U[t], beta)
        B[t+1] = B[t] + dBdt_val * dt
      plt.plot(absolute_time_steps, B[:, 1], c=colors[beta_idx], linestyle="-", label=f"box 1, beta={beta}")
      plt.plot(absolute_time_steps, B[:, 2], c=colors[beta_idx], linestyle="--", label=f"box 2, beta={beta}")
    plt.title("GtC for Box 1(biomass+upper soil) and Box 2(below ground)  (Task 2)")
    plt.xlabel("Year")
    plt.ylabel("GtC")
    plt.legend()
    show_plot("t2b")
    

  # Task 3
  # - "Implement a time-discrete model that reproduces the impulse responses shown..."
  def task3():
    for cumulative_U in (0, 140, 560, 1680):
      our_time_steps = np.arange(501)
      I_vals = np.zeros_like(our_time_steps, dtype=np.float64)

      for t in our_time_steps:
        I_vals[t] = I(t, 0, [cumulative_U])
      plt.plot(our_time_steps, I_vals, label=f"{cumulative_U} GtC")
    plt.xlabel("Time since emissions(year)")
    plt.ylabel("Proportion still in atmosphere")
    plt.legend()
    plt.title("Impulse response for CO₂ based on earlier cumulative emissions (Task 3)")
    show_plot("t3a")

  # "Implement a model based on Equation 8 and run it using the emissions data from the file utslappRCP45.csv to calculate how atmospheric CO₂ concentration would have developed if carbon were only taken up by the ocean"
  # Q: Compare to koncentrationerRCP45.csv
  # A: Model follows the .csv file somewhat, but the atmospheric CO2 is higher because we don'time_steps model co2 uptake by biosphere and ground.
  def task4():
    M_vals = np.zeros_like(time_steps, dtype=np.float64)
    cumulative_Us = np.zeros_like(time_steps, dtype=np.float64)
    # Set initial values
    M_vals[0] = M0
    cumulative_Us[0] = U[0]

    # Simulate
    for t in time_steps[1:]: # Note that we simulate for time_steps=0
      cumulative_Us[t] = cumulative_Us[t - 1] + U[t - 1]
      M_vals[t] = M(t, U, cumulative_Us)

    co2 = co2_per_gtc * M_vals
    plt.plot(absolute_time_steps, co2, label="A")
    plt.plot(absolute_time_steps, P_co2_data, "black", label="koncentrationerRCP4.csv")
    plt.xlabel("Year")
    plt.ylabel("CO²")
    plt.title("Atmospheric CO² by only modelling oceanic absorption (Task 4)")
    plt.legend()
    show_plot("t4a")

  # Task 5: Draw a new box model that also includes oceanic carbon uptake, and add anthropogenic emissions.

  # TODO: Adjust beta so that model-calculated concentrationsn align with kocntrationerRCP45.csv
  # Task 6:
  # - "Connect the impulse response model for oceanic CO₂ uptake with the box model for biospheric CO2 uptake"
  # - Find reasonable value for beta
  # Task 7:
  # - Analyze long term fate of antrhopogenic CO2 emissions
  def task6_and_7():
    betas = [0.25, 0.35, 0.45]
    ks = [1e-3, k_default, 5e-6]
    # 6 and half of 7
    for k_idx, k in enumerate(ks):
      for beta_idx, beta in enumerate(betas):
        B = simulate_carbon_balance(k, beta)
        co2 = co2_per_gtc * B[:, 0]
        plt.plot(absolute_time_steps, co2, color=colors[k_idx], linestyle=styles[beta_idx], label=f"model, k={k}, beta={beta}")
    plt.plot(absolute_time_steps, P_co2_data, "black", linestyle="-", label="koncentrationerRCP4.csv")
    plt.title(f"CO², beta={beta} (Task 6)")
    plt.legend()
    show_plot("t6a")
    # Last part of 7
    k = ks[1]
    beta = betas[1]
    B = simulate_carbon_balance(k, beta)
    for idx in range(4):
      plt.plot(absolute_time_steps, B[:, idx] - B[0, idx], color=colors[idx], label=f"model, box={idx}({box_names[idx]}), k={k}, beta={beta}")
      plt.plot(absolute_time_steps, np.sum(B - B[0], axis=1), color=colors[4], label=(f"model, net carbon change" if idx==3 else None))
    plt.plot(absolute_time_steps+1, np.cumsum(U), ":", color="black", label=f"Cumulative (prior) emissions")
    plt.xlabel("Year")
    plt.ylabel("GtC")
    plt.title(f"Carbon change relative to pre-industrial times, beta={beta} (Task 7)")
    plt.legend()
    show_plot("t6b")

  if args.q in (0, 1): task1()
  if args.q in (0, 2): task2()
  if args.q in (0, 3): task3()
  if args.q in (0, 4): task4()
  if args.q in (0, 6, 7): task6_and_7()

def part2():
  # Task 8
  # - "Construct a radiative forcing module"
  # Q: "alculate and visualize the radiative forcing for CO₂, and compare these values with the CO₂ radiative forcing in radiativeForcingRCP45.csv"
  # A: Our model is very close to the data given, it's just slightly below. The reasons seems to be that there is a "jump" from the inital time point to the first year in the .csv file. We could adjust the model to take that into consideration and the difference between our model and the .csv would probably become very close. 
  def task8():
    P_co2 = P_co2_data
    P_co2_0 = P_co2[0]
    rf_co2 = calc_rf_co2(P_co2)

    plt.plot(absolute_time_steps, rf_co2)
    plt.plot(absolute_time_steps, 5.35 * np.log(P_co2/P_co2_0), label="model")
    plt.plot(absolute_time_steps, rf_data_co2, label="radiativeForcingRCP45.csv")
    plt.title("Radiative forcing (Task 8)")
    plt.legend()
    show_plot("t8a")

  # Task 9:
  # - "Sum the radiative forcing for other climate-affecting substances and aerosols"
  def task9():
    s = 1
    rf_total = rf_data_co2 + s * rf_data_aerosols + rf_data_other
    plt.plot(absolute_time_steps, rf_data_co2     , label="co2")
    plt.plot(absolute_time_steps, s * rf_data_aerosols, label="s * aerosols")
    plt.plot(absolute_time_steps, rf_data_other   , label="other")
    plt.plot(absolute_time_steps, rf_total   , label="all")
    plt.ylabel("W/m²")
    plt.xlabel("Year")
    plt.legend()
    plt.title("Radiative forcing (Task 9)")
    show_plot("t9a")

  
  def task10():
    s = 1
    rf_total = rf_data_co2 + s * rf_data_aerosols + rf_data_other

    lambda_param = None # (K W⁻¹ m²), climate sensitivity parameter 
    kappa = None # (W K⁻¹ m⁻²), exchange coefficition between box 1 and box 2
    

    # # time_steps - Time in yrs
    # # rf_net - Net radiative forcing in (W/m²)
    # def simulate(time_steps, rf_net, lambda_param=0.8, kappa=0.5):
    #   # T_diff - Temperature difference since preindustrial times in Kelvin
    #   dTdt = np.zeros((len(time_steps), 2), dtype=np.float64)
    #   T_diff = np.zeros((len(time_steps), 2), dtype=np.float64)
    #   T_diff[0] = [0, 0] # Initial conditions

    #   # dTdt_arr = np.zeros((len(time_steps), 2), dtype=np.float64)
    #   for t in range(len(time_steps) - 1):
    #     temp_exchange = kappa * (T_diff[t, 0] - T_diff[t, 1]) # (W m⁻²)
    #     dTdt[t] = np.array([
    #       rf_net[t] - T_diff[t, 0] / lambda_param - temp_exchange,
    #       temp_exchange
    #     ]) / C # (K/yr)
    #     T_diff[t+1] = T_diff[t] + dTdt[t] * dt # (Kelvin)
        
    #     # dTdt_arr[t+1] = dTdt
    #   return T_diff, dTdt   
    
    
    

    def plot_T_diff(time_steps, T_diff, lambda_param=0.8, kappa=0.5, label_suffix="", color=None):
      equilibrium = 1 * lambda_param # rf_net = 1 for all time_steps
      e_folding_time = None
      for t in time_steps:
        if T_diff[t, 0] > (1 - 1/math.e) * equilibrium:
          e_folding_time = t
          print("Folding time:", e_folding_time)
          break

      plt.plot(T_diff[:,0], c=color, label = "Box 0(atomsphere + upper ocean)" + label_suffix)
      plt.plot(T_diff[:,1], c=color, linestyle="--", label = "Box 1(deep ocean)" + label_suffix)
      # plt.plot(absolute_time_steps, dTdt_arr[:,0], label = "derivate of Box 0")
      # plt.plot(absolute_time_steps, dTdt_arr[:,1], label = "derivate of Box 1")
      plt.hlines(equilibrium, time_steps[0], time_steps[-1], "black", linestyle=":")#, label="Equilibrium temperature difference" + label_suffix)
      plt.vlines(e_folding_time, 0, equilibrium, "black", linestyle=":")
      plt.xlabel("Year")
      plt.ylabel("Kelvin")
      plt.text(0.5 * kappa*time_steps[-1], equilibrium+0.01, f"Folding time: {e_folding_time}") # Weird x coordinate because we don'time_steps want text to overlap
      plt.legend()
    
    
    # Task 10a
    # "Test the model by analyzing the temperature response based on a radiative forcing step of 1 W/m2"
    T = 10**3
    t_test = np.arange(0, T)
    rf_test = np.zeros_like(t_test, dtype=np.float64) + 1
    plt.title("Temperature response (Task T0a)")
    T_diff, _ = simulate_temp(t_test, rf_test)
    plot_T_diff(t_test, T_diff)
    show_plot("t10a")

    # Task 10b:
    # - "nalyze the effect of the climate sensitivity parameter 𝜆 and the exchange coefficient κ on the time required to reach equilibrium temperature"
    plt.title("Temperature response for different lambdas (Task T0b)")
    lambda_params = np.array([0.5, 0.8, 1.3])
    for idx in range(len(lambda_params)):
      lambda_param = lambda_params[idx]
      color = colors[idx]
      T_diff, _ = simulate_temp(t_test, rf_test, lambda_param=lambda_param)
      plot_T_diff(t_test, T_diff, lambda_param=lambda_param, color=color, label_suffix=f", lambda={lambda_param}")
    show_plot("t10b")

    plt.title("Temperature response for different kappas (Task T0b)")
    lambda_param = 0.8
    kappas = np.array([0.2, 0.5, 1])
    for idx in range(len(kappas)):
      kappa = kappas[idx]
      color = colors[idx]
      T_diff, _ = simulate_temp(t_test, rf_test, kappa=kappa)
      plot_T_diff(t_test, T_diff, kappa=kappa, color=color, label_suffix=f", kappa={kappa}")
    show_plot("t10b2")

    # TODO: Task 10c
    # Q: Analyze energy fluxes
    # A: Net flux is 0 for all time_steps, that is energy is conserved!
    #    We can also see that the heat radiation flux and the heat energy flux are perfectly mirrored against y=0.5, which makes sense because they both should add up to 1. 
    for idx in range(2*len(kappas)):
      kappa = kappas[idx if idx < 3 else 1]
      lambda_param = kappas[idx-3 if idx >= 3 else 1]
      color = colors[idx % 3]
      T_diff, dTdt = simulate_temp(t_test, rf_test, kappa=kappa, lambda_param=lambda_param)
      heat_energy_flux = np.array([C[0]*dTdt[:,0], C[1]*dTdt[:,1]]) # (W yr K⁻¹ m⁻²) * (K/yr) = W/m²
      total_heat_energy_flux = heat_energy_flux[0] + heat_energy_flux[1]
      print("rf test", rf_test)
      in_flux = rf_test # W/m²
      out_flux = T_diff[:, 0] / lambda_param # K / (K W⁻¹ m²) = W/m²
      net_flux = in_flux - out_flux - total_heat_energy_flux
      print(in_flux)
      plt.plot(t_test, total_heat_energy_flux, linestyle=":", color=color, label=f"heat energy flux kappa={kappa}, lambda={lambda_param}")
      plt.plot(t_test, out_flux, color=color, linestyle="-.", label=f"out, kappa={kappa}, lambda={lambda_param}")
      plt.plot(t_test, net_flux, color="black", label=f"net flux" if idx == 2 or idx == 5 else None)
      if idx == 2 or idx == 5:
        plt.plot(t_test, in_flux, color="gray", label=f"in")
        plt.xlabel("Years")
        plt.ylabel("W/m²")
        if idx == 2:
          plt.title(f"Energy flux, lambda={lambda_param} (task T0c)")
        else:
          plt.title(f"Energy flux, kappa={kappa} (task T0c)")
        plt.legend()
        show_plot(f"t10c{(idx+1)//3}")
    
  if args.q in (0, 8): task8()
  if args.q in (0, 9): task9()
  if args.q in (0, 10): task10()

def part3():
  # TODO: Task 11a
  # TODO: Task 11b
  # TODO: Task 11c
  def task11():
    betas = [0.25, 0.35, 0.45]
    ks = [1e-3, k_default, 5e-6]
    s = 1
    
    B = simulate_carbon_balance()

    # Use .csv-data for rf_co2 from 1765 to 2024, then we use the rf_co2 acquired through simulation after 2024, but the radiative forcing from aerosols and other sources are given by the .csv-files for all time_steps.
    P_co2 = co2_per_gtc * B[:,0]
    rf_co2 = calc_rf_co2(P_co2)
    t2024 = 2024 - start_year
    print(t2024)
    rf_co2[t2024:] = rf_data_co2[t2024]
    rf_net = rf_co2 + s * rf_data_aerosols + rf_data_other
    
    T_diff, dTdt = simulate_temp(time_steps, rf_net)

    plt.plot(absolute_time_steps, T_diff[:, 0], label="Simulation")
    plt.plot(T_diff_data_t, T_diff_data, label="NASA's estimate")
    plt.legend()
    plt.title("(Task T1)")
    show_plot("t11a")
    
  def task12():

    ## Task 12a
    # - Use lambda=0.8 and use kappa & s that were a good fit from task 11
    # - Generate and present temperature projections from 1765 to 2100 under the following scenarios:
    #   i. "CO2 decrease linearly to zero by 2070" and continue to decrease(negative) til 2100. After 2100 CO2 emissions are constant.
    #   ii. CO2 emissions remain constant with current levels
    #   iii. CO2 emissions increase linearly until 2100 where it reach 200% of current emissions. After 2100 emissions are constant.
    lambda_param = 0.8 # (K/(W/m²))
    kappas = 1234
    s = 1234

    ## Task 12b
    # Q: "How much does the temperature increase over the century relative to pre-industrial levels in the three cases i–iii? Illustrate in a figure."
    # A:

    ## Task 12c:
    # Q: "How large is the spread in temperature responses by 2100 in the three cases i–iii, given the uncertainty in the climate sensitivity parameter λ (along with estimates of s and κ that provide a good fit to historical temperatures)? Illustrate in a figure"
    # A:

    ## Task 12d:
    # Q: "s you may now observe, temperature projections for 2100 can vary significantly due to both uncertainties in the climate system (mainly due to   λ , but also   s   and   κ ) and our choices regarding CO₂ emissions. Given this uncertainty, how should we think about how much CO₂ emissions should be reduced in the coming decades to meet the climate goals of the Paris Agreement? Use the Konjunkturrådets report and texts on sustainable development as inspiration for your reasoning"     
    # A:

    ## Task 12e:
    # Q: "The low emissions scenario has negative CO2 emissions towards the end of the century. How can such negative CO2 emissions be materialised? Negative CO2 emissions tend to lead to a peak and decline in global mean surface temperature (make a plot to verify that this is the case). In what way are such temperature pathways relevant for reaching the temperature goals stated in the Paris Agreement?" 
    # A:

  # def task13():
    ## Task 13a:
    # Q: "Simulating the Termination of Geoengineering. Analyze what would happen if geoengineering were  implemented for a period but then, for some reason, abruptly stopped. Test a scenario where incoming solar radiation is reduced by 4 W/m² from 2050 to 2100. What would happen to global mean temperature in such a scenario (over the entire period up to 2200)?"
    # A:

    ## Task 13b:
    # Q: "Geoengineering from a Sustainable Development Perspective. Critically discuss the feasibility of  geoengineering as a climate change response from the perspective of sustainable development. In your discussion, compare geoengineering with strategies for reducing CO₂ emissions in the energy system, including the rol e of negative CO₂ emissions. Evaluate the extent to which geoengineering is consistent with long-term climate objectives and the core principles of sustainable development. Draw on course materials, including the texts on sustainable development, to frame and support your analysis."   
    # A:
    
  if args.q in (0, 11): task11()


## Do tasks
if args.p in (0,1): part1()
if args.p in (0,2): part2()
if args.p in (0,3): part3()
  

