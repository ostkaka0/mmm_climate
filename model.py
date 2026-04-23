#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

def solve_euler(dfdt, t, f0, args=None):
  f = np.zeros((len(t), *np.shape(f0)))
  f[0] = f0
  for i, t_i in enumerate(t):
    if i == 0: continue
    dt = t[i] - t[i-1]
    f[i] = f[i-1] + dfdt(t_i, i, f[i-1], f0, *args) * dt
  return f

################################################################################
# Biosphere uptake of CO2:
################################################################################
# Using a box model we have:
# - Box 1: Carbon in atmosphere
# - Box 2: Carbon in biomass above ground
# - Box 3: Carbon in biomass below ground
# B[i] - Box i.
# F[i, j] - Annual flow from box i to box j 
# B0 - Sizes of boxes pre-industrial time
# F0 - Annual flow pre-industrial time

B0 = np.array([600, 600, 1500])
F0 = np.array([
  [0, 60, 0],
  [15, 0, 45],
  [45, 0, 0],
])
NPP0 = F0[0, 1]
print("NPP0", NPP0)

alpha = np.zeros((3, 3))
for i in range(3):
  for j in range(3):
    alpha[i, j] = F0[i, j] / B0[i]

# NPP(Net Primary Production) - Flow from atmosphere to above-ground biomass
# beta - Fertilization factor
def NPP(NPP0, B, B0, beta):
  return NPP0*(1 + beta * np.log(B[0] / B0[0]))

# dB/dt
# t_i - Time at discrete time point i
# i   - Discrete time point (0, 1, 2, ...)
# U   - Emission indexed by i
# alpha[i,j[i,j]] = F0[i,j] / B0[i]
# beta - Fertilization factor
def dBdt(t_i, i, B, B0, NPP0, U, alpha, beta):
  NPP_val = NPP(NPP0, B, B0, beta)
  return np.array([
    alpha[2, 0]*B[2] + alpha[1, 0]*B[1] - NPP_val + U[i],
    NPP_val - alpha[1, 2]*B[1] - alpha[1, 0]*B[1],
    alpha[1, 2]*B[1] - alpha[2, 0]*B[2]
  ])

################################################################################
# Impulse response function
################################################################################
k = 3.06e-3
A = [0.113, 0.213, 0.258, 0.273, 0.1430]
tau0 = [2.0, 12.2, 50.4, 243.3, 1e12]
n = 5
M0 = B0[0] # Carbon stock in atmosphere before industrialization

# def tau(i, t):
#   return tau0[i] * (1 + k * np.sum(U(s) for s in range(0, t)))

# def tau_by_sum(i, cumulative_U):
#   return tau0[i] * (1 + k * cumulative_U)

def tau(i, s_idx, cumulative_U):
  return tau0[i] * (1 + k * cumulative_U[s_idx-1])

# def impulse_control(t):
#   return sum(A[i] * np.exp(-t / tau(i, t)) for i in range(n))

# def impulse_control_by_sum(t, cumulative_U):
#   return sum(A[i] * np.exp(-t / tau_by_sum(i, cumulative_U)) for i in range(len(len(tau0))))

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
# Emission data
emission_data = np.genfromtxt("utslappRCP45.csv", delimiter=",")
t = np.arange(emission_data[1, 0], emission_data[-1, 0]+1)
U = emission_data[1:, 1]
# Co2 concentration data
co2_concentration_data = np.genfromtxt("koncentrationerRCP45.csv", delimiter=",")
t2 = co2_concentration_data[1:, 0]
y = co2_concentration_data[1:, 1]

print("emission_data")
print(emission_data)
print("co2_concentration_data")
print(co2_concentration_data)

################################################################################
# Tasks
################################################################################
# Constants
co2_per_gtc = 0.469

# Task 1:
# - Construct a model for the carbon cycle
# - Analyze how flows between differnt boxes are affected by emission_data.
# - Answer "Why do you think your calculated concentrations differ?"
def task1():
  # Solve B
  beta = 0.35
  B = solve_euler(dBdt, t, B0, args=(NPP0, U, alpha, beta))
  # Calculate atomstpheric CO2 concentrations
  co2 = co2_per_gtc * B[:,0]
  # Plot CO2 according to model and data
  plt.plot(t, co2, label=f"beta={beta}")
  plt.plot(t2, y, "black", label="koncentrationerRCP45.csv")
  plt.title("CO²")
  plt.legend()
  plt.show()

# Same as task 1, but with the solve-euler function expanded
def task1_alt():
  beta = 0.35
  B = np.zeros((len(t), *np.shape(B0)))
  B[0] = B0
  # Loop for euler method:
  for t_idx, t_val in enumerate(t):
    if t_idx == 0: continue
    dt = t[t_idx] - t[t_idx-1]
    B[t_idx] = B[t_idx-1] + dBdt(t_val, t_idx, B[t_idx-1], B0, *args) * dt

  # Calculate atomstpheric CO2 concentrations
  co2 = co2_per_gtc * B[:,0]
  # Plot CO2 according to model and data
  plt.plot(t, co2, label=f"beta={beta}")
  plt.plot(t2, y, "black", label="koncentrationerRCP45.csv")
  plt.title("CO²")
  plt.legend()
  plt.show()

# Task 2:
# - Same as Task 1, but we vary beta(CO2 fertilization factor)
# - Describe what happens to the CO2 and carbon biomass
# - "Explain the results by considering how an increased or decreased fertilization effect influences net primary production (NPP), carbon uptake by vegetation, and overall carbon cycling between the atmosphere, biosphere, and soil."
def task2():
  # Solve B for different values of beta, and plot atmospheric CO2 concentrations
  for beta in np.linspace(0.1, 0.8, 15):
    B = solve_euler(dBdt, t, B0, args=(NPP0, U, alpha, beta))
    co2 = co2_per_gtc * B[:,0]
    plt.plot(t, co2, label=f"beta={beta}")
  # Plot CO2 concentration data for comparison
  plt.plot(t2, y, "black", label="koncentrationerRCP45.csv")
  plt.title("CO²")
  plt.legend()
  plt.show()


  

def task3():
  for cumulative_U_val in (0, 140, 560, 1680):
    cumulative_U = [cumulative_U_val]
    t = np.arange(501)
    print(t)
    I_vals = np.zeros_like(t)

    for t_idx, t_val in enumerate(t):
      I_vals[i] = I(t_val, 0, cumulative_U)

    plt.plot(t, I_vals, label=cumulative_U)
  plt.show()

def task4():
  M_vals = np.zeros_like(t)
  cumulative_U = np.zeros_like(t)
  # Set initial values
  M_vals[0] = M0
  cumulative_U[0] = U[0]

  # Simulate
  for t_idx, t_val in enumerate(t[1:], 1): # Note that we simulate for t=0
    cumulative_U[t_idx] = cumulative_U[t_idx - 1] + U[t_idx]
    M_vals[t_idx] = M(t, t, t_idx, cumulative_U)

  co2 = co2_per_gtc * M_vals
  plt.plot(t, co2, label="A")
  plt.plot(t2, y, "black", label="koncentrationerRCP45.csv")
  plt.title("CO²")
  plt.show()

# TODO: Adjust beta so that model-calculated concentrationsn align with kocntrationerRCP45.csv
# TODO: Task 7
def task6_and_7():
  ## Task 6
  # global t
  # global t2
  # global y
  # # Use a subset of t
  # t = t[0:300]
  # t2 = t2[0:300]
  # y = t[0:300]
  # Consts
  beta = 0.35

  # Simulation outputs  
  cumulative_U = np.zeros_like(t)
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
  plt.plot(t2, y, "black", label="koncentrationerRCP45.csv")
  plt.title("CO²")
  plt.legend()
  plt.show()

  ## Task 7
  plt.plot(t, B[:,0] / B0[0], label="1")
  plt.plot(t, B[:,1] / B0[1], label="2")
  plt.plot(t, B[:,2] / B0[2], label="3 / 10")
  # plt.plot(t2, (y/y[0]), "black", label="koncentrationerRCP45.csv")
  plt.title("carbon")
  plt.legend()
  plt.show()



#   # Simulate
#   M_vals = np.zeros_like(t)
#   for t_idx, t_val in enumerate(t):
#     M_vals[t_idx] = M(t, t, t_idx, cumulative_U)

#   # Simulate
#   for t_idx, t_val in enumerate(t):
    
    
  

## Parse arguments
parser = argparse.ArgumentParser();
# Question/task
parser.add_argument(
  "-q",
  default="0"
)

args = parser.parse_args()

## Do tasks
if args.q in ("0", "1"): task1()
if args.q in ("1alt"  ): task1_alt()
if args.q in ("0", "2"): task2()
if args.q in ("0", "3"): task3()
if args.q in ("0", "4"): task4()
if args.q in ("0", "6", "7"): task6_and_7()

