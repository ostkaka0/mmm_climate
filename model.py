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
# 
################################################################################
################################################################################

## Load data from .csv-files
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


## Tasks
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
  k = 3.06e-3
  A = [0.113, 0.213, 0.258, 0.273, 0.1430]
  tau0 = [2.0, 12.2, 50.4, 243.3, 1e12]
  n = 5

  # def tau(i, t):
  #   return tau0[i] * (1 + k * np.sum(U(s) for s in range(0, t)))

  def tau_by_sum(i, cumulative_U):
    return tau0[i] * (1 + k * cumulative_U)
  
  # def impulse_control(t):
  #   return sum(A[i] * np.exp(-t / tau(i, t)) for i in range(n))

  def impulse_control_by_sum(t, cumulative_U):
    return sum(A[i] * np.exp(-t / tau_by_sum(i, cumulative_U)) for i in range(n))

  for cumulative_U in (0, 140, 560, 1680):
    t = np.arange(501)
    print(t)
    plt.plot(t, impulse_control_by_sum(t, cumulative_U), label=cumulative_U)
  plt.show()

## Parse arguments
parser = argparse.ArgumentParser();
# Question/task
parser.add_argument(
  "-q",
  type=int,
  default=0
)

args = parser.parse_args()

## Do tasks
if args.q in (0, 1): task1()
if args.q in (0, 2): task2()
if args.q in (0, 3): task3()
# if args.q in (0, 4): task4()

