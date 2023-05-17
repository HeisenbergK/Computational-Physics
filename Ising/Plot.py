import matplotlib.pyplot as plt
import pandas as pd
from numpy import absolute

df = pd.read_csv("Ising/Results/Antiferro_Hex_16_1500_Simulation_1000.csv")

plt.clf()
plt.plot(df["Temperature"], absolute(df["Magnetization"]), "c.")
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.show()

plt.clf()
plt.plot(df["Temperature"], (df["Energy"]), ".")
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.show()
