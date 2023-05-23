from numpy import linspace, divide
import pandas as pd
from multiprocessing.pool import Pool
from Ising import Ising

lattice = 16
iterations = 1500


def make_whole_simulation(T):
    model = Ising(T, lattice, iterations, prop="antiferro", mesh="hex", plots=False)
    result = divide(model.simulate(), (iterations * lattice * lattice))
    return {"Temperature": T, "Energy": result[0], "Magnetization": result[1]}


if __name__ == "__main__":
    temperature_points = 1000

    df = pd.DataFrame(columns=["Temperature", "Energy", "Magnetization"])
    T_sample = linspace(0.01, 1.5, temperature_points)

    with Pool() as pool:
        for result in pool.imap(make_whole_simulation, T_sample):
            df = df.append(result, ignore_index=True)
    df.sort_values(by=["Temperature"], inplace=True)

    df.to_csv(
        f"Ising/Results/Antiferro_Hex_{lattice}_{iterations}_Simulation_{temperature_points}.csv",
        index=False,
    )
