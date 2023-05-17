from numpy import exp, meshgrid, sum, linspace, divide
from numpy.random import rand, randint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from multiprocessing.pool import Pool


class Ising:
    # initialization
    def __init__(self, temperature, lattice, iterations, plots=True):
        self.temperature = temperature
        self.lattice = lattice
        self.iterations = iterations
        self.plots = plots

    # energy of the crystal
    def E(self, crystal, lattice):
        energy = 0
        for i in range(len(crystal)):
            for j in range(len(crystal)):
                S = crystal[i, j]
                nb = (
                    crystal[(i + 1) % lattice, j]
                    + crystal[i, (j + 1) % lattice]
                    + crystal[(i - 1) % lattice, j]
                    + crystal[i, (j - 1) % lattice]
                )
                if (i + 1) % 2 == 0:
                    nb += (
                        crystal[(i - 1) % lattice, (j - 1) % lattice]
                        + crystal[(i + 1) % lattice, (j - 1) % lattice]
                    )
                else:
                    nb += (
                        crystal[(i + 1) % lattice, (j + 1) % lattice]
                        + crystal[(i - 1) % lattice, (j + 1) % lattice]
                    )
                energy += nb * S
        return energy / 6.0

    # magnetization of crystal
    def M(self, crystal):
        mag = sum(crystal)
        return mag

    # monte carlo steps
    def mcStep(self, crystal, lattice, beta):
        for i in range(lattice):
            for j in range(lattice):
                a = randint(0, lattice)
                b = randint(0, lattice)
                s = crystal[a, b]
                nb = (
                    crystal[(a + 1) % lattice, b]
                    + crystal[a, (b + 1) % lattice]
                    + crystal[(a - 1) % lattice, b]
                    + crystal[a, (b - 1) % lattice]
                )
                if (a + 1) % 2 == 0:
                    nb += (
                        crystal[(a - 1) % lattice, (b - 1) % lattice]
                        + crystal[(a + 1) % lattice, (b - 1) % lattice]
                    )
                else:
                    nb += (
                        crystal[(a + 1) % lattice, (b + 1) % lattice]
                        + crystal[(a - 1) % lattice, (b + 1) % lattice]
                    )
                metric = -2 * s * nb
                if metric < 0:
                    s *= -1
                elif rand() < exp(-metric * beta):
                    s *= -1
                crystal[a, b] = s
        return crystal

    # simulate the crystal
    def simulate(self):
        E = 0
        M = 0
        crystal = 2 * randint(2, size=(self.lattice, self.lattice)) - 1

        if self.plots:
            f = plt.figure(figsize=(15, 15))
            self.configPlot(f, crystal, 0, self.lattice, 1)

        for i in range(self.iterations):
            self.mcStep(crystal, self.lattice, 1.0 / self.temperature)
            if self.plots:
                if i == 1:
                    self.configPlot(f, crystal, i, self.lattice, 2)
                if i == 4:
                    self.configPlot(f, crystal, i, self.lattice, 3)
                if i == 32:
                    self.configPlot(f, crystal, i, self.lattice, 4)
                if i == 100:
                    self.configPlot(f, crystal, i, self.lattice, 5)
                if i == self.iterations - 1:
                    self.configPlot(f, crystal, i, self.lattice, 6)
            E += self.E(crystal, self.lattice)
            M += self.M(crystal)

        if self.plots:
            plt.show()

        return [E, M]

    # make a single plot
    def configPlot(self, f, crystal, i, lattice, n_):
        X, Y = meshgrid(range(lattice), range(lattice))
        X, Y = X * 2, Y * 2

        # Turn this into a hexagonal grid
        for ii, k in enumerate(X):
            if ii % 2 == 1:
                X[ii] += 1
                Y[:, ii] += 1

        sp = f.add_subplot(3, 3, n_)
        sp.set_xticks([])
        sp.set_yticks([])
        im = sp.hexbin(
            X.reshape(-1),
            Y.reshape(-1),
            C=crystal.reshape(-1),
            gridsize=int(crystal.shape[0] / 2),
            cmap=plt.cm.binary,
            linewidths=0.0,
        )
        sp.set_title("Time=%d" % i)
        sp.axis("tight")
        divider = make_axes_locatable(sp)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        f.colorbar(im, cax=cax, orientation="vertical")


def make_whole_simulation(T, lattice=16, iterations=1500):
    model = Ising(T, lattice, iterations, False)
    result = divide(model.simulate(), (iterations * lattice * lattice))
    return {"Temperature": T, "Energy": result[0], "Magnetization": result[1]}


if __name__ == "__main__":
    temperature_points = 1000

    df = pd.DataFrame(columns=["Temperature", "Energy", "Magnetization"])
    T_sample = linspace(0.01, 5.5, temperature_points)
    lattice = 16
    iterations = 1500

    with Pool() as pool:
        for result in pool.imap(make_whole_simulation, T_sample):
            df = df.append(result, ignore_index=True)
    df.sort_values(by=["Temperature"], inplace=True)

    df.to_csv(
        f"Ising/Results/Antiferro_Hex_{lattice}_{iterations}_Simulation_{temperature_points}.csv",
        index=False,
    )
