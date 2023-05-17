from numpy import exp, meshgrid, sum, linspace, diff
from numpy.random import rand, randint
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


class Ising:
    # initialization
    def __init__(
        self, temperature, lattice, iterations, mesh="ortho", prop="ferro", plots=True
    ):
        self.temperature = temperature
        self.lattice = lattice
        self.iterations = iterations
        self.plots = plots

        if mesh == "ortho":
            self.mesh = "ortho"
        elif mesh == "hex":
            self.mesh = "hex"
        else:
            raise Exception("Incorrect mesh")

        if prop == "ferro":
            self.prop = "ferro"
        elif prop == "antiferro":
            self.prop = "antiferro"
        else:
            raise Exception("Incorrect property")

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
                if self.mesh == "hex" and ((i + 1) % 2 == 0):
                    nb += (
                        crystal[(i - 1) % lattice, (j - 1) % lattice]
                        + crystal[(i + 1) % lattice, (j - 1) % lattice]
                    )
                elif self.mesh == "hex":
                    nb += (
                        crystal[(i + 1) % lattice, (j + 1) % lattice]
                        + crystal[(i - 1) % lattice, (j + 1) % lattice]
                    )

                if self.prop == "ferro":
                    energy += -nb * S
                elif self.prop == "antiferro":
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
                if self.mesh == "hex" and ((a + 1) % 2 == 0):
                    nb += (
                        crystal[(a - 1) % lattice, (b - 1) % lattice]
                        + crystal[(a + 1) % lattice, (b - 1) % lattice]
                    )
                elif self.mesh == "hex":
                    nb += (
                        crystal[(a + 1) % lattice, (b + 1) % lattice]
                        + crystal[(a - 1) % lattice, (b + 1) % lattice]
                    )

                if self.prop == "ferro":
                    metric = 2 * s * nb
                elif self.prop == "antiferro":
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

        if self.plots and self.mesh == "hex":
            f = plt.figure(figsize=(15, 15))
            self.configPlotHex(f, crystal, 0, self.lattice, 1)
        elif self.plots and self.mesh == "ortho":
            f = plt.figure(figsize=(15, 15))
            self.configPlot(f, crystal, 0, self.lattice, 1)

        for i in range(self.iterations):
            self.mcStep(crystal, self.lattice, 1.0 / self.temperature)
            if self.plots and self.mesh == "hex":
                if i == 1:
                    self.configPlotHex(f, crystal, i, self.lattice, 2)
                if i == 4:
                    self.configPlotHex(f, crystal, i, self.lattice, 3)
                if i == 32:
                    self.configPlotHex(f, crystal, i, self.lattice, 4)
                if i == 100:
                    self.configPlotHex(f, crystal, i, self.lattice, 5)
                if i == self.iterations - 1:
                    self.configPlotHex(f, crystal, i, self.lattice, 6)
            elif self.plots and self.mesh == "ortho":
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

    # configure hex plot
    def configPlotHex(self, f, crystal, i, lattice, n_):
        nx = lattice
        ny = lattice
        cmap = colors.ListedColormap(["black", "white"])
        bounds = [0, 1, 2]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        x = linspace(0, 1, nx)
        y = linspace(0, 1, ny)

        dx = diff(x)[0]

        patches = []
        for k in x:
            for n, j in enumerate(y):
                if n % 2:
                    polygon = mpatches.RegularPolygon([k - dx / 2.0, j], 6, 0.6 * dx)
                else:
                    polygon = mpatches.RegularPolygon([k, j], 6, 0.6 * dx)
                patches.append(polygon)

        collection = PatchCollection(patches, cmap=cmap, norm=norm, alpha=1.0)

        sp = f.add_subplot(3, 3, n_)
        sp.set_xticks([])
        sp.set_yticks([])
        sp.add_collection(collection)
        collection.set_array(crystal.ravel())
        sp.set_title("Time=%d" % i)
        sp.axis("tight")

    # make a single plot
    def configPlot(self, f, crystal, i, lattice, n_):
        X, Y = meshgrid(range(lattice), range(lattice))
        sp = f.add_subplot(3, 3, n_)
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)
        plt.pcolormesh(X, Y, crystal, cmap=plt.cm.binary)
        plt.title("Time=%d" % i)
        plt.axis("tight")
