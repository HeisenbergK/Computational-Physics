from numpy import exp, meshgrid, sum, linspace, diff, asarray, sqrt, round, empty, nan
from numpy.random import rand, randint
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.tri as tri
from collections import Counter


class Ising:
    """
    Ising Model class.
    We take in init:
        temperature: the temperature of the crystal in our own units
        lattice: the lattice size (always a square shape)
        iterations: the iterations the user wants to run in modeling
        mesh: the lattice shape (ortho/hex)
        prop: the phenomenon of the lattice (ferro/antiferro)
        plots: whether the user wants plots (True), no plots (False) or animation (animate)
    """

    # initialization
    def __init__(
        self, temperature, lattice, iterations, mesh="ortho", prop="ferro", plots=True
    ):
        self.temperature = temperature
        self.lattice = lattice
        self.iterations = iterations
        # Initialize crystal with random up/down (1/-1) spins
        self.crystal = 2 * randint(2, size=(lattice, lattice)) - 1

        if plots:
            if plots == "animate":
                self.plots = True
                self.animate = True
            else:
                self.plots = True
                self.animate = False
        else:
            self.plots = False
            self.animate = False

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

    # Is x even
    def IsEven(self, x):
        if x % 2:
            return False
        else:
            return True

    # Energy of a given crystal state
    def E(self):
        lattice = self.lattice  # I'm lazy
        energy = 0  # initialize energy
        # loop into all crystal positions
        for i in range(len(self.crystal)):
            for j in range(len(self.crystal)):
                S = self.crystal[i, j]  # valye at xy
                nb = 0  # initialize the sum of the values of neighbors
                # add the value of each neighbor to the sum, depending on the mesh shape
                if self.mesh == "ortho" or self.mesh == "hex":
                    nb += (
                        self.crystal[(i + 1) % lattice, j]
                        + self.crystal[i, (j + 1) % lattice]
                        + self.crystal[(i - 1) % lattice, j]
                        + self.crystal[i, (j - 1) % lattice]
                    )
                if self.mesh == "hex" and (self.IsEven(i + 1)):
                    nb += (
                        self.crystal[(i - 1) % lattice, (j - 1) % lattice]
                        + self.crystal[(i + 1) % lattice, (j - 1) % lattice]
                    )
                elif self.mesh == "hex":
                    nb += (
                        self.crystal[(i + 1) % lattice, (j + 1) % lattice]
                        + self.crystal[(i - 1) % lattice, (j + 1) % lattice]
                    )

                # add the appropriate value to the running sum
                if self.prop == "ferro":
                    energy += -nb * S
                elif self.prop == "antiferro":
                    energy += nb * S

        # divide by the degrees of freedom to acquire a comparable value
        if self.mesh == "hex":
            return energy / 6.0
        elif self.mesh == "ortho":
            return energy / 4.0

    # Magnetization of a given crystal state
    def M(self):
        mag = sum(self.crystal)
        return mag

    # Metropolis MCMC step
    def mcStep(self):
        beta = 1.0 / self.temperature  # beta parameter, units k_B=J=1
        lattice = self.lattice  # I'm lazy
        for i in range(lattice):
            for j in range(lattice):
                a = randint(0, lattice)  # random x-coordinate in the lattice
                b = randint(0, lattice)  # random y-coordinate in the lattice
                s = self.crystal[a, b]  # value of the crystal at xy
                nb = 0  # initialize the sum of the values of neighbors
                # add the value of each neighbor to the sum, depending on the mesh shape
                if self.mesh == "ortho" or self.mesh == "hex":
                    nb += (
                        self.crystal[(a + 1) % lattice, b]
                        + self.crystal[a, (b + 1) % lattice]
                        + self.crystal[(a - 1) % lattice, b]
                        + self.crystal[a, (b - 1) % lattice]
                    )
                if self.mesh == "hex" and (self.IsEven(a + 1)):
                    nb += (
                        self.crystal[(a - 1) % lattice, (b - 1) % lattice]
                        + self.crystal[(a + 1) % lattice, (b - 1) % lattice]
                    )
                elif self.mesh == "hex":
                    nb += (
                        self.crystal[(a + 1) % lattice, (b + 1) % lattice]
                        + self.crystal[(a - 1) % lattice, (b + 1) % lattice]
                    )

                # compute the cost metric (energy gain/loss if spin at xy is swapped)
                if self.prop == "ferro":
                    metric = 2 * s * nb
                elif self.prop == "antiferro":
                    metric = -2 * s * nb

                # if the cost is negative, make the swap
                if metric < 0:
                    s *= -1
                # if the cost isnt negative, make the swap only if thermal beats hamilton
                elif rand() < exp(-metric * beta):
                    s *= -1
                # assign change
                self.crystal[a, b] = s

    # Simulate the crystal
    def simulate(self):
        E = 0  # Initialize energy
        M = 0  # Initialize magnetization

        # If plots, add initial state to plot, if animation, save initial frame
        if self.plots and (not self.animate) and self.mesh == "hex":
            f = plt.figure(figsize=(15, 15))
            self.configPlotHex(f, 0, 1)
        elif self.plots and (not self.animate) and self.mesh == "ortho":
            f = plt.figure(figsize=(15, 15))
            self.configPlot(f, 0, 1)
        elif self.plots and (self.animate) and self.mesh == "hex":
            self.savePlotHex(0)
        elif self.plots and (self.animate) and self.mesh == "ortho":
            self.savePlot(0)

        # Loop Metropolis iteration times
        for i in range(self.iterations):
            # do one step to the crystal
            self.mcStep()

            # If plots, add plot at times 1,4,32,100,iterations end, if animation, save frame at every iteration
            if self.plots and (not self.animate) and self.mesh == "hex":
                if i == 1:
                    self.configPlotHex(f, i, 2)
                if i == 4:
                    self.configPlotHex(f, i, 3)
                if i == 32:
                    self.configPlotHex(f, i, 4)
                if i == 100:
                    self.configPlotHex(f, i, 5)
                if i == self.iterations - 1:
                    self.configPlotHex(f, i, 6)
            elif self.plots and (not self.animate) and self.mesh == "ortho":
                if i == 1:
                    self.configPlot(f, i, 2)
                if i == 4:
                    self.configPlot(f, i, 3)
                if i == 32:
                    self.configPlot(f, i, 4)
                if i == 100:
                    self.configPlot(f, i, 5)
                if i == self.iterations - 1:
                    self.configPlot(f, i, 6)
            elif self.plots and (self.animate) and self.mesh == "hex":
                self.savePlotHex(i)
            elif self.plots and (self.animate) and self.mesh == "ortho":
                self.savePlot(i)

            E += self.E()  # add energy to total
            M += self.M()  # add magnetization to total

        # show the plots if done
        if self.plots:
            plt.show()

        # return the energy and magnetization for set temperature, after all iterations
        return [E, M]

    # configure hex plot
    def configPlotHex(self, f, i, n_):
        nx = self.lattice
        ny = self.lattice
        cmap = colors.ListedColormap(["black", "white"])
        bounds = [0, 1, 2]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        x = linspace(0, 1, nx)
        y = linspace(0, 1, ny)

        dx = diff(x)[0]

        patches = []
        for k in x:
            for n, j in enumerate(y):
                if not self.IsEven(n):
                    polygon = mpatches.RegularPolygon([k - dx / 2.0, j], 6, 0.6 * dx)
                else:
                    polygon = mpatches.RegularPolygon([k, j], 6, 0.6 * dx)
                patches.append(polygon)

        collection = PatchCollection(patches, cmap=cmap, norm=norm, alpha=1.0)

        sp = f.add_subplot(3, 3, n_)
        sp.set_xticks([])
        sp.set_yticks([])
        sp.add_collection(collection)
        collection.set_array(self.crystal.ravel())
        sp.set_title("Time=%d" % i)
        sp.axis("tight")

    # make a single plot
    def configPlot(self, f, i, n_):
        X, Y = meshgrid(range(self.lattice), range(self.lattice))
        sp = f.add_subplot(3, 3, n_)
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)
        plt.pcolormesh(X, Y, self.crystal, cmap=plt.cm.binary)
        plt.title("Time=%d" % i)
        plt.axis("tight")

    # configure hex plot
    def savePlotHex(self, i):
        nx = self.lattice
        ny = self.lattice
        cmap = colors.ListedColormap(["black", "white"])
        bounds = [0, 1, 2]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        x = linspace(0, 1, nx)
        y = linspace(0, 1, ny)

        dx = diff(x)[0]

        patches = []
        for k in x:
            for n, j in enumerate(y):
                if not self.IsEven(n):
                    polygon = mpatches.RegularPolygon([k - dx / 2.0, j], 6, 0.6 * dx)
                else:
                    polygon = mpatches.RegularPolygon([k, j], 6, 0.6 * dx)
                patches.append(polygon)

        collection = PatchCollection(patches, cmap=cmap, norm=norm, alpha=1.0)

        fig, ax = plt.subplots(1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_collection(collection)
        collection.set_array(self.crystal.ravel())
        ax.set_title("Time=%d" % i)
        ax.axis("tight")
        fig.savefig(f"Animation/{i:04d}.png")

    # make a single plot
    def savePlot(self, i):
        fig, ax = plt.subplots(1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        X, Y = meshgrid(range(self.lattice), range(self.lattice))
        ax.pcolormesh(X, Y, self.crystal, cmap=plt.cm.binary)
        ax.set_title("Time=%d" % i)
        ax.axis("tight")
        fig.savefig(f"Animation/{i:04d}.png")
