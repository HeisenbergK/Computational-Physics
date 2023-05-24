from numpy import exp, meshgrid, sum, linspace, diff, asarray, sqrt, round, empty, nan
from numpy.random import rand, randint
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.tri as tri
from collections import Counter


class Ising:
    # initialization
    def __init__(
        self, temperature, lattice, iterations, mesh="ortho", prop="ferro", plots=True
    ):
        self.temperature = temperature
        self.lattice = lattice
        self.iterations = iterations

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
        elif mesh == "tri":
            self.mesh = "tri"
        else:
            raise Exception("Incorrect mesh")

        if prop == "ferro":
            self.prop = "ferro"
        elif prop == "antiferro":
            self.prop = "antiferro"
        else:
            raise Exception("Incorrect property")

    # is value even
    def IsEven(self, x):
        if x % 2:
            return False
        else:
            return True

    # in a triangular model, get the corners from x and y
    def CornersFromXY(self, x, y):
        if self.IsEven(x):
            if self.IsEven(y):
                return [[x + 1, y], [x, y + 1], [x + 2, y + 1]]
            else:
                return [[x + 1, y + 1], [x, y], [x + 2, y]]
        else:
            if self.IsEven(y):
                return [[x + 1, y + 1], [x, y], [x + 2, y]]
            else:
                return [[x + 1, y], [x, y + 1], [x + 2, y + 1]]

    # energy of the crystal
    def E(self, crystal, lattice):
        energy = 0
        for i in range(len(crystal)):
            for j in range(len(crystal)):
                S = crystal[i, j]
                nb = 0
                if self.mesh == "ortho" or self.mesh == "hex":
                    nb += (
                        crystal[(i + 1) % lattice, j]
                        + crystal[i, (j + 1) % lattice]
                        + crystal[(i - 1) % lattice, j]
                        + crystal[i, (j - 1) % lattice]
                    )
                if self.mesh == "hex" and (self.IsEven(i + 1)):
                    nb += (
                        crystal[(i - 1) % lattice, (j - 1) % lattice]
                        + crystal[(i + 1) % lattice, (j - 1) % lattice]
                    )
                elif self.mesh == "hex":
                    nb += (
                        crystal[(i + 1) % lattice, (j + 1) % lattice]
                        + crystal[(i - 1) % lattice, (j + 1) % lattice]
                    )

                if self.mesh == "tri" and (self.IsEven(i + j)):
                    nb += (
                        crystal[(i - 1) % lattice, j]
                        + crystal[(i + 1) % lattice, j]
                        + crystal[i, (j + 1) % lattice]
                    )
                elif self.mesh == "tri ":
                    nb += (
                        crystal[(i - 1) % lattice, j]
                        + crystal[(i + 1) % lattice, j]
                        + crystal[i, (j - 1) % lattice]
                    )

                if self.prop == "ferro":
                    energy += -nb * S
                elif self.prop == "antiferro":
                    energy += nb * S
        if self.mesh == "hex":
            return energy / 6.0
        elif self.mesh == "ortho":
            return energy / 4.0
        elif self.mesh == "tri":
            return energy / 3.0

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
                nb = 0
                if self.mesh == "ortho" or self.mesh == "hex":
                    nb += (
                        crystal[(a + 1) % lattice, b]
                        + crystal[a, (b + 1) % lattice]
                        + crystal[(a - 1) % lattice, b]
                        + crystal[a, (b - 1) % lattice]
                    )
                if self.mesh == "hex" and (self.IsEven(a + 1)):
                    nb += (
                        crystal[(a - 1) % lattice, (b - 1) % lattice]
                        + crystal[(a + 1) % lattice, (b - 1) % lattice]
                    )
                elif self.mesh == "hex":
                    nb += (
                        crystal[(a + 1) % lattice, (b + 1) % lattice]
                        + crystal[(a - 1) % lattice, (b + 1) % lattice]
                    )

                if self.mesh == "tri" and (self.IsEven(a + b)):
                    nb += (
                        crystal[(a - 1) % lattice, b]
                        + crystal[(a + 1) % lattice, b]
                        + crystal[a, (b + 1) % lattice]
                    )
                elif self.mesh == "tri ":
                    nb += (
                        crystal[(a - 1) % lattice, b]
                        + crystal[(a + 1) % lattice, b]
                        + crystal[a, (b - 1) % lattice]
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

        if self.plots and (not self.animate) and self.mesh == "hex":
            f = plt.figure(figsize=(15, 15))
            self.configPlotHex(f, crystal, 0, self.lattice, 1)
        elif self.plots and (not self.animate) and self.mesh == "ortho":
            f = plt.figure(figsize=(15, 15))
            self.configPlot(f, crystal, 0, self.lattice, 1)
        elif self.plots and (not self.animate) and self.mesh == "tri":
            f = plt.figure(figsize=(15, 15))
            self.configPlotTri(f, crystal, 0, self.lattice, 1)
        elif self.plots and (self.animate) and self.mesh == "hex":
            self.savePlotHex(crystal, 0, self.lattice)
        elif self.plots and (self.animate) and self.mesh == "ortho":
            self.savePlot(crystal, 0, self.lattice)
        elif self.plots and (self.animate) and self.mesh == "tri":
            self.savePlotTri(crystal, 0, self.lattice)

        for i in range(self.iterations):
            self.mcStep(crystal, self.lattice, 1.0 / self.temperature)
            if self.plots and (not self.animate) and self.mesh == "hex":
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
            elif self.plots and (not self.animate) and self.mesh == "ortho":
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
            elif self.plots and (not self.animate) and self.mesh == "tri":
                if i == 1:
                    self.configPlotTri(f, crystal, i, self.lattice, 2)
                if i == 4:
                    self.configPlotTri(f, crystal, i, self.lattice, 3)
                if i == 32:
                    self.configPlotTri(f, crystal, i, self.lattice, 4)
                if i == 100:
                    self.configPlotTri(f, crystal, i, self.lattice, 5)
                if i == self.iterations - 1:
                    self.configPlotTri(f, crystal, i, self.lattice, 6)
            elif self.plots and (self.animate) and self.mesh == "hex":
                self.savePlotHex(crystal, i, self.lattice)
            elif self.plots and (self.animate) and self.mesh == "ortho":
                self.savePlot(crystal, i, self.lattice)
            elif self.plots and (self.animate) and self.mesh == "tri":
                self.savePlotTri(crystal, i, self.lattice)
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
        collection.set_array(crystal.ravel())
        sp.set_title("Time=%d" % i)
        sp.axis("tight")

    # configure tri plot
    def configPlotTri(self, f, crystal, i, lattice, n_):
        y = linspace(0, lattice * sqrt(3.0) / 2.0, lattice + 1)
        x = []
        for yi, yy in enumerate(y):
            if self.IsEven(yi):
                if self.IsEven(lattice):
                    xx = linspace(
                        0.5, 0.5 + (lattice / 2.0), int(round((lattice / 2) + 1))
                    )
                else:
                    xx = linspace(
                        0.5,
                        0.5 + ((lattice - 1) / 2.0),
                        int(round(((lattice - 1) / 2) + 1)),
                    )
            else:
                if self.IsEven(lattice):
                    xx = linspace(0, (lattice / 2.0), int(round((lattice / 2) + 1)))
                else:
                    xx = linspace(
                        0, ((lattice + 1) / 2.0), int(round(((lattice + 1) / 2) + 1))
                    )
            x.append(xx)
        xy = []
        for ii in range(0, len(y)):
            xx = x[ii]
            yy = y[ii]
            for j in range(0, len(xx)):
                xy.append([xx[j], yy])
        xy = asarray(xy)
        xx = xy[:, 0]
        yy = xy[:, 1]
        triangulation = tri.Triangulation(xx, yy)

        triangles = triangulation.triangles
        countervalue = []
        for triangle in triangles:
            if any(count > 1 for count in Counter(xx[triangle]).values()):
                countervalue.append(False)
            else:
                countervalue.append(True)
        triangles = [
            triangles[ii] for ii in range(0, len(triangles)) if countervalue[ii]
        ]
        triangulation.triangles = triangles

        better_triangles = []
        for triangle in triangles:
            xtri = round(xx[triangle] / 0.5).astype(int)
            ytri = round(yy[triangle] / (sqrt(3.0) / 2.0)).astype(int)
            tricoos = [[xtri[ii], ytri[ii]] for ii in range(3)]
            better_triangles.append(tricoos)

        crystallist = []
        my_tri_list = []
        for ii in range(lattice):
            for j in range(lattice):
                crystallist.append(crystal[ii, j])
                corners = self.CornersFromXY(ii, j)
                for k in range(len(better_triangles)):
                    if (
                        (corners[0] in better_triangles[k])
                        and (corners[1] in better_triangles[k])
                        and (corners[2] in better_triangles[k])
                    ):
                        my_tri_list.append(k)
                        break

        facecolors = empty(len(triangles))
        facecolors[:] = nan
        for ii in range(len(crystallist)):
            facecolors[my_tri_list[ii]] = crystallist[ii]

        sp = f.add_subplot(3, 3, n_)
        sp.tripcolor(triangulation, facecolors=facecolors, cmap=plt.cm.binary)
        sp.triplot(triangulation, lw=0.5, color="k")
        sp.set_xticks([])
        sp.set_yticks([])
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

    # configure hex plot
    def savePlotHex(self, crystal, i, lattice):
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
        collection.set_array(crystal.ravel())
        ax.set_title("Time=%d" % i)
        ax.axis("tight")
        fig.savefig(f"Animation/{i:04d}.png")

    # make a single plot
    def savePlot(self, crystal, i, lattice):
        fig, ax = plt.subplots(1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        X, Y = meshgrid(range(lattice), range(lattice))
        ax.pcolormesh(X, Y, crystal, cmap=plt.cm.binary)
        ax.set_title("Time=%d" % i)
        ax.axis("tight")
        fig.savefig(f"Animation/{i:04d}.png")

    # save tri plot
    def savePlotTri(self, f, crystal, i, lattice, n_):
        y = linspace(0, lattice * sqrt(3.0) / 2.0, lattice + 1)
        x = []
        for yi, yy in enumerate(y):
            if self.IsEven(yi):
                if self.IsEven(lattice):
                    xx = linspace(0.5, 0.5 + (lattice / 2.0), int(round((lattice / 2) + 1)))
                else:
                    xx = linspace(
                        0.5, 0.5 + ((lattice - 1) / 2.0), int(round(((lattice - 1) / 2) + 1))
                    )
            else:
                if self.IsEven(lattice):
                    xx = linspace(0, (lattice / 2.0), int(round((lattice / 2) + 1)))
                else:
                    xx = linspace(
                        0, ((lattice + 1) / 2.0), int(round(((lattice + 1) / 2) + 1))
                    )
            x.append(xx)
        xy = []
        for ii in range(0, len(y)):
            xx = x[ii]
            yy = y[ii]
            for j in range(0, len(xx)):
                xy.append([xx[j], yy])
        xy = asarray(xy)
        xx = xy[:, 0]
        yy = xy[:, 1]
        triangulation = tri.Triangulation(xx, yy)

        triangles = triangulation.triangles
        countervalue = []
        for triangle in triangles:
            if any(count > 1 for count in Counter(xx[triangle]).values()):
                countervalue.append(False)
            else:
                countervalue.append(True)
        triangles = [
            triangles[ii] for ii in range(0, len(triangles)) if countervalue[ii]
        ]
        triangulation.triangles = triangles

        better_triangles = []
        for triangle in triangles:
            xtri = round(xx[triangle] / 0.5).astype(int)
            ytri = round(yy[triangle] / (sqrt(3.0) / 2.0)).astype(int)
            tricoos = [[xtri[ii], ytri[ii]] for ii in range(3)]
            better_triangles.append(tricoos)

        crystallist = []
        my_tri_list = []
        for ii in range(lattice):
            for j in range(lattice):
                crystallist.append(crystal[ii, j])
                corners = self.CornersFromXY(ii, j)
                for k in range(len(better_triangles)):
                    if (
                        (corners[0] in better_triangles[k])
                        and (corners[1] in better_triangles[k])
                        and (corners[2] in better_triangles[k])
                    ):
                        my_tri_list.append(k)
                        break

        facecolors = empty(len(triangles))
        facecolors[:] = nan
        for ii in range(len(crystallist)):
            facecolors[my_tri_list[ii]] = crystallist[ii]

        fig, sp = plt.subplot(1, 1)
        sp.tripcolor(triangulation, facecolors=facecolors, cmap=plt.cm.binary)
        sp.triplot(triangulation, lw=0.5, color="k")
        sp.set_xticks([])
        sp.set_yticks([])
        sp.set_title("Time=%d" % i)
        sp.axis("tight")
        fig.savefig(f"Animation/{i:04d}.png")
