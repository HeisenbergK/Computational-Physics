from scipy import *
from numpy import *
from scipy import integrate
from scipy import optimize
from pylab import *


def Schroed_deriv(y, r, l, En):
    "Given y=[u,u'] returns dy/dr=[u',u''] "
    (u, up) = y
    return array([up, (l * (l + 1) / r**2 - 2 / r - En) * u])


def SolveSchroedinger(En, l, R):
    Rb = R[::-1]
    du0 = -1e-5
    urb = integrate.odeint(Schroed_deriv, [0.0, du0], Rb, args=(l, En))
    ur = urb[:, 0][::-1]
    norm = integrate.simps(ur**2, x=R)
    ur *= 1. / sqrt(norm)
    return ur


l = 2
n = 5
En = -1. / (n**2)  # 2p orbital

R = logspace(-6, 2, 500)
ur = SolveSchroedinger(En, l, R)


# ylim([0,0.5])
plot(R, ur / R, 'o-')
xlim([0, 20])
show()
