import numpy as np
import math
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

n = 10
a = 0
b = 2*math.pi
x = np.linspace(a,b,n) + np.random.normal(0, 1, n)
y = np.sin(x)

poly = lagrange(x,y)

poly_coefs = Polynomial(poly.coef[::-1]).coef
print(poly_coefs)

step = 0.1
x_new = np.arange(min(x), max(x)+step, step)

plt.scatter(x, y, label='data')
y_new = Polynomial(poly.coef[::-1])(x_new)
plt.plot(x_new, y_new, label='Polynomial')
plt.savefig('section4_lagrange/lagrange_plot.png')

sum = 0
for x in x_new:
    sum = sum + abs(math.sin(x) - Polynomial(poly.coef[::-1])(x))
print(sum)
