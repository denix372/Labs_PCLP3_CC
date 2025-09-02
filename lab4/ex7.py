import matplotlib.pyplot as plt
import numpy as np


xpoints = np.linspace(-4, 4, 100)

f = lambda x : x**3 + 2*x**2 + 4 * x - 6

ypoints = [f(x) for x in xpoints]

# realizăm graficul
plt.plot(xpoints, ypoints, label="f(x) = x^3 + 2x^2 + 4x - 6")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()