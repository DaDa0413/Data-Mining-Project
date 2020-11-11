import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 50 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)
# plt.show() # using agg, not support gui