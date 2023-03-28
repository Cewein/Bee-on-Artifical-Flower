# %%
#%load_ext autoreload
#%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import os

# %%
paths=[]

for file in os.listdir("data"):
    if file.endswith(".txt"):
        paths.append(os.path.join("data", file))

for path in paths:
    data = np.loadtxt(path,usecols=(4,))
    plt.plot(data[::1], label=path)

plt.legend()
plt.show()

# %%
