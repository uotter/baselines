import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sns.set(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
data = pd.DataFrame(np.random.random(size=[10,10]))
# Plot the responses for different events and regions
# cmap = sns.cubehelix_palette(n_colors=1, start=1, rot=3, gamma=0.8, as_cmap=True)
cmap = sns.color_palette("Blues",n_colors=100)
sns.heatmap(data, xticklabels=1, yticklabels=1, cmap=cmap, linewidths=0.05)
plt.show()