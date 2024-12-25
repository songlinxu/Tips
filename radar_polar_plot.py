import matplotlib.pyplot as plt
import numpy as np

# Data for the segments
angles = np.linspace(0, 2 * np.pi, 9)  # 8 segments
radii = np.random.rand(8) * 10
colors = ['orange', 'yellow', 'green', 'blue', 'orange', 'yellow', 'green', 'blue']

# Create the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

bars = ax.bar(angles[:-1], radii, width=0.5, color=colors, edgecolor='black')

# Adjust aesthetics
ax.set_yticklabels([])  # Remove radial ticks
ax.set_xticklabels([])  # Remove angular ticks
ax.spines['polar'].set_visible(True)  # Outer circle

plt.show()
