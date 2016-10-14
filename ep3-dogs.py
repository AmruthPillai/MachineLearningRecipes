import numpy as np
import matplotlib.pyplot as plt

# assume we have 1000 dogs
greyhounds = 500
labradors = 500

# greyhounds are usually 28 inches
grey_height = 28 + (4 * np.random.randn(greyhounds))

# labradors are usually 24 inches
lab_height = 24+ (4 * np.random.randn(labradors))

# plot a histogram of the two heights
plt.hist([grey_height, lab_height], stacked = True, color = ['r', 'b'])
plt.show()
