from src.stabilzer import OptimalCameraPathStabilizer

__author__ = 'Marek'

from pulp import *
import numpy as np
from matplotlib import pyplot as plt

frame_height = 40
crop_height = frame_height*0.6
corners = {"up": crop_height/2, "low": -crop_height/2}
weights = [1, 10, 100]
original_trans = (np.random.rand(100)*6 - 3)
original_path = np.cumsum(original_trans)

stabilizer = OptimalCameraPathStabilizer(original_trans, (frame_height-crop_height)/2)

new_trans, status = stabilizer.stabilize()
stabilizer.cleanup()
new_path = [original_path[t] + b for t, b in enumerate(new_trans)]

# The status of the solution is printed to the screen
print "Status:", status

plt.plot(original_path, ".", color="red")
plt.plot(np.array(original_path)+frame_height/2, '-', dashes=[5, 1], color="red")
plt.plot(np.array(original_path)-frame_height/2, '-', dashes=[5, 1], color="red")

plt.plot(new_path, ".", color="blue")
plt.plot(np.array(new_path)+crop_height/2, '-', dashes=[5, 1], color="blue")
plt.plot(np.array(new_path)-crop_height/2, '-', dashes=[5, 1], color="blue")
residual = np.array(original_trans) + np.array(new_trans) - np.array([0] + new_trans[1::])

plt.show()



