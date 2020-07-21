from modeltest import *
import time
from shapely.geometry import Point
from shapely.geometry import Polygon
import random



path = './output/Neural Network/model_train_with_100_random_initial_point_another_map_2_8_low_step.h5'
# x = random.randrange(45, 55)
# y = random.randrange(20, 40)
# testmodel_fromzero(path, 10, 27) #unsafe
# print(x, y)
# while True:
#     time.sleep(2)
# testmodel_random_nolimit(path)
#
# while True:
#     time.sleep(1)
#     testmodel_random_nolimit(path)
#     import sys
#     sys.exit(0)

model = load_model(path)
x = np.zeros((1, 2))
x[0][0] = 16
x[0][1] = 8
y = model.predict(x)
print(y)