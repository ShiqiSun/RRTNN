# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
from RRT.src.rrt.rrt import RRT
# from RRT.src.search_space.search_space import SearchSpace
# from RRT.src.utilities.plotting import Plot
import tensorflow as tf
# import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import region as rg
from modeltest import *


filepath = './output/Neural Network/model_train_with_100_random_initial_point_another_map_2_8_low_step.h5'
# Define size of environment
x_up_bound = 60 #120
x_low_bound = 0
y_up_bound = 60 #120
y_low_bound = 0

# create environment and obstacles
x_init_number = 100


x_goal = (30, 60)  # goal location(100, 100)
X_dimensions = np.array([(x_low_bound, x_up_bound), (y_low_bound, y_up_bound)])  # dimension of serach space
Obstacles = np.array([(0, 0, 20, 20), (0, 40, 20, 60),
                      (40, 0, 60, 20), (40, 40, 60, 60)])  # obstacles
# Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80),
#                       (60, 20, 80, 40), (60, 60, 80, 80)])  # obstacles

x_init = rg.get_initial_points(x_up_bound, x_low_bound, y_up_bound, y_low_bound, Obstacles, x_init_number)


Q = np.array([(2, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal

X = SearchSpace(X_dimensions, Obstacles)

max_iteration = 10000 #100000
data_size = 1000000 #1000000

data_size_point = data_size/x_init_number
x = np.zeros((data_size, 2))
y = np.zeros((data_size, 2))
data_position = 0

for index, init in enumerate(x_init):
    data_point = 0
    for i in range(max_iteration):
        # create rrt_search
        rrt = RRT(X, Q, init, x_goal, max_samples, r, prc)
        path = rrt.rrt_search()
        for k in range(len(path)-1):
            x[data_position] = path[k]
            y[data_position] = path[k+1]
            data_position += 1
            data_point += 1
            if data_point >= data_size_point:
                break
            if data_position >= data_size:
                break
        if data_position >= data_size:
            break
        if data_point >= data_size_point:
            break
    print("We get", data_point, "data in ", index+1, "initial point")
    if data_position >= data_size:
        break

print("Data Collected Completed， we get ", len(x), "data")

# random data set
state = np.random.get_state()
np.random.shuffle(x)

np.random.set_state(state)
np.random.shuffle(y)

# Neural Network
model = Sequential()

# model.add(Dense(16, input_dim=2, activation=tf.nn.relu)) #16
# model.add(Dense(64, activation=tf.nn.relu)) #64
# model.add(Dense(16, activation=tf.nn.relu)) #16
# model.add(Dense(2))


model.add(Dense(8, input_dim=2, activation=tf.nn.relu)) #16
model.add(Dense(8, activation=tf.nn.relu)) #16
model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y, epochs=100, batch_size=64) #100 64

model.save(filepath)
print("Train finished")
del model

# testmodel_fromzero(filepath)





