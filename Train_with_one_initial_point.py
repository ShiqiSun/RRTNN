# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
from RRT.src.rrt.rrt import RRT
from RRT.src.search_space.search_space import SearchSpace
from RRT.src.utilities.plotting import Plot
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# create environment and obstacles
x_init = (0, 0)  # starting location
x_goal = (100, 100)  # goal location
X_dimensions = np.array([(0, 120), (0, 120)])  # dimension of serach space
Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80),
                      (60, 20, 80, 40), (60, 60, 80, 80)])  # obstacles


Q = np.array([(8, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal
max_iteration = 100000 #100000
data_size = 100000 #100000
x = np.zeros((data_size, 2))
y = np.zeros((data_size, 2))
data_position = 0

X = SearchSpace(X_dimensions, Obstacles)

for i in range(max_iteration):
    # create rrt_search
    rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
    path = rrt.rrt_search()
    for k in range(len(path)-1):
        x[data_position] = path[k]
        y[data_position] = path[k+1]
        data_position += 1
        if data_position >= data_size:
            break
    if data_position >= data_size:
        break

print("Data Collected")

# Neural Network
model = Sequential()

model.add(Dense(16, input_dim=2, activation=tf.nn.relu)) #16
model.add(Dense(64, activation=tf.nn.relu)) #64
model.add(Dense(16, activation=tf.nn.relu)) #16
model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y, epochs=100, batch_size=64) #100 64

print("Train finished")

x_position = np.zeros((1, 2))
x_position[0][0] = x_init[0]
x_position[0][1] = x_init[1]
path = list()
x_temp = (x_position[0][0], x_position[0][1])
path.append(x_temp)
while True:
    y_pred = model.predict(x_position)
    print(y_pred)
    x_position = y_pred
    if x_position[0][0] == x_goal[0] and x_position[0][1] == x_goal[1]:
        break
    if x_position[0][0] >= 110 or x_position[0][1] >= 110:
        break
    x_temp = (x_position[0][0], x_position[0][1])
    path.append(x_temp)

print("Final position is", x_position)

plot = Plot("rrt_2d")
plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)

model.save('./output/Neural Network/model_train_with_one_initial_point.h5')
del model





