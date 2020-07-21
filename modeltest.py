from keras.models import load_model
import numpy as np
from RRT.src.utilities.plotting import Plot
from RRT.src.search_space.search_space import SearchSpace
from region import *




def testmodel_fromzero(path, xinit, yinit):
    x_up_bound = 60  # 120
    x_low_bound = 0
    y_up_bound = 60  # 120
    y_low_bound = 0
    model = load_model(path)
    x_init = (xinit, yinit)
    x_goal = (30, 60)  # goal location(100, 100)
    X_dimensions = np.array([(x_low_bound, x_up_bound), (y_low_bound, y_up_bound)])  # dimension of serach space
    Obstacles = np.array([(0, 0, 20, 20), (0, 40, 20, 60),
                          (40, 0, 60, 20), (40, 40, 60, 60)])  # obstacles
    X = SearchSpace(X_dimensions, Obstacles)

    x_position = np.zeros((1, 2))
    x_position[0][0] = xinit
    x_position[0][1] = yinit
    path = list()
    x_temp = (x_position[0][0], x_position[0][1])
    path.append(x_temp)
    for i in range(100):
        # print(x_position)
        y_pred = model.predict(x_position)
        print(y_pred)
        # breakpoint()
        x_position = y_pred
        position = [x_position[0][0], x_position[0][1]]
        x_temp = (x_position[0][0], x_position[0][1])
        path.append(x_temp)
        if x_position[0][0] > x_up_bound or x_position[0][1] > y_up_bound or x_position[0][0] < x_low_bound or x_position[0][1] < y_low_bound:
            break
    print("Final position is", x_position)

    plot = Plot("rrt_2d")
    plot.plot_path(X, path)
    plot.plot_obstacles(X, Obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open=True)

    del model


def testmodel_random(path):
    model = load_model(path)

    # Define size of environment
    x_up_bound = 120
    x_low_bound = 0
    y_up_bound = 120
    y_low_bound = 0


    x_goal = (100, 100)

    X_dimensions = np.array([(0, 120), (0, 120)])  # dimension of serach space
    Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80),
                          (60, 20, 80, 40), (60, 60, 80, 80)])  # obstacles
    X = SearchSpace(X_dimensions, Obstacles)

    x_init = (random.randrange(x_low_bound, x_up_bound), random.randrange(y_low_bound, y_up_bound))
    while is_Obstacle(x_init, Obstacles):
        x_init = (random.randrange(x_low_bound, x_up_bound), random.randrange(y_low_bound, y_up_bound))

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
        position = [x_position[0][0], x_position[0][1]]
        x_temp = (x_position[0][0], x_position[0][1])
        path.append(x_temp)
        if testregion(position, x_goal, 3):
            break
        if x_position[0][0] > 120 or x_position[0][1] > 120 or x_position[0][0] < 0 or x_position[0][1] < 0:
            break
    print("Final position is", x_position)

    plot = Plot("rrt_2d")
    plot.plot_path(X, path)
    plot.plot_obstacles(X, Obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open=True)

    del model

def testmodel_random_nolimit(path):
    model = load_model(path)

    # Define size of environment
    x_up_bound = 60 # 120
    x_low_bound = 0
    y_up_bound = 60 #120
    y_low_bound = 0


    # x_goal = (100, 100)
    #
    # X_dimensions = np.array([(0, 120), (0, 120)])  # dimension of serach space
    # Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80),
    #                       (60, 20, 80, 40), (60, 60, 80, 80)])  # obstacles
    x_goal = (30, 60)  # goal location(100, 100)
    X_dimensions = np.array([(x_low_bound, x_up_bound), (y_low_bound, y_up_bound)])  # dimension of serach space
    Obstacles = np.array([(0, 0, 20, 20), (0, 40, 20, 60),
                          (40, 0, 60, 20), (40, 40, 60, 60)])  # obstacles
    X = SearchSpace(X_dimensions, Obstacles)

    x_init = (random.randrange(x_low_bound, x_up_bound), random.randrange(y_low_bound, y_up_bound))
    while is_Obstacle(x_init, Obstacles):
        x_init = (random.randrange(x_low_bound, x_up_bound), random.randrange(y_low_bound, y_up_bound))

    x_position = np.zeros((1, 2))
    x_position[0][0] = x_init[0]
    x_position[0][1] = x_init[1]
    path = list()
    x_temp = (x_position[0][0], x_position[0][1])
    path.append(x_temp)
    for i in range(100):
        y_pred = model.predict(x_position)
        print(y_pred)
        x_position = y_pred
        position = [x_position[0][0], x_position[0][1]]
        x_temp = (x_position[0][0], x_position[0][1])
        path.append(x_temp)
        if x_position[0][0] > x_up_bound or x_position[0][1] > y_up_bound or x_position[0][0] < x_low_bound or x_position[0][1] < y_low_bound:
            break
    print("Final position is", x_position)

    plot = Plot("rrt_2d")
    plot.plot_path(X, path)
    plot.plot_obstacles(X, Obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open=True)

    del model
