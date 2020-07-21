import random


def testregion(position, goal, size):
    if -1*size < position[0] - goal[0] < size and -1*size < position[1] - goal[1] < size:
        return True
    return False


def is_Obstacle(x, obstacles):
    for obstacle in obstacles:
        if obstacle[0] <= x[0] <= obstacle[2] and obstacle[1] <= x[1] <= obstacle[3]:
            return True
    return False


def get_initial_points(x_up_bound, x_low_bound, y_up_bound, y_low_bound, obstacles, x_init_number):
    x_init = [(random.randrange(x_low_bound, x_up_bound),
               random.randrange(y_low_bound, y_up_bound)) for i in range(x_init_number)]
    for index in range(len(x_init)):
        while is_Obstacle(x_init[index], obstacles):
            x_init[index] = (random.randrange(x_low_bound, x_up_bound), random.randrange(y_low_bound, y_up_bound))
    return x_init

