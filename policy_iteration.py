from math import gamma
import numpy as np

maze_size = 100 # a global variable

def construct_map(plot = False):
    '''
    0: no obstacle
    1: obstacle
    2: exit
    '''
    map = np.full(100,0)
    total_set = set(np.arange(0,100,1))
    obstacle_set = set(np.arange(0,10)) | set(np.arange(90,100)) | set(np.arange(10,90,10)) | set(np.arange(19,99,10)) \
                    | set([74,75,64,54,44,57,47,23,24,25,26])
    map[list(obstacle_set)] = 1
    map[18] = 2
    
    if plot:
        plot_map = map.reshape(10,10)
        for i in range(10):
            print(plot_map[9-i])
    return map


def policy_iteration(num_interation, discount_factor):
    '''
    Run policy iteration for a given number of times
    '''
    policy = np.full(100, 2)
    map = construct_map()
    for i in range(num_interation):
        T = construct_T(policy, map)
        runtime_cost = compute_reward_everyCell(policy, map)
        J_pi = policy_evaluation(T, discount_factor, runtime_cost)
        policy = policy_improvement(discount_factor, J_pi, map)

    return policy, J_pi


def policy_evaluation(T, disc_factor,runtime_cost):
    '''
    policy evaluation: for a given policy, compute the converged values for each state
    INPUT:
        T: (100,100) transition matrix
        disc_factor: (100, )
        last_policy: (100, ) each element will be action like "east", "north", "west", "south"

    OUTPUT:
        value: (100, ) values of each cell
    '''

    J_pi = np.linalg.inv((np.eye(maze_size) - disc_factor * T)) @ runtime_cost.reshape(-1,1)

    return J_pi.flatten()

def policy_improvement(discount_factot, J_pi, map):
    '''
    improvement policy by choosing the action that maximize the reward for each state 
    '''
    num_states = maze_size
    actions = [0, 1, 2, 3]
    new_policy = np.full(100,None)
    for x in range(10):
        for y in range(10):
            current_state = position_to_index(x,y)
            reward_of_actions = np.zeros(4)
            for i, action in enumerate(actions):
                P = np.zeros(4) # the order of probability is: move as expected, two wrong movement, stay put
                q = np.zeros(4)
                J = np.zeros(4)
                if map[current_state] == 0:
                    next_state_right = find_next_state((x,y),action) # the next state the actual action is executed
                    P[0] = 0.7
                    q[0] = compute_runtime_reward((x,y), action, map)
                    J[0] = J_pi[next_state_right]
                    # compute the next_state if the wrong action is executed
                    wrong_action = find_possible_action(action)
                    next_state_wrong1 = find_next_state((x,y), wrong_action[0])
                    next_state_wrong2 = find_next_state((x,y), wrong_action[1])
                    P[1] = 0.1
                    P[2] = 0.1
                    q[1] = compute_runtime_reward((x,y), wrong_action[0], map)
                    q[2] = compute_runtime_reward((x,y), wrong_action[1], map)
                    J[1] = J_pi[next_state_wrong1]
                    J[2] = J_pi[next_state_wrong2]
                    # 0.1 possibility to stay put
                    P[3] = 0.1
                    q[3] = -1
                    J[3] = J_pi[current_state]

                elif map[current_state] == 1:
                    P[3] = 1
                    q[3] = -10
                    J[3] = J_pi[current_state]

                elif map[current_state] == 2:
                    P[3] = 1
                    q[3] = 10
                    J[3] = J_pi[current_state]

                reward_of_actions[i] = (P @ (q + discount_factot*J).reshape(-1,1)).flatten()

            new_policy[current_state] = np.argmax(reward_of_actions).item()

    return new_policy


def construct_T(policy, map):
    '''
    construct a transition matrix, return a (100,100) ndarray, each T[i] is P(next x | x_i)
    INPUT: 
        policy: the current policy, (100, ) if policy[i] = 0, move north
                                                            1, south
                                                            2, east
                                                            3, west
        map: (100, ) ndarray,   0: no obstacle
                                1: obstacle
                                2: exit
    '''
    T = np.zeros((maze_size,maze_size))
    for x in range(10):
        for y in range(10):
            current_state = position_to_index(x,y) # compute state index
            if map[current_state] == 0:
                next_state_right = find_next_state((x,y), policy[current_state]) # the next state the actual action is executed
                T[current_state, next_state_right] = 0.7

                # compute the next_state if the wrong action is executed
                wrong_action = find_possible_action(policy[current_state])
                next_state_wrong1 = find_next_state((x,y), wrong_action[0])
                next_state_wrong2 = find_next_state((x,y), wrong_action[1])
                T[current_state, next_state_wrong1] = 0.1
                T[current_state, next_state_wrong2] = 0.1

                # 0.1 possibility to stay put
                T[current_state,current_state] = 0.1

            elif map[current_state] == 1:
                T[current_state,current_state] = 1

            elif map[current_state] == 2:
                T[current_state,current_state] = 1

    return T

def find_possible_action(desired_action):
    '''
    if desried action is go north, then possible action is go east or west but not the opposite direction
    '''

    if desired_action == 0 or desired_action == 1:
        possible_action = [2, 3]
    else: # desired_action == 2 or desired_action ==3:
        possible_action = [0, 1]
    return possible_action

def find_next_state(current_position, action):
    # compute the next state given the action and current state, if the next state is out of boundary
    # the agent doesn't move
    x = current_position[0]
    y = current_position[1]
    if action == 0:
        next_x = x
        next_y = y+1
        if out_of_boundary(next_x, next_y):
            next_state = position_to_index(x,y)
        else:
            next_state = position_to_index(next_x,next_y)
        
    elif action == 1:
        next_x = x
        next_y = y-1
        if out_of_boundary(next_x, next_y):
            next_state = position_to_index(x,y)
        else:
            next_state = position_to_index(next_x,next_y)

    elif action == 2:
        next_x = x+1
        next_y = y
        if out_of_boundary(next_x, next_y):
            next_state = position_to_index(x,y)
        else:
            next_state = position_to_index(next_x,next_y)
    else:
        next_x = x-1
        next_y = y
        if out_of_boundary(next_x, next_y):
            next_state = position_to_index(x,y)
        else:
            next_state = position_to_index(next_x,next_y)
    return next_state


def compute_reward_everyCell(action,map):
    '''
    compute the reward for every state given their actions
    INPUT:
        action: (100, ), if action[i] = 0, move north
                                        1, south
                                        2, east
                                        3, west
        map: (100, ),   0: no obstacle
                        1: obstacle
                        2: exit
    OUTPUT:
        reward: (100, ), the reward to execute the given action at a given state i
    '''
    reward = np.zeros(maze_size)
    for x in range(10):
        for y in range(10):
            current_state = position_to_index(x,y)
            reward[current_state] = compute_runtime_reward((x,y), action[current_state], map)
    return reward

def position_to_index(x,y):
    index = y*10+x
    return index

def compute_runtime_reward(current_position, action, map):
    '''
    Compute the runtime reward for a single state and a single action
    INPUT:
        current_position: (x,y)
        action: 0, 1, 2, 3
        map: (100, )
    OUTPUT:
        reward: int
    '''
    current_state = position_to_index(current_position[0], current_position[1])
    # if map[current_state] == 1:
    #     reward = -10
    # elif map[current_state] == 2:
    #     reward = 10

    if map[current_state] == 2:
        reward = 10
    else:
        # compute the next state given the action and current state, if the next state is out of boundary
        # the agent doesn't move
        next_state = find_next_state(current_position, action)
        # assign reward according to the next state
        if map[next_state] == 1: # there is obstacle
            reward = -10
        elif map[next_state] == 0: # no obstacle
            reward = -1
        else: # meet end state
            reward = 10

    return reward

def out_of_boundary(x,y):
    '''
    test whether a position is out od boundary
    '''        
    return (x>=10) or (x<0) or (y>=10) or (y<0)