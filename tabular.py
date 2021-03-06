from distutils.command.build_scripts import first_line_re
import math

# from regex import R

import gym
import numpy as np
import torch

import utils
from policies import QPolicy

# Modified by Mohit Goyal (mohit@illinois.edu) on 04/20/2022

class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update 
        @param gamma: discount factor
        @param model (optional): Load a saved table of Q-values for each state-action
            model = np.zeros(self.buckets + (actionsize,))
            
        """
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env
        self.buckets = buckets
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma
        self.model = np.zeros(self.buckets + (actionsize,)) if model is None else model

        # table to store values
        self.nums = dict()

        # if model is not None :
        #     self.model = model
        # else :
        #     # may need to change this
        #     comma = (actionsize,)
        #     self_vector = self.buckets + comma
        #     self.model = np.zeros(self_vector)

    def discretize(self, obs):
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation  
        """
        upper_bounds = [self.env.observation_space.high[0], 5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        first_state = states[0]
        discrete_state = self.discretize(first_state)
        arr_model= [self.model[discrete_state]]

        return np.array(arr_model)

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        curr_state_d = self.discretize(state)
        next_state_d = self.discretize(next_state)

        C = .1
        denominator = C+self.model[curr_state_d][action]
        self.lr = min(self.lr, C/denominator)

        if done == True :
            target = reward
        else :
            mult = np.amax(self.model[next_state_d])
            target = reward + (self.gamma * mult)
        
        first_val = self.model[curr_state_d][action]
        second_val = self.lr * (target - self.model[curr_state_d][action])
        self.model[curr_state_d][action] = first_val + second_val

        subtract = self.model[curr_state_d][action]
        return np.square(np.subtract(target, subtract))

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    # env.reset(seed=42) # seed the environment
    # np.random.seed(42) # seed numpy
    # import random
    # random.seed(42)

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(15, 6, 15, 7), actionsize=actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'tabular.npy')
# from distutils.command.build_scripts import first_line_re
# import math

# # from regex import R

# import gym
# import numpy as np
# import torch

# import utils
# from policies import QPolicy

# # Modified by Mohit Goyal (mohit@illinois.edu) on 04/20/2022

# class TabQPolicy(QPolicy):
#     def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
#         """
#         Inititalize the tabular q policy

#         @param env: the gym environment
#         @param buckets: specifies the discretization of the continuous state space for each dimension
#         @param actionsize: dimension of the descrete action space.
#         @param lr: learning rate for the model update 
#         @param gamma: discount factor
#         @param model (optional): Load a saved table of Q-values for each state-action
#             model = np.zeros(self.buckets + (actionsize,))
            
#         """
#         super().__init__(len(buckets), actionsize, lr, gamma)
#         self.env = env
#         self.buckets = buckets
#         self.actionsize = actionsize
#         self.lr = lr
#         self.gamma = gamma
#         self.model = np.zeros(self.buckets + (actionsize,)) if model is None else model

#         # table to store values
#         self.nums = dict()

#         # if model is not None :
#         #     self.model = model
#         # else :
#         #     # may need to change this
#         #     comma = (actionsize,)
#         #     self_vector = self.buckets + comma
#         #     self.model = np.zeros(self_vector)

#     def discretize(self, obs):
#         """
#         Discretizes the continuous input observation

#         @param obs: continuous observation
#         @return: discretized observation  
#         """
#         upper_bounds = [self.env.observation_space.high[0], 5, self.env.observation_space.high[2], math.radians(50)]
#         lower_bounds = [self.env.observation_space.low[0], -5, self.env.observation_space.low[2], -math.radians(50)]
#         ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
#         new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
#         new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
#         return tuple(new_obs)

#     def qvals(self, states):
#         """
#         Returns the q values for the states.

#         @param state: the state
        
#         @return qvals: the q values for the state for each action. 
#         """
#         first_state = states[0]
#         discrete_state = self.discretize(first_state)
#         arr_model= [self.model[discrete_state]]

#         return np.array(arr_model)

#     def td_step(self, state, action, reward, next_state, done):
#         """
#         One step TD update to the model

#         @param state: the current state
#         @param action: the action
#         @param reward: the reward of taking the action at the current state
#         @param next_state: the next state after taking the action at the
#             current state
#         @param done: true if episode has terminated, false otherwise
#         @return loss: total loss the at this time step
#         """
#         curr_state_d = self.discretize(state)
#         next_state_d = self.discretize(next_state)

#         C = .1
#         denominator = C+self.model[curr_state_d][action]
#         self.lr = min(self.lr, C/denominator)

#         if done == True :
#             target = reward
#         else :
#             mult = np.amax(self.model[next_state_d])
#             target = reward + (self.gamma * mult)
        
#         first_val = self.model[curr_state_d][action]
#         second_val = self.lr * (target - self.model[curr_state_d][action])
#         self.model[curr_state_d][action] = first_val + second_val

#         subtract = self.model[curr_state_d][action]
#         return np.square(np.subtract(target, subtract))

#     def save(self, outpath):
#         """
#         saves the model at the specified outpath
#         """
#         torch.save(self.model, outpath)


# if __name__ == '__main__':
#     args = utils.hyperparameters()

#     env = gym.make('CartPole-v1')
#     # env.reset(seed=42) # seed the environment
#     # np.random.seed(42) # seed numpy
#     # import random
#     # random.seed(42)

#     statesize = env.observation_space.shape[0]
#     actionsize = env.action_space.n
#     # policy = TabQPolicy(env, buckets=(5, 8, 5, 8), actionsize=actionsize, lr=args.lr, gamma=args.gamma)
#     policy = TabQPolicy(env, buckets=(4, 8, 4, 8), actionsize=actionsize, lr=args.lr, gamma=args.gamma) # best score

#     utils.qlearn(env, policy, args)

#     torch.save(policy.model, 'tabular.npy')
