import math

from regex import R

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

        # table to store values
        self.nums = dict()

        if model is not None :
            self.model = model
        else :
            # may need to change this
            comma = (actionsize,)
            self_vector = self.buckets + comma
            self.model = np.zeros(self_vector)

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
        start_state = self.discretize(first_state)

        start_pair = (1,3)
        values_of_q = np.zeros(start_pair)

        # set qvals
        zero_addition = (0,)
        zero_index = start_state + zero_addition
        values_of_q[0][0] = self.model[zero_index]

        one_addition = (1,)
        one_index = start_state + one_addition
        values_of_q[0][1] = self.model[one_index]

        two_addition = (2,)
        two_index = start_state + two_addition
        values_of_q[0][2] = self.model[two_index]

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
        curr = self.discretize(state)
        curr_index = curr + (action,)
        values_of_q = self.model[curr_index]

        # lrate
        const_val = .01
        self.nums[values_of_q] = self.nums.get(values_of_q, 0) + 1

        denominator = self.nums[values_of_q] + const_val
        min_lr = const_val / denominator
        self.lr = min(self.lr, min_lr)

        # next
        n_state = self.discretize(next_state)
        ###TODO figues out goal_position
        if done == True :
            rew = 1.0
            result = rew

        else :
            # check states
            first = self.model[n_state + (0,)]
            second = self.model[n_state + (1,)]
            third = self.model[n_state + (2,)]
            max_val = max(first, second, third)

            result = (self.gamma * max_val) + rew
        
        diff_state = curr + (action,)
        use_lr = (result - values_of_q) * self.lr
        self.model[diff_state] = values_of_q + use_lr

        inner = (values_of_q - result)
        return inner*inner

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    env.reset(seed=42) # seed the environment
    np.random.seed(42) # seed numpy
    import random
    random.seed(42)

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(1, 1, 1, 1), actionsize=actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'tabular.npy')
