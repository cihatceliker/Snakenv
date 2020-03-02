import dqn
from dqn import Agent, Brain
from environment import Environment
from snake import POSSIBLE_DIRECTIONS_TO_GO, DIRECTIONS_TO_LOOK
import numpy as np
import sys
import pickle
import torch
import threading

env = Environment(row=20, col=20, num_snakes=5, throw_food_every=20)

brain_sizes = [49, 64, 64, 3]

def train():
    #agent = Agent(Q=Brain(*brain_sizes),target_Q=Brain(*brain_sizes),num_actions=num_actions)
    pickle_in = open("weights/221.8.pickle","rb"); agent = pickle.load(pickle_in)    

    num_iter = 5000
    best = 100
    scores = []
    steps = 0
    for episode in range(1, num_iter):
        obs = env.reset()
        score = 0
        while env.snakes or env.eggs:
            action_list = []
            for state, _, _, _ in obs:
                action = agent.select_action(state)
                action_list.append(action)

            obs_ = env.step(action_list)
            for i in range(min(len(obs), len(obs_))):
                agent.store_experience(obs[i][0], action_list[i], obs_[i][1], obs_[i][0], 1-obs_[i][2])
                score += obs_[i][1]
            obs = obs_
            #LEARN
            agent.learn()

        agent.eps_start = max(agent.eps_end, agent.eps_decay * agent.eps_start)

        scores.append(score)
        if episode % 10 == 0:
            avg_score = np.mean(scores[max(0, episode-10):(episode+1)])
            print('episode: ', episode,'score: %.6f' % score, ' average score %.3f' % avg_score)
            if avg_score >= best:
                best = avg_score
                pickle_out = open(str(best)+".pickle","wb")
                pickle.dump(agent, pickle_out)
                pickle_out.close()
                print("weights are safe for ", best)
        else: print('episode: ', episode,'score: %.6f' % score)

env.render()
#train()
