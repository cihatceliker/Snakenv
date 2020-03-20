from dqn import Agent, Brain, DuelingDQNBrain
from environment import Environment
import numpy as np
import pickle

env = Environment(row=30, col=30, num_snakes=6, throw_food_every=20)
num_states = env.observation_space
num_actions = env.action_space

#agent = Agent(local_Q=Brain(num_states, num_actions), target_Q=Brain(num_states, num_actions), num_actions=num_actions)
pickle_in = open("weights/brain.snk","rb"); agent = pickle.load(pickle_in)


start = 1 if len(agent.episodes) == 0 else agent.episodes[-1] + 1
num_iter = 5000
for episode in range(start, num_iter):
    # observations are (s, a, r, s', done) tuples per snake
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
        agent.learn()

    agent.eps_start = max(agent.eps_end, agent.eps_decay * agent.eps_start)

    agent.episodes.append(episode)
    agent.scores.append(score)
    if episode % 10 == 0:
        avg_score = np.mean(agent.scores[max(0, episode-10):(episode+1)])
        print("episode: ", episode,"score: %.6f" % score, " average score %.3f" % avg_score)
        pickle_out = open(""+str(episode)+".snk","wb")
        pickle.dump(agent, pickle_out)
        pickle_out.close()
        print("weights are safe for ", episode)
    else: print("episode: ", episode,"score: %.6f" % score)
