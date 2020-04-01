import pickle
import torch
import sys
from dqn import Brain, Agent

# this surgery is done to add a new state
pickle_in = open("old_brain.snk","rb")
agent = pickle.load(pickle_in)
new_agent = Agent(Brain(57, 3), Brain(57, 3), 3) # old input size is 49

old_local_brain = agent.local_Q
old_target_brain = agent.target_Q

local_weights = old_local_brain.state_dict()["fc1.weight"]
target_weights = old_target_brain.state_dict()["fc1.weight"]

new_local_weights = new_agent.local_Q.state_dict()["fc1.weight"]
new_target_weights = new_agent.target_Q.state_dict()["fc1.weight"]

new_local_state_dict = {}
new_target_state_dict = {}

for param in old_local_brain.state_dict():
    if param == "fc1.weight": continue
    new_local_state_dict[param] = old_local_brain.state_dict()[param]
    new_target_state_dict[param] = old_target_brain.state_dict()[param]

k = 0
for i in range(57):
    if (i-2)%7 == 0: continue
    new_local_weights[:,i] = local_weights[:,k]
    new_target_weights[:,i] = target_weights[:,k]
    k += 1

new_local_state_dict["fc1.weight"] = new_local_weights
new_target_state_dict["fc1.weight"] = new_target_weights

new_agent.local_Q.load_state_dict(new_local_state_dict)
new_agent.target_Q.load_state_dict(new_target_state_dict)

new_agent.eps_start = agent.eps_start
new_agent.optimizer = torch.optim.Adam(new_agent.local_Q.parameters(), agent.alpha)
new_agent.scores = agent.scores
new_agent.episodes = agent.episodes

pickle_out = open("new_brain.snk","wb")
pickle.dump(new_agent, pickle_out)
pickle_out.close()
