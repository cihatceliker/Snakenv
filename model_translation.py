import numpy as np
import pickle

# this code makes cpu version of trained pytorch model

class NumpyAgent():
    
    def select_action(self, x):
        x = np.dot(self.w1,x)+self.b1
        x = np.maximum(x,0)
        x = np.dot(self.w2,x)+self.b2
        x = np.maximum(0, x)
        x = np.dot(self.w3,x)+self.b3
        return np.argmax(x)

    def forward_visual_mode(self, x):
        x = np.array(x)
        ins = [x.clip(0,1)]
        x = np.dot(self.w1,x)
        ins.append((x-x.min())/(x.max()-x.min()))
        x = np.dot(self.w2,x)
        ins.append((x-x.min())/(x.max()-x.min()))
        x = np.dot(self.w3,x)
        ins.append((x-x.min())/(x.max()-x.min()))
        return ins


if __name__ == "__main__":
    new_agent = NumpyAgent()
    agent = pickle.load(open("2130.snk","rb"))
    new_agent.w1 = agent.local_Q.state_dict()["fc1.weight"].cpu().numpy()
    new_agent.b1 = agent.local_Q.state_dict()["fc1.bias"].cpu().numpy()
    new_agent.w2 = agent.local_Q.state_dict()["fc2.weight"].cpu().numpy()
    new_agent.b2 = agent.local_Q.state_dict()["fc2.bias"].cpu().numpy()
    new_agent.w3 = agent.local_Q.state_dict()["out.weight"].cpu().numpy()
    new_agent.b3 = agent.local_Q.state_dict()["out.bias"].cpu().numpy()

    pickle_out = open("numpy_brain.snk","wb")
    pickle.dump(new_agent, pickle_out)
    pickle_out.close()
