import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
import numpy as np
from DQN_model import Q_net

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

class Agent:
    def __init__(self):
        state_dim = (1, 84, 84)
        action_space = 12

        self.Q_eval = Q_net(state_dim, action_space)
        self.load_net()

        for p in self.Q_eval.parameters():
            p.requires_grad = False

        self.transforms = transforms.Compose(
                                [
                                    transforms.Grayscale(),
                                    transforms.Resize(state_dim[1:], antialias=True),
                                    transforms.Normalize(0, 255)
                                ]
                            )
        

    def act(self, observation):
        # transform state (240, 256, 3) to (1, 1, 84, 84)
        state_tensor = torch.tensor(np.transpose(observation, (2,0,1)).copy(), dtype = torch.float)
        state_trans = self.transforms(state_tensor).unsqueeze(0)
        Q_output = self.Q_eval(state_trans)
        
        action = torch.argmax(Q_output, dim=1)
    
        return action

    def load_net(self):
        path = '111034521_hw2_data'
        self.Q_eval.load_state_dict(torch.load(path))

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env.reset()

    agent = Agent()

    test_count = 100
    rewards_list = []
    for _ in tqdm(range(test_count)):
        acc_rewards = 0
        state = env.reset()
        die_num = 0 

        while True:
            
            action = agent.act(state)
            # print(action)
            state_next, reward, terminal, info = env.step(int(action))
            
            state = state_next
            acc_rewards += reward

            if terminal:
                break

        rewards_list.append(acc_rewards)
        print(acc_rewards)

    print(f'Average reward: {np.average(rewards_list)}')
        
    