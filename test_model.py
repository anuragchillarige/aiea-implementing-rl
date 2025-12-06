import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
from a2c_car_train import ActorCritic # only this import needed bc ppo and a2c have the same ActorCritic class
import sys


DEVICE = torch.device("mps") # running on apple silicon

# make the environment for testing the model
def make_test_env():
    env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    
    env = GrayscaleObservation(env, keep_dim=False)
    
    env = ResizeObservation(env, (84, 84))
    
    new_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84), dtype=np.float32)
    env = TransformObservation(env, lambda obs: obs / 255.0, new_space)
    
    env = FrameStackObservation(env, stack_size=4)
    return env

if __name__ == "__main__":
    model = sys.argv[1]

    if (model != "a2c" and model != "ppo"):
        print("only supports a2c and ppo models")
        exit()
    
    print(f"Testing {model}")
    env = make_test_env()
    action_dim = env.action_space.n
    
    agent = ActorCritic(action_dim).to(DEVICE)
    
    try: # load the model weights
        agent.load_state_dict(torch.load(f"{model}_carracing_weights.pth", map_location=DEVICE))
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print(f"Error: f'{model}_carracing_weights.pth' not found.")
        exit()
        
    agent.eval() # set the model to evaluation mode

    episodes = 5
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            
            if state_tensor.shape[-1] == 4:
                state_tensor = state_tensor.permute(2, 0, 1) # reshape the tensor to the correct shape 
            
            state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                probs, _ = agent(state_tensor)
            
            action = torch.argmax(probs, dim=1).item() # getting the best action 
            
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            
        print(f"Episode {ep+1} Score: {score:.2f}")
    
    env.close()