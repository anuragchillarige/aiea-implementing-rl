import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation

# --- 1. Device Config (Match Training) ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# --- 2. Redefine Architecture ---
class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

# --- 3. Setup Single Test Environment (FIXED) ---
def make_test_env():
    env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    
    # 1. Grayscale: keep_dim=False (Matches Training Fix)
    env = GrayscaleObservation(env, keep_dim=False)
    
    # 2. Resize
    env = ResizeObservation(env, (84, 84))
    
    # 3. Transform: EXPLICITLY provide observation_space (The Fix)
    new_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84), dtype=np.float32)
    env = TransformObservation(env, lambda obs: obs / 255.0, new_space)
    
    # 4. Stack
    env = FrameStackObservation(env, stack_size=4)
    return env

if __name__ == "__main__":
    env = make_test_env()
    action_dim = env.action_space.n
    
    # Initialize and move to Device
    agent = ActorCritic(action_dim).to(DEVICE)
    
    # Load weights (Handle MPS/CPU mapping automatically)
    try:
        agent.load_state_dict(torch.load("a2c_carracing_weights.pth", map_location=DEVICE))
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print("Error: 'a2c_carracing_weights.pth' not found. Train the model first!")
        exit()
        
    agent.eval()

    episodes = 5
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            # 1. Prepare State: (H, W, 4) -> (4, H, W) and Add Batch Dimension
            # Gym output: (84, 84, 4) or (4, 84, 84) depending on version. 
            # We cast to Tensor and move to device first.
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            
            # Fix dimensions if needed (Channel Last -> Channel First)
            if state_tensor.shape[-1] == 4:
                state_tensor = state_tensor.permute(2, 0, 1)
            
            # Add Batch Dim: (1, 4, 84, 84)
            state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                probs, _ = agent(state_tensor)
            
            # Use argmax for testing (Best possible move)
            action = torch.argmax(probs, dim=1).item()
            
            # If you want randomness:
            # action = torch.distributions.Categorical(probs).sample().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            
        print(f"Episode {ep+1} Score: {score:.2f}")
    
    env.close()