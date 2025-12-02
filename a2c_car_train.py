import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
from torch.utils.tensorboard import SummaryWriter
import datetime


DEVICE = torch.device("mps") # running on apple silicon
   

def make_env():
    env = gym.make("CarRacing-v3", continuous=False)
    
    env = GrayscaleObservation(env, keep_dim=True) # make black and white, color doesnt matter
    env = ResizeObservation(env, (84, 84)) # resize so computation is faster
    
    new_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84), dtype=np.float32)
    env = TransformObservation(env, lambda obs: obs / 255.0, new_space) # make values between 0 and 1, makes computation faster
    
    env = FrameStackObservation(env, stack_size=4) 
    return env

class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
    
    # returns actor and critic values 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

# 
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

# CONSTATNS
NUM_ENVS = 4
LEARNING_RATE = 1e-4
MAX_STEPS = 100000
N_STEPS = 10
GAMMA = 0.99
ENTROPY_BETA = 0.01
CRITIC_LOSS_COEF = 0.5 

if __name__ == "__main__":
    # setting up logs 
    run_name = f"CarRacing_Manual_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"Logging to: runs/{run_name}")

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)]) # create 4 environments
    action_dim = envs.single_action_space.n
    agent = ActorCritic(action_dim).to(DEVICE) 
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    # reset the environment
    state, _ = envs.reset()
    global_step = 0
    
 
    current_scores = np.zeros(NUM_ENVS)

    finished_games_buffer = []

    # training loop 
    for update in range(1, (MAX_STEPS // N_STEPS) + 1):
        log_probs, values, rewards, masks, entropies = [], [], [], [], []
        # loop through N_STEPS steps
        for _ in range(N_STEPS): 
            global_step += NUM_ENVS
            
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            if state_tensor.shape[-1] == 4:
                state_tensor = state_tensor.permute(0, 3, 1, 2)
            
            probs, value = agent(state_tensor) # get agent and critic values
            dist = torch.distributions.Categorical(probs) # get distribution of actions
            action = dist.sample() 
            
            
            next_state, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated) # check if the episode is done
            
            
            current_scores += reward # add reward to current scores
            
            # loop through environments
            for i in range(NUM_ENVS):
                if done[i]:
                   
                    final_score = current_scores[i] 
                    writer.add_scalar("Charts/Total_Episode_Reward", final_score, global_step) # log the score
                    finished_games_buffer.append(final_score) # add the score to the buffer
                    
                   
                    current_scores[i] = 0

            log_probs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())
            rewards.append(torch.tensor(reward, dtype=torch.float32).to(DEVICE).unsqueeze(1))
            masks.append(torch.tensor(1 - done, dtype=torch.float32).to(DEVICE).unsqueeze(1))
            
            state = next_state

        
        next_state_tensor = torch.FloatTensor(next_state).to(DEVICE)
        if next_state_tensor.shape[-1] == 4:
            next_state_tensor = next_state_tensor.permute(0, 3, 1, 2)
            
        _, next_value = agent(next_state_tensor)
        returns = compute_returns(next_value, rewards, masks, GAMMA)
        
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)
        entropies = torch.cat(entropies)
        advantage = returns - values
        
        loss = -(log_probs * advantage.detach()).mean() + \
               (CRITIC_LOSS_COEF * advantage.pow(2).mean()) - \
               (ENTROPY_BETA * entropies.mean()) # loss formula 
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

        
        if update % 10 == 0:
            if len(finished_games_buffer) > 0:
                avg_score = np.mean(finished_games_buffer[-10:]) # get the average score from the buffer
            else:
                avg_score = 0.0 # edge case where games are not finished (usually only happens at the start )
            
            print(f"Step {global_step} | Loss: {loss.item():.4f} | Avg Total Score: {avg_score:.2f}")
            
            
            writer.add_scalar("Loss/Total", loss.item(), global_step)

    torch.save(agent.state_dict(), "a2c_carracing_weights.pth") # save the model to use in the test script
    envs.close()
    writer.close()