import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gymnasium.wrappers import ResizeObservation, FrameStackObservation, TransformObservation
from torch.utils.tensorboard import SummaryWriter
import datetime


DEVICE = torch.device("cuda") # changed to cuda bc running on the nautilus instance

# wrapper class to end the episode if there is poor driving 
class AutoResetWrapper(gym.Wrapper):
    def __init__(self, env, max_negative_steps=100):
        super().__init__(env)
        self.max_negative_steps = max_negative_steps
        self.negative_step_counter = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if reward < 0:
            self.negative_step_counter += 1
        else:
            self.negative_step_counter = 0 

        if self.negative_step_counter >= self.max_negative_steps:
            truncated = True
        
        return obs, reward, terminated, truncated, info

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info

# EDIT: incorporating color as well 
def make_env():
    env = gym.make("CarRacing-v3", continuous=True)
    env = SkipFrame(env, skip=4)
    env = AutoResetWrapper(env, max_negative_steps=50)

    crop_space = gym.spaces.Box(low=0, high=255, shape=(84, 96, 3), dtype=np.uint8)
    env = TransformObservation(env, lambda obs: obs[:84, :, :], crop_space)

    env = ResizeObservation(env, (84, 84))

    norm_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84, 84, 3), dtype=np.float32)
    env = TransformObservation(env, lambda obs: obs.astype(np.float32) / 255.0, norm_space)

    permute_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 84, 84), dtype=np.float32)
    env = TransformObservation(env, lambda obs: np.transpose(obs, (2, 0, 1)), permute_space)

    env = FrameStackObservation(env, stack_size=4)
    return env

# EDIT: added more hidden layers
class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)

        self.critic = nn.Linear(512, 1)

        self.actor_mean = nn.Linear(512, action_dim)

        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -2.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = torch.distributions.Normal(action_mean, action_std)
        return probs, self.critic(x)

def compute_gae(next_value, rewards, masks, values, gamma, lam):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# EDIT: changed some of the hyperparameters
NUM_ENVS = 16 # was 4
LEARNING_RATE = 3e-4 # was 1e-4
MAX_STEPS = 2000000 # was 100000
N_STEPS = 256 # was 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_BETA = 0.001
CRITIC_LOSS_COEF = 0.5
PPO_EPOCHS = 4
PPO_CLIP = 0.2

if __name__ == "__main__":
    run_name = f"CarRacing_RGB_PPO_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"Logging to: runs/{run_name}")

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])

    action_dim = envs.single_action_space.shape[0]

    agent = ActorCritic(action_dim).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    state, _ = envs.reset()
    global_step = 0
    current_scores = np.zeros(NUM_ENVS)
    finished_games_buffer = []


    def preprocess_state(state_input):
        s = torch.FloatTensor(state_input).to(DEVICE)

        return s.view(s.shape[0], -1, 84, 84)

    num_updates = MAX_STEPS // (N_STEPS * NUM_ENVS)

    for update in range(1, num_updates + 1):
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_masks = []
        batch_values = []

        for _ in range(N_STEPS):
            global_step += NUM_ENVS

            state_tensor = preprocess_state(state)

            with torch.no_grad():
                dist, value = agent(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1)

            clipped_action = action.clone()
            clipped_action[:, 0] = torch.clamp(clipped_action[:, 0], -1.0, 1.0)
            clipped_action[:, 1] = torch.clamp(clipped_action[:, 1], 0.0, 1.0)
            clipped_action[:, 2] = torch.clamp(clipped_action[:, 2], 0.0, 1.0)

            next_state, reward, terminated, truncated, info = envs.step(clipped_action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            current_scores += reward
            for i in range(NUM_ENVS):
                if done[i]:
                    final_score = current_scores[i]
                    writer.add_scalar("Charts/Total_Episode_Reward", final_score, global_step)
                    finished_games_buffer.append(final_score)
                    current_scores[i] = 0

            scaled_reward = reward / 100.0

            batch_obs.append(state_tensor)
            batch_actions.append(action)
            batch_log_probs.append(log_prob)
            batch_rewards.append(torch.tensor(scaled_reward, dtype=torch.float32).to(DEVICE).unsqueeze(1))
            batch_masks.append(torch.tensor(1 - done, dtype=torch.float32).to(DEVICE).unsqueeze(1))
            batch_values.append(value)

            state = next_state

        next_state_tensor = preprocess_state(next_state)
        with torch.no_grad():
            _, next_value = agent(next_state_tensor)

        returns = compute_gae(next_value, batch_rewards, batch_masks, batch_values, GAMMA, GAE_LAMBDA)

        b_obs = torch.cat(batch_obs)
        b_actions = torch.cat(batch_actions)
        b_old_log_probs = torch.cat(batch_log_probs)
        b_returns = torch.cat(returns).detach()

        for _ in range(PPO_EPOCHS):
            dist, values = agent(b_obs)
            new_log_probs = dist.log_prob(b_actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            ratio = (new_log_probs - b_old_log_probs).exp()
            advantages = b_returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (b_returns - values).pow(2).mean()

            loss = actor_loss + (CRITIC_LOSS_COEF * critic_loss) - (ENTROPY_BETA * entropy)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

        if update % 5 == 0:
            if len(finished_games_buffer) > 0:
                avg_score = np.mean(finished_games_buffer[-100:])
            else:
                avg_score = 0.0
            print(f"Step {global_step} | Loss: {loss.item():.4f} | Avg Total Score: {avg_score:.2f}")
            writer.add_scalar("Charts/Mean_Episode_Reward", avg_score, global_step)
            writer.add_scalar("Loss/Total", loss.item(), global_step)

    torch.save(agent.state_dict(), "ppo_carracing_rgb_weights.pth")
    envs.close()
    writer.close()
