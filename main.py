import matplotlib.pyplot as plt

# Load dataset and initialize environment
data_path = '/content/btc_ohlcv_data.csv'  # Update with your file path
env = TradingEnvironment(data_path)

# Initialize DQN Agent
state_size = 1
action_size = 3
agent = DQNAgent(state_size=state_size, action_size=action_size)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.model.to(device)
agent.target_model.to(device)

# Training parameters
episodes = 1000  # Increase episodes for better training
batch_size = 128  # Larger batch size for GPU
update_target_every = 10  # Update target model every 10 episodes
rewards = []

# Training Loop
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Select action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = agent.act(state_tensor.cpu().numpy())  # Move state to CPU for compatibility

        # Take action in environment
        next_state, reward, done, _ = env.step(action)

        # Store transition
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # Train the agent
        agent.replay(batch_size, device)
    
    # Update target model periodically
    if episode % update_target_every == 0:
        agent.target_model.load_state_dict(agent.model.state_dict())

    rewards.append(total_reward)
    print(f"Episode {episode+1}/{episodes} - Reward: {total_reward}")

# Plot Rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Trading Bot Training Performance")
plt.show()
