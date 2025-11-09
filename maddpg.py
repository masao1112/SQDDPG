import torch
import torch.nn.functional as F
from agent import MADDPGAgent

class MADDPG:
    def __init__(self, critic_dims, actor_dims, n_agents, n_actions, chkpt_dir,
                 fc1=64, fc2=64, alpha=0.01, beta=0.01, gamma=0.99, tau=0.01,
                 evaluate=False):
        self.n_ = n_agents
        self.obs_dim = critic_dims
        self.actor_dims = actor_dims
        self.agents = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for agent_idx in range(n_agents):
            agent = MADDPGAgent(critic_dims, actor_dims[agent_idx], fc1, fc2, agent_idx, 
                  n_agents, n_actions, chkpt_dir=chkpt_dir, alpha=alpha, beta=beta, gamma=gamma, tau=tau, evaluate=evaluate)
            self.agents.append(agent)
            
    def choose_action(self, obs):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(obs[i], 0.1)
            actions.append(action)
        return actions
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
            
    def learn(self, memory):
        if not memory.ready():
            return
        
        obs, actions, rewards, next_obs, dones = memory.sample_buffer()
        # shapes:
        # obs: list[n_] of (B, actor_dim_i)
        # actions: list[n_] of (B, n_actions)
        # rewards: (B, n_)
        # next_obs: list[n_] of (B, actor_dim_i)
        # dones: (B, n_)
        
        # Calculate per agent actor outputs (current mu) and target actor for next_state
        all_next_actions = []
        all_mu_actions = []
        old_agents_actions = [] 
        next_critic_inputs = []
        critic_inputs = []
        for agent_idx, agent in enumerate(self.agents):
            # get the desired observation for corresponding agent
            obs_i = torch.tensor(obs[agent_idx], dtype=torch.float32).to(self.device)  # (B, actor_dim_i)
            next_obs_i = torch.tensor(next_obs[agent_idx], dtype=torch.float32).to(self.device)  # (B, actor_dim_i)
            # convert to torch for torch.cat
            action_i = torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device)  # (B, n_actions)
            
            # compute mu i.e deterministic policy
            mu_i = agent.actor(obs_i)
            next_mu_i = agent.target_actor(next_obs_i)
            all_mu_actions.append(mu_i)
            all_next_actions.append(next_mu_i)
            old_agents_actions.append(action_i)
            next_critic_inputs.append(next_obs_i)
            critic_inputs.append(obs_i)
            
        # Concatenate actions for critic (dim = B, n_*n_actions)
        mu_cat = torch.cat(all_mu_actions, dim=-1).to(self.device)
        next_actions_cat = torch.cat(all_next_actions, dim=-1).to(self.device)
        old_actions_cat = torch.cat(old_agents_actions, dim=-1).to(self.device)  # (B, n_*n_actions)
        critic_input_cat = torch.cat(critic_inputs, dim=-1).to(self.device)  # (B, total_obs_dim)
        next_critic_input_cat = torch.cat(next_critic_inputs, dim=-1).to(self.device)
        
        # MAIN: critic and actor update for each agent
        for i, agent in enumerate(self.agents):
            # Critic Loss construction
            # Compute target critic values (with no_grad to avoid unnecessary graph)
            with torch.no_grad():
                critic_value_ = agent.target_critic(next_critic_input_cat, next_actions_cat).squeeze(1)  # (B,)
            rewards_i = torch.tensor(rewards[:, i], dtype=torch.float32).to(self.device)  # (B,)
            dones_i = torch.tensor(dones[:, i], dtype=torch.float32).to(self.device)  # (B,)
            target = rewards_i + agent.gamma * (1.0 - dones_i) * critic_value_
            
            # Compute current critic value
            critic_value = agent.critic(critic_input_cat, old_actions_cat).squeeze(1)
            # Apply MSE Loss
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()
            # Construct Actor Loss (detach other agents' actions already done; no grad to critic by using separate forward)
            mu_actor_loss = []
            for j in range(self.n_):
                if i == j:
                    mu_actor_loss.append(all_mu_actions[j])
                else:
                    mu_actor_loss.append(all_mu_actions[j].detach())
                    
            mu_cat_actor_loss = torch.cat(mu_actor_loss, dim=-1)
            actor_q = agent.critic(critic_input_cat, mu_cat_actor_loss).squeeze(1)
            actor_loss = -torch.mean(actor_q)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.5)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()
            agent.update_target_networks()
