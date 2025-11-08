import torch
import numpy as np
import random
import torch.nn.functional as F
from agent import Agent

# ASSUME: each agent has the same obs_dim and n_actions
torch.autograd.set_detect_anomaly(True)
class SQDDPG:
    def __init__(self, critic_dims, actor_dims, n_agents, n_actions, batch_size, 
                 sample_size, chkpt_dir, fc1=64, fc2=64, alpha=0.01, beta=0.01, 
                 gamma=0.99, tau=0.01, evaluate=False):
        self.n_ = n_agents
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.obs_dim = critic_dims
        self.actor_dims = actor_dims
        self.agents = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for agent_idx in range(n_agents):
            agent = Agent(critic_dims, actor_dims[agent_idx], fc1, fc2, agent_idx, 
              n_agents, n_actions, chkpt_dir=chkpt_dir, alpha=alpha, beta=beta, 
              gamma=gamma, tau=tau, evaluate=evaluate)
            self.agents.append(agent)
            
    def choose_action(self, obs, noise_std):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(obs[i], noise_std)
            actions.append(action)
            
        return actions
    
    def sample_grandcoalitions(self, batch_size):
        seq_set = torch.tril(torch.ones(self.n_, self.n_), diagonal=0, out=None)
        grand_coalitions_pos = torch.multinomial(torch.ones(batch_size*self.sample_size, self.n_)/self.n_, self.n_, replacement=False) # shape = (b*n_s, n)
        individual_map = torch.zeros(batch_size*self.sample_size*self.n_, self.n_)
        individual_map.scatter_(1, grand_coalitions_pos.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_, self.n_)
        subcoalition_map = torch.matmul(individual_map, seq_set)

        # FIX: construct torche grand coalition (in sequence by agent_idx) from torche grand_coalitions_pos (e.g., pos_idx <- grand_coalitions_pos[agent_idx])
        offset = (torch.arange(batch_size*self.sample_size)*self.n_).reshape(-1, 1)
        grand_coalitions_pos_alter = grand_coalitions_pos + offset
        grand_coalitions = torch.zeros_like(grand_coalitions_pos_alter.flatten())
        grand_coalitions[grand_coalitions_pos_alter.flatten()] = torch.arange(batch_size*self.sample_size*self.n_)
        grand_coalitions = grand_coalitions.reshape(batch_size*self.sample_size, self.n_) - offset

        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size, \
            self.n_, self.n_).contiguous().view(batch_size, self.sample_size, self.n_, self.n_) # shape = (b, n_s, n, n)

        return subcoalition_map.to(self.device).detach(), grand_coalitions.to(self.device).detach()

    def marginal_contribution(self, obs, act, is_target=False):
        batch_size = self.batch_size
        # Assume obs and act are lists of per-agent tensors: obs[i] has shape (batch_size, obs_dim_i), act[i] has shape (batch_size, n_actions)
        # Move to device
        obs = [o.to(self.device) for o in obs]
        act = [a.to(self.device) for a in act]
        # Concatenate observations into a global observation tensor
        global_obs = torch.cat(obs, dim=1)  # shape = (batch_size, total_obs_dim), where total_obs_dim = sum(obs_dim_i for all agents)
        # Stack actions into a joint action tensor assuming uniform action dimensions across agents
        act = torch.stack(act, dim=1)  # shape = (batch_size, n_, n_actions)
        
        subcoalition_map, grand_coalitions = self.sample_grandcoalitions(batch_size) # shape = (b, n_s, n, n)
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, self.sample_size, self.n_, self.n_, self.n_actions) # shape = (b, n_s, n, n, a)
        act = act.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.n_actions).gather(3, grand_coalitions)
	    # shape = (b, n, a) -> (b, 1, 1, n, a) -> (b, n_s, n, n, a)
        act_map = subcoalition_map.unsqueeze(-1).float() # shape = (b, n_s, n, n, 1)
        act = act * act_map
        act = act.contiguous().view(batch_size, self.sample_size, self.n_, -1) # shape = (b, n_s, n, n*a)
        
        # Expand the global_obs (full concatenated observations) across sample_size and n_ dimensions
        # No need to split per agent since the critic expects the full concatenated obs regardless of per-agent dims
        total_obs_dim = global_obs.shape[1]
        global_obs = global_obs.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, total_obs_dim) # shape = (b, n_s, n, total_obs_dim)
        
        values = []
        if not is_target:
            for i, agent in enumerate(self.agents):
                value = agent.critic(global_obs[:, :, i, :], act[:, :, i, :])
                values.append(value)
            values = torch.stack(values, dim=2)
        else:
            for i, agent in enumerate(self.agents):
                value = agent.target_critic(global_obs[:, :, i, :], act[:, :, i, :])
                values.append(value)
            values = torch.stack(values, dim=2)
            
        return values
    
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
        # batch_size = memory.batch_size
        # shapes:
        # obs: (n_, B, actor_dim)
        # actions: (n_, B, n_actions)
        # rewards: (B, n_)
        # next_obs: (n_, B, actor_dim)
        # dones: (B, n_)
        
        # Calculate per agent actor outputs(current mu) and target actor for next_state
        all_next_actions = []
        all_mu_actions = []
        old_agents_actions = [] 
        next_critic_inputs = []
        critic_inputs = []
        for agent_idx, agent in enumerate(self.agents):
            # get the desired observation for corresponding agent
            obs_i = torch.tensor(obs[agent_idx], dtype=torch.float32).to(self.device) # obs_i: (batch, actor_dim)
            next_obs_i = torch.tensor(next_obs[agent_idx], dtype=torch.float32).to(self.device) # obs_i: (batch, actor_dim)
            # convert to torch for torch.cat
            action_i = torch.tensor(actions[agent_idx], dtype=torch.float32).to(self.device)
        
            # compute mu i.e deterministic policy
            mu_i = agent.actor(obs_i)
            next_mu_i = agent.target_actor(next_obs_i)
            all_mu_actions.append(mu_i)
            all_next_actions.append(next_mu_i)
            old_agents_actions.append(action_i)
            next_critic_inputs.append(next_obs_i)
            critic_inputs.append(obs_i)
            
        # Concatenate actions for critic (dim = batch, n_agents * act_dim)
        # mu_cat = torch.cat(all_mu_actions, dim=-1).to(self.device)
        # next_actions_cat = torch.cat(all_next_actions, dim=-1).to(self.device)
        # old_actions_cat = torch.cat(old_agents_actions, dim=-1).to(self.device) # (B, n_*act_dim)
        # critic_input_cat = torch.cat(critic_inputs, dim=-1).to(self.device) # (B, n_*obs_dim)
        # next_critic_input_cat = torch.cat(next_critic_inputs, dim=-1).to(self.device)

        # Compute shapley value 
        # shapley_values_sum = self.marginal_contribution(critic_input_cat, old_actions_cat).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(self.batch_size, self.n_)
        
        # next_shapley_values_sum = self.marginal_contribution(next_critic_input_cat, next_actions_cat, is_target=True).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(self.batch_size, self.n_)
        
        # shapley_values = self.marginal_contribution(critic_input_cat, mu_cat).mean(dim=1).contiguous().view(-1, self.n_)
        # MAIN: critic and actor update for each agent
        # CAUTION: detach all actions that are not of the current agent due to grad computing 
        for i, agent in enumerate(self.agents):
            next_actions = []
            old_actions = []
            mu_actor_loss = []
            for j in range(self.n_):
                if i==j:
                    next_actions.append(all_next_actions[j])
                    old_actions.append(old_agents_actions[j])
                    mu_actor_loss.append(all_mu_actions[j])
                else:
                    next_actions.append(all_next_actions[j].detach())
                    old_actions.append(old_agents_actions[j].detach())
                    mu_actor_loss.append(all_mu_actions[j].detach())
                    
            # next_actions_cat = torch.cat(next_actions, dim=-1).to(self.device)
            # old_actions_cat = torch.cat(old_agents_actions, dim=-1).to(self.device) # (B, n_*act_dim)
            # mu_cat_actor_loss = torch.cat(mu_actor_loss, dim=-1).to(self.device)

            shapley_values_sum = self.marginal_contribution(critic_inputs, old_agents_actions).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(self.batch_size, self.n_)
            
            next_shapley_values_sum = self.marginal_contribution(next_critic_inputs, next_actions, is_target=True).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(self.batch_size, self.n_)
        
            critic_value_ = next_shapley_values_sum[:, i].detach()  # replace target critic with shapley bootstrap
            
            rewards_i = torch.tensor(rewards[:, i], dtype=torch.float32, device=self.device)
            dones_i = torch.tensor(dones[:, i], dtype=torch.float32, device=self.device)
            global_rewards = torch.sum(rewards, dim=1) 
            target = global_rewards + agent.gamma * (1 - dones_i) * critic_value_
            critic_value = shapley_values_sum[:, i] #agent.critic(critic_input_cat, old_actions_cat).squeeze(1)
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()
            
            # Actor loss using shapley advantages
            shapley_values = self.marginal_contribution(critic_inputs, mu_actor_loss).mean(dim=1).contiguous().view(-1, self.n_)
            actor_loss = -torch.mean(shapley_values[:, i])
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.5)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()
            # soft update the target
            agent.update_target_networks()
            
