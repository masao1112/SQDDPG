import os
import torch
import numpy as np
from utilities import _update_target_networks
from networks import CriticNetwork, ActorNetwork

class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, fc1_dim, fc2_dim,
                 agent_idx, n_agents, n_actions, chkpt_dir,
                 alpha=0.01, beta=0.01, gamma=0.99, tau=0.01, evaluate=False):
        
        self.tau = tau
        self.gamma = gamma
        self.n_ = n_agents
        self.evaluate = evaluate
        self.n_actions = n_actions
        agent_name = 'agent_%s' % agent_idx
        self.critic = CriticNetwork(beta, obs_dim, fc1_dim, fc2_dim,
                                   n_agents, n_actions, name=agent_name+"_critic.zip", chkpt_dir=chkpt_dir)
        self.target_critic = CriticNetwork(beta, obs_dim, fc1_dim, fc2_dim,
                                   n_agents, n_actions, name=agent_name+"_target_critic.zip", chkpt_dir=chkpt_dir)
        
        self.actor = ActorNetwork(alpha, act_dim, fc1_dim, fc2_dim, 
                                  n_actions, name=agent_name+"_actor.zip", chkpt_dir=chkpt_dir)
        self.target_actor = ActorNetwork(alpha, act_dim, fc1_dim, fc2_dim, 
                                  n_actions, name=agent_name+"_target_actor.zip", chkpt_dir=chkpt_dir)
        
        # perform hard update at initialization
        self.update_target_networks(tau=1)
        
    def choose_action(self, state, noise_std):
        with torch.no_grad(): 
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.actor.device)
            actions = self.actor(state)
            if not self.evaluate:
                noise = torch.randn_like(actions) * noise_std # gaussian noise
                action = torch.clamp(actions + noise, 0, 1)
            else:
                action = actions
            return action.detach().cpu().numpy()[0]

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        _update_target_networks(self.target_actor, self.actor, tau)
        # _update_target_networks(self.target_critic, self.critic, tau)
        
            
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

class SQDDPGAgent:
    def __init__(self, obs_dim, act_dim, fc1_dim, fc2_dim,
                 agent_idx, n_agents, n_actions, chkpt_dir,
                 alpha=0.01, beta=0.01, gamma=0.99, tau=0.01, evaluate=False):
        
        self.tau = tau
        self.gamma = gamma
        self.n_ = n_agents
        self.evaluate = evaluate
        self.n_actions = n_actions
        agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, act_dim, fc1_dim, fc2_dim, 
                                  n_actions, name=agent_name+"_actor.zip", chkpt_dir=chkpt_dir)
        self.target_actor = ActorNetwork(alpha, act_dim, fc1_dim, fc2_dim, 
                                  n_actions, name=agent_name+"_target_actor.zip", chkpt_dir=chkpt_dir)
        
        # perform hard update at initialization
        self.update_target_networks(tau=1)
        
    def choose_action(self, state, noise_std):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.actor.device)
            actions = self.actor(state)
            if not self.evaluate:
                noise = torch.randn_like(actions) * noise_std # gaussian noise
                action = torch.clamp(actions + noise, 0, 1)
            else:
                action = actions
            return action.cpu().numpy()[0]

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        _update_target_networks(self.target_actor, self.actor, tau)
        # _update_target_networks(self.target_critic, self.critic, tau)
        
            
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        # self.critic.save_checkpoint()
        # self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        # self.critic.load_checkpoint()
        # self.target_critic.load_checkpoint()