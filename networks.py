import os
import torch
from torch import nn, optim

class CriticNetwork(nn.Module):
    """
    Args:
        actor_dims: list of actor networks' input shapes
        critic_dims: critic's input shape, the same for every critic nets
        n_agents: number of agents
        n_actions: number of actions
    
    Return:
        nn.ModuleList of all critic nets
    """
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir="tmp/maddpg"):
        """
        Args: 
            beta: learning rate for critic nets
            input_dims: normally from env.observation_space
            fc1_dims: the 1st fully connected layer
            fc2_dims: the 2nd fully connected layer
            n_agents: number of agents
            n_actions: number of actions
            shared_parameters: whether to share parameters within a network or not 
            name: critic name, for tracking purposes
            chkpt_dir: where to save the model
            
        Return:
            A list of critic nets: nn.ModuleList
        """
        super(CriticNetwork, self).__init__()
        actual_input_dims = input_dims+n_actions*n_agents  
        self.layers = nn.Sequential(
            nn.Linear(actual_input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # define an optimizer for this network
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.1)
        # move tensors to device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)

        
    def forward(self, obs, actions):
        sa_concat = torch.concat([obs, actions], dim=-1) # obs: (n_, n_state), actions: (n_, 1)
        critic_value = self.layers(sa_concat) # forward pass
        
        return critic_value
    
    def save_checkpoint(self):
        """Save model to checkpoint path"""
        # create directories if not exists
        os.makedirs(self.chkpt_dir, exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)
        
    def load_checkpoint(self):
        """Load model from checkpoint path"""
        self.load_state_dict(torch.load(self.chkpt_file, weights_only=True, map_location=torch.device('cpu')))        

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                n_actions, name, chkpt_dir="tmp/maddpg"):
        super(ActorNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Sigmoid() # for continuous actions
        )
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # define an optimizer for this network
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.1)
        # move tensors to device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)
        
    def forward(self, state):
        pi = self.layers(state)
        return pi
    
    def save_checkpoint(self):
        """Save model to checkpoint path"""
        # create directories if not exists
        os.makedirs(self.chkpt_dir, exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)
        
    def load_checkpoint(self):
        """Load model from checkpoint path"""
        self.load_state_dict(torch.load(self.chkpt_file, weights_only=True, map_location=torch.device('cpu')))