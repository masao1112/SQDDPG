# sqddpg.py
import torch
import torch.nn.functional as F
from helper.utilities import _update_target_networks
from helper.networks import CriticNetwork
from helper.agent import SQDDPGAgent


class SQDDPG:
    def __init__(self, critic_dims, actor_dims, n_agents, n_actions, batch_size,
                 sample_size, chkpt_dir, fc1=64, fc2=64,
                 alpha=0.01, beta=0.01, gamma=0.99, tau=0.01, evaluate=False):
        self.n_ = n_agents
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- GLOBAL CRITIC ----------
        self.global_critic = CriticNetwork(
            beta, critic_dims, fc1, fc2, n_agents, n_actions,
            name="global_critic.zip", chkpt_dir=chkpt_dir).to(self.device)

        self.global_target_critic = CriticNetwork(
            beta, critic_dims, fc1, fc2, n_agents, n_actions,
            name="global_target_critic.zip", chkpt_dir=chkpt_dir).to(self.device)

        _update_target_networks(self.global_target_critic, self.global_critic, tau=1.0)

        # ---------- AGENTS (only actors) ----------
        self.agents = []
        for idx in range(n_agents):
            agent = SQDDPGAgent(
                obs_dim=critic_dims,          # not used for critic any more
                act_dim=actor_dims[idx],
                fc1_dim=fc1, fc2_dim=fc2,
                agent_idx=idx, n_agents=n_agents,
                n_actions=n_actions, chkpt_dir=chkpt_dir,
                alpha=alpha, gamma=gamma, tau=tau, evaluate=evaluate)
            self.agents.append(agent)

    # ------------------------------------------------------------------ #
    #  Action selection
    # ------------------------------------------------------------------ #
    def choose_action(self, obs, noise_std):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(obs[i], noise_std)
            actions.append(action)
        return actions
    # ------------------------------------------------------------------ #
    #  Learning
    # ------------------------------------------------------------------ #
    def learn(self, memory):
        if not memory.ready():
            return

        # ----- sample -----
        obs, actions, rewards, next_obs, dones = memory.sample_buffer()

        # ----- to torch (B, ...) -----
        obs = [torch.tensor(o, dtype=torch.float32, device=self.device) for o in obs] # (n_, B, obs_dim)
        next_obs = [torch.tensor(o, dtype=torch.float32, device=self.device) for o in next_obs]
        actions = [torch.tensor(a, dtype=torch.float32, device=self.device) for a in actions] # (n_, B, actor_dim)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)   # (B, n_)
        dones   = torch.tensor(dones[:, 0], dtype=torch.float32, device=self.device)  # (B,) if one agent is done, others as well

        # ----- concatenate for critic -----
        critic_inputs      = torch.cat(obs, dim=-1)                     # (B, sum_obs)
        next_critic_inputs = torch.cat(next_obs, dim=-1)
        old_actions_cat = torch.cat(actions, dim=-1)                   # (B, n_*act)
        # ----- target actions (mu') -----
        with torch.no_grad():
           next_actions = []
           next_actions = [ag.target_actor(o.unsqueeze(0)).squeeze(0)
                           for ag, o in zip(self.agents, next_obs)]

           next_actions_cat = torch.cat(next_actions, dim=-1)

        # ----- Shapley marginal contributions -----
        shapley_sum = self.marginal_contribution(
            critic_inputs, old_actions_cat, is_target=False)      # (B, M, n_)
        next_shapley_sum = self.marginal_contribution(
            next_critic_inputs, next_actions_cat, is_target=True)

        shapley_sum = shapley_sum.mean(dim=1).sum(dim=1)          # (B,)  global Q
        next_shapley_sum = next_shapley_sum.mean(dim=1).sum(dim=1)

        # ---------- GLOBAL CRITIC UPDATE (once per batch) ----------
        target = rewards.sum(dim=1) + self.gamma * (1 - dones) * next_shapley_sum.detach()
        critic_loss = F.mse_loss(shapley_sum, target)
        self.global_critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), 0.5)
        self.global_critic.optimizer.step()
        self.global_critic.scheduler.step()
        print(critic_loss)
        # ---------- PER-AGENT ACTOR UPDATE ----------
        for i, agent in enumerate(self.agents):
            # mu actions for this agent, detach others
            mu_actions = [self.agents[j].actor(obs[j].unsqueeze(0)).squeeze(0)
                          if j == i else self.agents[j].actor(obs[j].unsqueeze(0)).detach().squeeze(0)
                          for j in range(self.n_)]

            mu_cat = torch.cat(mu_actions, dim=-1)

            shapley = self.marginal_contribution(
                critic_inputs, mu_cat, is_target=False)          # (B, M, n_)
            shapley = shapley.mean(dim=1)[:, i]                  # (B,) get the shapley Q of agent ith

            actor_loss = -shapley.mean()
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # soft-update actor target
            agent.update_target_networks()

        # ----- soft-update global target critic (once per batch) -----
        _update_target_networks(self.global_target_critic, self.global_critic, self.tau)

    # ------------------------------------------------------------------ #
    #  Shapley marginal contribution (Monte-Carlo)
    # ------------------------------------------------------------------ #
    def marginal_contribution(self, obs_batch, actions_batch, is_target=False):
        """
        obs_batch      : (B, sum_obs)
        actions_batch  : (B, n_*act)
        Returns        : (B, sample_size, n_agents)   marginals per agent
        """
        B = obs_batch.shape[0]
        critic = self.global_target_critic if is_target else self.global_critic

        # random permutations (sample_size per batch item)
        perms = torch.stack([torch.randperm(self.n_) for _ in range(B * self.sample_size)])
        perms = perms.view(B, self.sample_size, self.n_).to(self.device)   # (B, M, n_)

        # repeat inputs for each permutation
        obs_rep    = obs_batch.unsqueeze(1).repeat(1, self.sample_size, 1).view(B*self.sample_size, -1)
        actions_rep = actions_batch.unsqueeze(1).repeat(1, self.sample_size, 1).view(B*self.sample_size, -1)

        # Start from empty coalition
        cur_actions = torch.zeros_like(actions_rep)
        Q_prev = critic(obs_rep, cur_actions).squeeze(-1)  # (B*M,)

        # Marginals per position, then remap to agents
        marginals_pos = torch.zeros(B * self.sample_size, self.n_, device=self.device)

        for k in range(self.n_):
            # Entering agents for this position (flat)
            enter_flat = perms[:, :, k].flatten()  # (B*M,)

            # Copy entering agent's actions into current coalition
            # View as (B*M, n_agents, n_actions)
            cur_view = cur_actions.view(B * self.sample_size, self.n_, self.n_actions)
            rep_view = actions_rep.view(B * self.sample_size, self.n_, self.n_actions)
            cur_view[torch.arange(B * self.sample_size), enter_flat] = rep_view[torch.arange(B * self.sample_size), enter_flat]

            Q_cur = critic(obs_rep, cur_actions).squeeze(-1)
            marginals_pos[:, k] = Q_cur - Q_prev
            Q_prev = Q_cur

        # Remap position marginals to actual agent IDs
        shapley = torch.zeros(B * self.sample_size, self.n_, device=self.device)
        for k in range(self.n_):
            enter_flat = perms[:, :, k].flatten()  # (B*M,)
            shapley[torch.arange(B * self.sample_size), enter_flat] = marginals_pos[:, k]

        shapley = shapley.view(B, self.sample_size, self.n_)
        return shapley

    # ------------------------------------------------------------------ #
    #  Checkpoint handling
    # ------------------------------------------------------------------ #
    def save_checkpoint(self):
        print("...saving checkpoints...")
        self.global_critic.save_checkpoint()
        self.global_target_critic.save_checkpoint()
        for a in self.agents:
            a.save_models()

    def load_checkpoint(self):
        print("...loading checkpoints...")
        self.global_critic.load_checkpoint()
        self.global_target_critic.load_checkpoint()
        for a in self.agents:
            a.load_models()
