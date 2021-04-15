import numpy as np
from typing import Dict, List, Mapping, NamedTuple, cast, Tuple, Optional
from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch.networks import ValueNetwork, EncodedValueNetwork, SimpleActor
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, DDPGSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


class TorchDDPGOptimizer(TorchOptimizer):

    class TargetEntropy(NamedTuple):

        discrete: List[float] = []  # One per branch
        continuous: float = 0.0

    class LogEntCoef(nn.Module):
        def __init__(self, discrete, continuous):
            super().__init__()
            self.discrete = discrete
            self.continuous = continuous

    def __init__(self, policy: TorchPolicy, trainer_params: TrainerSettings):
        super().__init__(policy, trainer_params)
        reward_signal_configs = trainer_params.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        if policy.shared_critic:
            raise UnityTrainerException("DDPG does not support SharedActorCritic")
        
        hyperparameters: DDPGSettings = cast(DDPGSettings, trainer_params.hyperparameters)
        self.hyperparameters = hyperparameters

        self.tau = hyperparameters.tau
        self.init_entcoef = hyperparameters.init_entcoef

        self.policy = policy
        policy_network_settings = policy.network_settings

        self.tau = hyperparameters.tau
        self.burn_in_ratio = 0.0

        # Non-exposed SAC parameters
        self.discrete_target_entropy_scale = 0.2  # Roughly equal to e-greedy 0.05
        self.continuous_target_entropy_scale = 1.0

        self.stream_names = list(self.reward_signals.keys())
        # Use to reduce "survivor bonus" when using Curiosity or GAIL.
        self.gammas = [_val.gamma for _val in trainer_params.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }
        self._action_spec = self.policy.behavior_spec.action_spec

        # Use the encoder in value networks
        self._critic = EncodedValueNetwork(
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            policy.network_settings,
            act_size = int(self._action_spec.continuous_size),
            feature_size = hyperparameters.feature_size
        )

        self.target_network = EncodedValueNetwork(
            # self.policy.encoder,
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            policy.network_settings,
            act_size = int(self._action_spec.continuous_size),
            feature_size = hyperparameters.feature_size
        )

        self.target_actor = SimpleActor(
            observation_specs=policy.behavior_spec.observation_specs,
            network_settings=trainer_params.network_settings,
            action_spec=policy.behavior_spec.action_spec,
            conditional_sigma=True,
            tanh_squash=True,
            det_action=self.policy.actor.det_action
        )
        print("---critic----")
        print(self._critic)
        print("---target critic----")
        print(self.target_network)
        print("---actor----")
        print(self.policy.actor)
        print("---target----")
        print(self.target_actor)

        ModelUtils.soft_update(self._critic, self.target_network, 1.0)
        ModelUtils.soft_update(self.policy.actor, self.target_actor, 1.0)

        # We create one entropy coefficient per action, whether discrete or continuous.
        _disc_log_ent_coef = torch.nn.Parameter(
            torch.log(
                torch.as_tensor(
                    [self.init_entcoef] * len(self._action_spec.discrete_branches)
                )
            ),
            requires_grad=True,
        )
        _cont_log_ent_coef = torch.nn.Parameter(
            torch.log(torch.as_tensor([self.init_entcoef])), requires_grad=True
        )
        self._log_ent_coef = TorchDDPGOptimizer.LogEntCoef(
            discrete=_disc_log_ent_coef, continuous=_cont_log_ent_coef
        )
        _cont_target = (
            -1
            * self.continuous_target_entropy_scale
            * np.prod(self._action_spec.continuous_size).astype(np.float32)
        )
        _disc_target = [
            self.discrete_target_entropy_scale * np.log(i).astype(np.float32)
            for i in self._action_spec.discrete_branches
        ]
        self.target_entropy = TorchDDPGOptimizer.TargetEntropy(
            continuous=_cont_target, discrete=_disc_target
        )
        policy_params = list(self.policy.actor.parameters())
        model_params = list(self.policy.model.parameters())
        value_params = list(self._critic.parameters())
        encoder_params = list(self._critic.encoder.parameters())

        # for name, params in self._critic.named_parameters():
        #     print("critic params", name, params)

        logger.debug("value_vars")
        for param in value_params:
            logger.debug(param.shape)
        logger.debug("policy_vars")
        for param in policy_params:
            logger.debug(param.shape)

        self.decay_learning_rate = ModelUtils.DecayedValue(
            hyperparameters.learning_rate_schedule,
            hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_model_learning_rate = ModelUtils.DecayedValue(
            hyperparameters.model_lr_schedule,
            hyperparameters.model_learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )

        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=hyperparameters.learning_rate
        )
        
        if not self.hyperparameters.transfer_target:
            # source task learning, train the encoder and q networks, and fit a model
            self.value_optimizer = torch.optim.Adam(
                value_params, lr=hyperparameters.learning_rate
            ) 
            self.model_optimizer = torch.optim.Adam(
                model_params, lr=hyperparameters.model_learning_rate
            ) 
        else:
            self.value_optimizer = torch.optim.Adam(
                value_params, lr=hyperparameters.learning_rate
            )  
            self.model_optimizer = torch.optim.Adam(
                model_params, lr=hyperparameters.model_learning_rate
            ) 

        self.entropy_optimizer = torch.optim.Adam(
            self._log_ent_coef.parameters(), lr=hyperparameters.learning_rate
        )
        self._move_to_device(default_device())

    @property
    def critic(self):
        return self._critic

    def _move_to_device(self, device: torch.device) -> None:
        self._log_ent_coef.to(device)
        self.target_network.to(device)
        self._critic.to(device)
        self.target_actor.to(device)

    def ddpg_value_loss(
        self,
        obs,
        actions,
        q_memories,
        target_values: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        rewards: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cont_actions = actions.continuous_tensor
        q_out, _ = self._critic.q_pass(
            obs, 
            cont_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
        )
        q_losses = []
        # Multiple q losses per stream
        for i, name in enumerate(q_out.keys()):
            q_stream = q_out[name].squeeze()
            with torch.no_grad():
                q_backup = rewards[name] + (
                    (1.0 - self.use_dones_in_backup[name] * dones)
                    * self.gammas[i]
                    * target_values[name]
                )
            _q_loss = 0.5 * ModelUtils.masked_mean(
                torch.nn.functional.mse_loss(q_backup, q_stream), loss_masks
            )

            q_losses.append(_q_loss)
        q_loss = torch.mean(torch.stack(q_losses))
        return q_loss

    def ddpg_policy_loss(
        self,
        obs,
        act_masks,
        memories,
        q_memories,
        loss_masks
    ) -> torch.Tensor:

        sampled_actions = self.policy.actor.get_action_and_stats(
            obs,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        if not self.policy.actor.det_action:
            sampled_actions = sampled_actions[0]
        # print("action", sampled_actions[0])
        cont_sampled_actions = sampled_actions.continuous_tensor

        sampled_q, _ = self._critic.q_pass(
            obs,
            cont_sampled_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length
        )

        losses = []
        # Multiple q losses per stream
        for i, name in enumerate(sampled_q.keys()):
            q_stream = sampled_q[name].squeeze()
            loss = - 0.5 * ModelUtils.masked_mean(
                q_stream, loss_masks
            )
            losses.append(loss)
        policy_loss = torch.mean(torch.stack(losses))
        return policy_loss

    
    def ddpg_model_loss(
        self,
        obs,
        next_obs,
        actions,
        reward,
        memories,
        sequence_length,
    )-> torch.Tensor:
        
        encoded_next, _ = self.critic.encoder(
            obs,
            None, 
            memories,
            sequence_length
        )
        predict_next, predict_reward = self.policy.model(
            self.critic.encoder,
            obs,
            actions,
            memories,
            sequence_length,
            not self.hyperparameters.transfer_target
        )
        
        loss_fn = torch.nn.MSELoss()

        # print("encoded next", encoded_next)
        # print("pred next", predict_next)
        # print("rew", reward)
        # print("pred rew", predict_reward.squeeze())
        if not self.hyperparameters.transfer_target or self.hyperparameters.detach_next:
            encoded_next = encoded_next.detach()
        model_loss = loss_fn(encoded_next, predict_next) + loss_fn(reward, predict_reward.squeeze())

        return model_loss

    def _condense_q_streams(
        self, q_output: Dict[str, torch.Tensor], discrete_actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        condensed_q_output = {}
        onehot_actions = ModelUtils.actions_to_onehot(
            discrete_actions, self._action_spec.discrete_branches
        )
        for key, item in q_output.items():
            branched_q = ModelUtils.break_into_branches(
                item, self._action_spec.discrete_branches
            )
            only_action_qs = torch.stack(
                [
                    torch.sum(_act * _q, dim=1, keepdim=True)
                    for _act, _q in zip(onehot_actions, branched_q)
                ]
            )

            condensed_q_output[key] = torch.mean(only_action_qs, dim=0)
        return condensed_q_output

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param batch: Experience mini-batch.
        :param update_target: Whether or not to update target value network
        :param reward_signal_batches: Minibatches to use for updating the reward signals,
            indexed by name. If none, don't update the reward signals.
        :return: Output from update process.
        """
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        # Convert to tensors
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        # LSTM shouldn't have sequence length <1, but stop it from going out of the index if true.
        value_memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]
        offset = 1 if self.policy.sequence_length > 1 else 0
        next_value_memories_list = [
            ModelUtils.list_to_tensor(
                batch[BufferKey.CRITIC_MEMORY][i]
            )  # only pass value part of memory to target network
            for i in range(
                offset, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]

        if len(memories_list) > 0:
            memories = torch.stack(memories_list).unsqueeze(0)
            value_memories = torch.stack(value_memories_list).unsqueeze(0)
            next_value_memories = torch.stack(next_value_memories_list).unsqueeze(0)
        else:
            memories = None
            value_memories = None
            next_value_memories = None

        # Q and V network memories are 0'ed out, since we don't have them during inference.
        q_memories = (
            torch.zeros_like(next_value_memories)
            if next_value_memories is not None
            else None
        )

        # Copy normalizers from policy
        self._critic.encoder.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_network.encoder.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_actor.network_body.copy_normalization(
            self.policy.actor.network_body
        )

        
        with torch.no_grad():
            target_actions = self.target_actor.get_action_and_stats(
                next_obs,
                memories=next_value_memories,
                sequence_length=self.policy.sequence_length,
            )
            if not self.policy.actor.det_action:
                target_actions = target_actions[0]

            target_q, _ = self.target_network.q_pass(
                next_obs, 
                target_actions.continuous_tensor,
                memories=next_value_memories,
                sequence_length=self.policy.sequence_length,
            )

        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        
        # model_loss = self.sac_model_loss(
        #     current_obs, 
        #     next_obs, 
        #     actions, 
        #     rewards["extrinsic"], 
        #     memories=value_memories,
        #     sequence_length=self.policy.sequence_length,
        # )
        # for name, p in self.policy.actor.named_parameters():
        #     print("before", name, p)
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr*0.1)
        self.policy_optimizer.zero_grad()
        policy_loss = self.ddpg_policy_loss(current_obs, act_masks, memories, q_memories, masks)
        policy_loss.backward()
        # for name, p in self.policy.actor.named_parameters():
        #     print("grad", name, p.grad)
        self.policy_optimizer.step()
        # print("before", policy_loss.item())
        # new_policy_loss = self.ddpg_policy_loss(current_obs, act_masks, memories, q_memories, masks)
        # print("after", new_policy_loss.item())
        
        # for name, p in self.policy.actor.named_parameters():
        #     print("after", name, p)

        if not self.hyperparameters.transfer_target:
            ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
            self.value_optimizer.zero_grad()
            value_loss = self.ddpg_value_loss(
                current_obs, actions, q_memories, target_q, dones, rewards, masks
            )
            value_loss.backward()
            self.value_optimizer.step()

            # decay_model_lr = self.decay_model_learning_rate.get_value(self.policy.get_current_step())
            # ModelUtils.update_learning_rate(self.model_optimizer, decay_model_lr)
            # self.model_optimizer.zero_grad()
            # model_loss.backward()
            # self.model_optimizer.step()
        
        else:
            ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
            self.value_optimizer.zero_grad()
            (total_value_loss + 0.5 * model_loss).backward()
            self.value_optimizer.step()
        
        # for name, p in self.policy.actor.named_parameters():
        #     print("actor", name, p)
        # for name, p in self._critic.named_parameters():
        #     print("critic", name, p) 

        # Update target network
        ModelUtils.soft_update(self._critic, self.target_network, self.tau)
        ModelUtils.soft_update(self.policy.actor, self.target_actor, self.tau)
        update_stats = {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Value Loss": value_loss.item(),
            # "Losses/Model Loss": model_loss.item(),
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats

    def update_reward_signals(
        self, reward_signal_minibatches: Mapping[str, AgentBuffer], num_sequences: int
    ) -> Dict[str, float]:
        update_stats: Dict[str, float] = {}
        for name, update_buffer in reward_signal_minibatches.items():
            update_stats.update(self.reward_signals[name].update(update_buffer))
        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:target_network": self.target_network,
            "Optimizer:target_actor": self.target_actor,
            "Optimizer:policy_optimizer": self.policy_optimizer,
            "Optimizer:value_optimizer": self.value_optimizer,
            # "Optimizer:entropy_optimizer": self.entropy_optimizer,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
