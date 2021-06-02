import numpy as np
from typing import Dict, List, Mapping, NamedTuple, cast, Tuple, Optional
from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.policy.dqn_policy import DQNPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch.networks import ValueNetwork, EncodedQNetwork
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, DQNSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


class TorchDQNOptimizer(TorchOptimizer):

    def __init__(self, policy: DQNPolicy, trainer_params: TrainerSettings):
        super().__init__(policy, trainer_params)
        reward_signal_configs = trainer_params.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        if policy.shared_critic:
            raise UnityTrainerException("DQN does not support SharedActorCritic")

        self.hyperparameters: DQNSettings = cast(DQNSettings, trainer_params.hyperparameters)
        self.tau = self.hyperparameters.tau
        self.init_entcoef = self.hyperparameters.init_entcoef

        self.policy = policy
        policy_network_settings = policy.network_settings

        self.burn_in_ratio = 0.0

        self.stream_names = list(self.reward_signals.keys())
        # Use to reduce "survivor bonus" when using Curiosity or GAIL.
        self.gammas = [_val.gamma for _val in trainer_params.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }

        self._action_spec = self.policy.behavior_spec.action_spec

        # self.q_network = ValueNetwork(
        #     self.stream_names,
        #     self.policy.behavior_spec.observation_specs,
        #     policy_network_settings,
        #     int(self._action_spec.continuous_size),
        #     max(sum(self._action_spec.discrete_branches), 1)
        # )
        self.q_network = self.policy.q_network

        # self.target_network = ValueNetwork(
        #     self.stream_names,
        #     self.policy.behavior_spec.observation_specs,
        #     policy_network_settings,
        #     int(self._action_spec.continuous_size),
        #     max(sum(self._action_spec.discrete_branches), 1)
        # )
        self.target_network = EncodedQNetwork(
            # self.policy.encoder,
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
            self._action_spec,
            self.hyperparameters.feature_size
        )
        print("target network\n", self.target_network)

        ModelUtils.soft_update(self.q_network, self.target_network, 1.0)

        for name, param in self.q_network.named_parameters():
            print("q net param\n", name, param)

        for name, param in self.target_network.named_parameters():
            print("target net param\n", name, param)

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )

        self.decay_model_lr = ModelUtils.DecayedValue(
            self.hyperparameters.model_lr_schedule,
            self.hyperparameters.model_learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )

        if not self.hyperparameters.transfer_target:
            # source task learning, train the encoder and q networks, and fit a model
            self.value_optimizer = torch.optim.Adam(
                [{"params": self.policy.encoder.parameters()}, 
                {"params": self.q_network.parameters()}], 
                lr=self.hyperparameters.learning_rate
            )

            self.model_optimizer = torch.optim.Adam(
                self.policy.model.parameters(), 
                lr=self.hyperparameters.model_learning_rate
            )
        else:
            self.value_optimizer = torch.optim.Adam(
                [{"params": self.policy.encoder.parameters()}, 
                {"params": self.q_network.parameters()}], 
                lr=self.hyperparameters.learning_rate
            )

            self.model_optimizer = None
            # self.value_optimizer = torch.optim.Adam(
            #     self.q_network.parameters(), 
            #     lr=self.hyperparameters.learning_rate
            # )

            # self.model_optimizer = torch.optim.Adam(
            #     self.policy.encoder.parameters(), 
            #     lr=self.hyperparameters.model_learning_rate
            # )

        self._move_to_device(default_device())

        self.step = 0


    def _move_to_device(self, device: torch.device) -> None:
        self.target_network.to(device)
    
    def model_loss(
        self,
        obs,
        next_obs,
        actions,
        reward,
        memories,
        sequence_length,
    ):
        encoded_next, _ = self.policy.encoder(
            next_obs,
            None, 
            memories,
            sequence_length
        )
        predict_next, predict_reward = self.policy.model(
            self.policy.encoder,
            obs,
            actions,
            memories=memories,
            sequence_length=sequence_length,
            no_grad_encoder=not self.hyperparameters.transfer_target
        )
        
        loss_fn = torch.nn.MSELoss()

        # print("encoded next", encoded_next)
        # print("pred next", predict_next)
        # print("rew", reward)
        # print("pred rew", predict_reward.squeeze())
        if self.hyperparameters.detach_next:
            encoded_next = encoded_next.detach()
        model_loss = loss_fn(encoded_next, predict_next) + loss_fn(reward, predict_reward.squeeze())

        return model_loss


    def dqn_q_loss(
        self,
        q_out: Dict[str, torch.Tensor],
        target_q: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        rewards: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        q_losses = []
        
        # Multiple q losses per stream
        for i, name in enumerate(q_out.keys()):
            q_stream = q_out[name].squeeze()
            with torch.no_grad():
                q_backup = rewards[name] + (
                    (1.0 - self.use_dones_in_backup[name] * dones)
                    * self.gammas[i]
                    * target_q[name].max(1)[0]
                )
                
            _q_loss = 0.5 * ModelUtils.masked_mean(
                torch.nn.functional.mse_loss(q_backup, q_stream), loss_masks
            )
            q_losses.append(_q_loss)
        q_loss = torch.mean(torch.stack(q_losses))
        
        return q_loss


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
        # self.q_network.network_body.copy_normalization(
        #     self.policy.actor.network_body
        # )
        # self.target_network.network_body.copy_normalization(
        #     self.policy.actor.network_body
        # )

        cont_actions = actions.continuous_tensor

        q_out, _ = self.q_network(
            self.policy.encoder,
            current_obs,
            cont_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
            no_grad_encoder=self.hyperparameters.transfer_target
        )

        if self._action_spec.discrete_size > 0:
            disc_actions = actions.discrete_tensor
            q_stream = self._condense_q_streams(q_out, disc_actions)
        else:
            q_stream = q_out

        with torch.no_grad():
            target_values, _ = self.target_network(
                self.policy.encoder,
                next_obs,
                memories=next_value_memories,
                sequence_length=self.policy.sequence_length,
            )

        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)

        if not self.hyperparameters.transfer_target:
            decay_model_lr = self.decay_model_lr.get_value(self.policy.get_current_step())
            ModelUtils.update_learning_rate(self.model_optimizer, decay_model_lr)

            self.value_optimizer.zero_grad()
            q_loss = self.dqn_q_loss(
                q_stream, target_values, dones, rewards, masks
            )
            q_loss.backward()
            self.value_optimizer.step()

            # update model based on the encoder
            self.model_optimizer.zero_grad()
            model_loss = self.model_loss(
                current_obs, 
                next_obs, 
                actions, 
                rewards["extrinsic"], 
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
            )
            model_loss.backward()
            self.model_optimizer.step()
        
        else:
            # target task, train with model loss
            self.value_optimizer.zero_grad()
            q_loss = self.dqn_q_loss(
                q_stream, target_values, dones, rewards, masks
            )
            model_loss = self.model_loss(
                current_obs, 
                next_obs, 
                actions, 
                rewards["extrinsic"], 
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
            )
            (q_loss + 0.5 * model_loss).backward()
            self.value_optimizer.step()

            # decay_model_lr = self.decay_model_lr.get_value(self.policy.get_current_step())
            # ModelUtils.update_learning_rate(self.model_optimizer, decay_model_lr)
            # torch.autograd.set_detect_anomaly(True)
            # # update encoder based on the model
            # self.model_optimizer.zero_grad()
            # model_loss = self.model_loss(
            #     current_obs, 
            #     next_obs, 
            #     actions, 
            #     rewards["extrinsic"], 
            #     memories=q_memories,
            #     sequence_length=self.policy.sequence_length,
            # )
            # model_loss.backward()
            # self.model_optimizer.step()

            # self.value_optimizer.zero_grad()
            # q_loss = self.dqn_q_loss(
            #     q_stream, target_values, dones, rewards, masks
            # )
            # q_loss.backward()
            # self.value_optimizer.step()

        # Update target network
        ModelUtils.soft_update(self.q_network, self.target_network, self.tau)
        update_stats = {
            "Losses/Value Loss": q_loss.item(),
            "Losses/Model Loss": model_loss.item(),
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats
    
    @timed
    def update_model(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates only dynamics model using buffer.
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

        # update model based on the encoder
        self.model_optimizer.zero_grad()
        model_loss = self.model_loss(
            current_obs, 
            next_obs, 
            actions, 
            rewards["extrinsic"], 
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
        )
        model_loss.backward()
        self.model_optimizer.step()

        # print("model loss", model_loss.item())

        update_stats = {
            "Losses/Model Loss": model_loss.item(),
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
            "Optimizer:q_network": self.q_network,
            "Optimizer:target_network": self.target_network,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
