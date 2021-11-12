from typing import Dict, cast
from mlagents.torch_utils import torch, default_device
import math
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil

from mlagents_envs.timers import timed
from mlagents.trainers.policy.transfer_policy import TransferPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import TrainerSettings, PPOTransferSettings
from mlagents.trainers.torch.networks import ValueNetwork, EncodedValueNetwork
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs
from mlagents.trainers.torch.utils import ModelUtils, MovingMeanStd
from mlagents.trainers.trajectory import ObsUtil


class TorchPPOTransferOptimizer(TorchOptimizer):
    def __init__(self, policy: TransferPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TorchPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        hyperparameters: PPOTransferSettings = cast(PPOTransferSettings, trainer_settings.hyperparameters)
        self.hyperparameters = hyperparameters

        if policy.shared_critic:
            self._critic = policy.actor
        else:
            if self.hyperparameters.encode_critic:
                self._critic = EncodedValueNetwork(
                    reward_signal_names,
                    policy.behavior_spec.observation_specs,
                    policy.network_settings,
                    feature_size = hyperparameters.feature_size,
                    norm_latent = hyperparameters.norm_latent
                )
            else:
                self._critic = ValueNetwork(
                    reward_signal_names,
                    policy.behavior_spec.observation_specs,
                    network_settings=trainer_settings.network_settings,
                )
            self._critic.to(default_device())
        
        print("---actor----")
        print(self.policy.actor)
        print("---critic----")
        print(self._critic)

        params = list(self.policy.actor.parameters()) + list(self._critic.parameters())
        model_params = list(self.policy.model.parameters())
        
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.decay_model_learning_rate = ModelUtils.DecayedValue(
            hyperparameters.model_lr_schedule,
            hyperparameters.model_learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_coeff = ModelUtils.DecayedValue(
            hyperparameters.model_lr_schedule,
            hyperparameters.coeff,
            0,
            self.trainer_settings.max_steps,
        )

        if self.hyperparameters.auxiliary:
            self.optimizer = torch.optim.Adam(
                params+model_params, lr=self.trainer_settings.hyperparameters.learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(
                params, lr=self.trainer_settings.hyperparameters.learning_rate
            )
            self.model_optimizer = torch.optim.Adam(
                model_params, lr=hyperparameters.model_learning_rate
            ) 

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
            "Losses/Model Loss": "model_loss",
        }

        self.stream_names = list(self.reward_signals.keys())

        if self.hyperparameters.norm_reward:
            self.reward_ma = MovingMeanStd(self.hyperparameters.batch_size, default_device())

    @property
    def critic(self):
        return self._critic

    def ppo_value_loss(
        self,
        values: Dict[str, torch.Tensor],
        old_values: Dict[str, torch.Tensor],
        returns: Dict[str, torch.Tensor],
        epsilon: float,
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates value loss for PPO.
        :param values: Value output of the current network.
        :param old_values: Value stored with experiences in buffer.
        :param returns: Computed returns.
        :param epsilon: Clipping value for value estimate.
        :param loss_mask: Mask for losses. Used with LSTM to ignore 0'ed out experiences.
        """
        value_losses = []
        for name, head in values.items():
            old_val_tensor = old_values[name]
            returns_tensor = returns[name]
            clipped_value_estimate = old_val_tensor + torch.clamp(
                head - old_val_tensor, -1 * epsilon, epsilon
            )
            v_opt_a = (returns_tensor - head) ** 2
            v_opt_b = (returns_tensor - clipped_value_estimate) ** 2
            value_loss = ModelUtils.masked_mean(torch.max(v_opt_a, v_opt_b), loss_masks)
            value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    def ppo_policy_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate PPO policy loss.
        :param advantages: Computed advantages.
        :param log_probs: Current policy probabilities
        :param old_log_probs: Past policy probabilities
        :param loss_masks: Mask for losses. Used with LSTM to ignore 0'ed out experiences.
        """
        advantage = advantages.unsqueeze(-1)

        decay_epsilon = self.hyperparameters.epsilon
        r_theta = torch.exp(log_probs - old_log_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = (
            torch.clamp(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * advantage
        )
        policy_loss = -1 * ModelUtils.masked_mean(
            torch.min(p_opt_a, p_opt_b), loss_masks
        )
        return policy_loss
    
    def ppo_model_loss(
        self,
        encoder,
        obs,
        next_obs,
        actions,
        reward,
        memories,
        sequence_length,
        detach_next
    )-> torch.Tensor:
        loss_fn = torch.nn.MSELoss()
        if self.hyperparameters.model_raw:
            predict_next, predict_reward = self.policy.model.raw_forward(
                obs,
                actions,
                no_grad_encoder = not self.hyperparameters.transfer_target
            )
            encoded_next = torch.cat(next_obs, dim=1)
            return loss_fn(encoded_next, predict_next) + loss_fn(reward, predict_reward.squeeze())
        
        encoded_next, _ = encoder(
            next_obs,
            None, 
            memories,
            sequence_length
        )
        grad_encoder = self.hyperparameters.transfer_target or self.hyperparameters.auxiliary
        predict_next, predict_reward = self.policy.model(
            encoder,
            obs,
            actions,
            memories,
            sequence_length,
            no_grad_encoder = not grad_encoder
        )
        
        if detach_next:
            encoded_next = encoded_next.detach()
        
        if self.hyperparameters.norm_reward:
            reward = torch.div(reward - self.reward_ma.mean(), self.reward_ma.std())
            self.reward_ma.push(reward)
        
        if self.hyperparameters.predict_delta:
            encoded_cur, _ = encoder(
                obs,
                None, 
                memories,
                sequence_length
            )
            encoded_cur = encoded_cur.detach()
            model_loss = loss_fn(encoded_next-encoded_cur, predict_next) + loss_fn(reward, predict_reward.squeeze())
        else:
            model_loss = loss_fn(encoded_next, predict_next) + loss_fn(reward, predict_reward.squeeze())
            # model_loss = loss_fn(reward, predict_reward.squeeze())

        return model_loss
    
    def model_loss_batch(self, batch: AgentBuffer):
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        # Convert to tensors
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )
        if self.hyperparameters.encode_critic:
            model_loss = self.ppo_model_loss(
                self.critic.encoder,
                current_obs, 
                next_obs, 
                actions, 
                rewards["extrinsic"], 
                memories=None,  # does not support memory for now
                sequence_length=self.policy.sequence_length,
                detach_next=self.hyperparameters.detach_next
            )
        else:
            model_loss = 0
        if self.hyperparameters.encode_actor:
            actor_model_loss = self.ppo_model_loss(
                self.policy.actor.encoder,
                current_obs, 
                next_obs, 
                actions, 
                rewards["extrinsic"], 
                memories=None,  # does not support memory for now
                sequence_length=self.policy.sequence_length,
                detach_next=self.hyperparameters.detach_next
            )
            model_loss += actor_model_loss
        return model_loss
    
    def update_pretrain(self, op_batch):
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())

        model_loss = self.model_loss_batch(op_batch)

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        model_loss.backward()
        self.optimizer.step()

        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Model Loss": model_loss.item() if type(model_loss) is not int else model_loss,
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int, op_batch: AgentBuffer) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        if self.hyperparameters.random_policy:
            return self.update_pretrain(op_batch)
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_model_lr = self.decay_model_learning_rate.get_value(self.policy.get_current_step())
        coeff = self.decay_coeff.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        # Get value memories
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]
        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        log_probs, entropy = self.policy.evaluate_actions(
            current_obs,
            masks=act_masks,
            actions=actions,
            memories=memories,
            seq_len=self.policy.sequence_length,
        )
        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        value_loss = self.ppo_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        policy_loss = self.ppo_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
        )
        
        model_loss = self.model_loss_batch(op_batch)

        if not self.hyperparameters.transfer_target and not self.hyperparameters.auxiliary:
            loss = (
                policy_loss
                + 0.5 * value_loss
                - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
            )

            # Set optimizer learning rate
            ModelUtils.update_learning_rate(self.optimizer, decay_lr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.hyperparameters.encode_critic or self.hyperparameters.encode_actor:
                ModelUtils.update_learning_rate(self.model_optimizer, decay_model_lr)
                self.model_optimizer.zero_grad()
                model_loss.backward()
                self.model_optimizer.step()

        else:
            loss = (
                policy_loss
                + 0.5 * value_loss
                + coeff * model_loss
                - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
            )

            # Set optimizer learning rate
            ModelUtils.update_learning_rate(self.optimizer, decay_lr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Model Loss": model_loss.item() if type(model_loss) is not int else model_loss,
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))

        return update_stats

    def get_modules(self):
        modules = {"Optimizer": self.optimizer}
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
