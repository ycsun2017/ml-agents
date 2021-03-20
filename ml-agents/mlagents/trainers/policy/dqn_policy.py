from typing import Any, Dict, List, Tuple, Optional, cast
import numpy as np
from mlagents.torch_utils import torch, default_device
import copy
import random

from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.behavior_id_utils import get_global_agent_id
from mlagents.trainers.policy import Policy
from mlagents_envs.base_env import DecisionSteps, BehaviorSpec
from mlagents_envs.timers import timed

from mlagents.trainers.settings import TrainerSettings, ScheduleType, DQNSettings
from mlagents.trainers.torch.networks import SimpleActor, SharedActorCritic, GlobalSteps, LatentEncoder, EncodedQNetwork, DynamicModel
from mlagents.trainers.torch.decoders import ValueHeads
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs

from mlagents.trainers.torch.components.reward_providers import create_reward_provider

EPSILON = 1e-7  # Small value to avoid divide by zero


class DQNPolicy(Policy):
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        trainer_settings: TrainerSettings,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        separate_critic: bool = True,
        condition_sigma_on_obs: bool = True,
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous actions, as well as recurrent networks.
        :param seed: Random seed.
        :param behavior_spec: Assigned BehaviorSpec object.
        :param trainer_settings: Defined training parameters.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        :param tanh_squash: Whether to use a tanh function on the continuous output,
        or a clipped output.
        :param reparameterize: Whether we are using the resampling trick to update the policy
        in continuous output.
        """
        super().__init__(
            seed,
            behavior_spec,
            trainer_settings,
            tanh_squash,
            reparameterize,
            condition_sigma_on_obs,
        )
        self.global_step = (
            GlobalSteps()
        )  # could be much simpler if TorchPolicy is nn.Module
        self.grads = None

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
        }

        self.reward_signals = {}
        self.create_reward_signals(trainer_settings.reward_signals)
        self.stream_names = list(self.reward_signals.keys())
        self.num_actions = sum(behavior_spec.action_spec.discrete_branches)

        self.hyperparameters: DQNSettings = cast(
            DQNSettings, trainer_settings.hyperparameters
        )

        # The encoder
        self.encoder = LatentEncoder(
            behavior_spec.observation_specs, 
            trainer_settings.network_settings,
            int(behavior_spec.action_spec.continuous_size),
            self.hyperparameters.feature_size
        )

        # The Q net based on the encoder
        self.q_network = EncodedQNetwork(
            # self.encoder,
            self.stream_names,
            behavior_spec.observation_specs,
            trainer_settings.network_settings,
            behavior_spec.action_spec,
            self.hyperparameters.feature_size
        )

        # The dynamics model
        self.model = DynamicModel(
            self.hyperparameters.feature_size, 
            self.num_actions,
            trainer_settings.network_settings.hidden_units,
            self.hyperparameters.forward_layers, 
        )

        # self.q_network = ValueNetwork(
        #     self.stream_names,
        #     behavior_spec.observation_specs,
        #     trainer_settings.network_settings,
        #     int(behavior_spec.action_spec.continuous_size),
        #     self.num_actions
        # )
        print("encoder\n", self.encoder)
        print("policy q net\n", self.q_network)
        print("dynamics net\n", self.model)

        self.shared_critic = False

        # Save the m_size needed for export
        self._export_m_size = self.m_size
        # m_size needed for training is determined by network, not trainer settings
        self.m_size = 0

        self.decay_epsilon = ModelUtils.DecayedValue(
            ScheduleType.LINEAR,
            1.0,
            0.05,
            self.trainer_settings.max_steps,
        )

        self.q_network.to(default_device())
        self._clip_action = not tanh_squash
    
    def create_reward_signals(self, reward_signal_configs):
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        for reward_signal, settings in reward_signal_configs.items():
            # Name reward signals by string in case we have duplicates later
            self.reward_signals[reward_signal.value] = create_reward_provider(
                reward_signal, self.behavior_spec, settings
            )

    @property
    def export_memory_size(self) -> int:
        """
        Returns the memory size of the exported ONNX policy. This only includes the memory
        of the Actor and not any auxillary networks.
        """
        return self._export_m_size

    def _extract_masks(self, decision_requests: DecisionSteps) -> np.ndarray:
        mask = None
        if self.behavior_spec.action_spec.discrete_size > 0:
            num_discrete_flat = np.sum(self.behavior_spec.action_spec.discrete_branches)
            mask = torch.ones([len(decision_requests), num_discrete_flat])
            if decision_requests.action_mask is not None:
                mask = torch.as_tensor(
                    1 - np.concatenate(decision_requests.action_mask, axis=1)
                )
        return mask

    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        If this policy normalizes vector observations, this will update the norm values in the graph.
        :param buffer: The buffer with the observations to add to the running estimate
        of the distribution.
        """
        pass
        # if self.normalize:
        #     self.actor.update_normalization(buffer)

    def sample_actions(self, obs) -> AgentAction:
        """
        :return: An AgentAction corresponding to the actions sampled from the DQN agent
        """
        decay_eps = self.decay_epsilon.get_value(self.get_current_step())
        # if self.get_current_step() % 500 == 0:
        #     print("eps", decay_eps)
        continuous_action: Optional[torch.Tensor] = None
        discrete_action: Optional[List[torch.Tensor]] = None
        tensor_obs = [torch.as_tensor(np_ob) for np_ob in obs]
        
        discrete_action = []
        q_out = self.q_network(
            self.encoder,
            tensor_obs,
            None,
            memories=None,
            sequence_length=self.sequence_length,
        )
        
        discrete_action = []
        discrete_action.append(q_out[0]["extrinsic"].max(1,keepdim=True)[1].detach())

        random_action = [torch.randint_like(dis_a, 0, self.num_actions) for dis_a in discrete_action]
        
        if random.random() < decay_eps:
            discrete_action = random_action

        action = AgentAction(continuous_action, discrete_action)

        run_out = {}
        action_tuple = action.to_action_tuple()
        run_out["action"] = action_tuple
        # This is the clipped action which is not saved to the buffer
        # but is exclusively sent to the environment.
        env_action_tuple = action.to_action_tuple(clip=self._clip_action)
        run_out["env_action"] = env_action_tuple
        run_out["learning_rate"] = 0.0
        # run_out["log_probs"] = log_probs.to_log_probs_tuple()
        # run_out["entropy"] = ModelUtils.to_numpy(entropy)
        if self.use_recurrent:
            run_out["memory_out"] = ModelUtils.to_numpy(memories).squeeze(0)
        return run_out

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param worker_id:
        :param decision_requests: A dictionary of behavior names and DecisionSteps from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(decision_requests) == 0:
            return ActionInfo.empty()

        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]  # For 1-D array, the iterator order is correct.
        obs = decision_requests.obs
        run_out = self.sample_actions(obs)
        self.check_nan_action(run_out.get("action"))
        # print(run_out.get("action").discrete)
        return ActionInfo(
            action=run_out.get("action"),
            env_action=run_out.get("env_action"),
            outputs=run_out,
            agent_ids=list(decision_requests.agent_id),
        )

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        return self.global_step.current_step

    def set_step(self, step: int) -> int:
        """
        Sets current model step to step without creating additional ops.
        :param step: Step to set the current model step to.
        :return: The step the model was set to.
        """
        self.global_step.current_step = step
        return step

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        self.global_step.increment(n_steps)
        return self.get_current_step()

    def load_weights(self, values: List[np.ndarray]) -> None:
        self.actor.load_state_dict(values)

    def init_load_weights(self) -> None:
        pass

    def get_weights(self) -> List[np.ndarray]:
        return copy.deepcopy(self.actor.state_dict())

    def get_modules(self):
        return {"Policy": self.actor, "global_step": self.global_step}
