from __future__ import annotations

from dataclasses import MISSING, field
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"
    actor_obs_normalization: bool = MISSING
    critic_obs_normalization: bool = MISSING
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    activation: str = MISSING


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticRecurrent"
    rnn_type: str = MISSING
    rnn_hidden_dim: int = MISSING
    rnn_num_layers: int = MISSING


@configclass
class RslRlPpoAlgorithmCfg:
    class_name: str = "PPO"
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    entropy_coef: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING
    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    normalize_advantage_per_mini_batch: bool = False
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None


@configclass
class RslRlOnPolicyRunnerCfg:
    class_name: str = "OnPolicyRunner"
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool | None = None
    obs_groups: dict[str, list[str]] = field(default_factory=dict)
    clip_actions: float | None = None
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    policy: RslRlPpoActorCriticCfg = MISSING
    algorithm: RslRlPpoAlgorithmCfg = MISSING


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = ""  # same as task name
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DecoderSingleCfg:
    type: Literal["mlp", "vae"] = MISSING
    target_indices: list[int] = MISSING
    loss_weight: float = 1.0
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    activation: str = "elu"
    latent_dim: int = 32


@configclass
class VAEDecoderCfg(DecoderSingleCfg):
    type: Literal["vae"] = "vae"
    kl_weight: float = 1e-3


@configclass
class DecoderCollectionCfg:
    learning_rate: float = 1e-4
    loss_weight: float = 0.01
    configs: list[DecoderSingleCfg] = MISSING


@configclass
class ActorCriticDecoderCfg:
    actor: DecoderCollectionCfg | None = None
    critic: DecoderCollectionCfg | None = None


@configclass
class RecurrentDecoderPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    decoder_cfg: ActorCriticDecoderCfg | None = None


@configclass
class RecurrentDecoderPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = ""
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=512,
        rnn_num_layers=1,
    )
    algorithm = RecurrentDecoderPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        decoder_cfg=ActorCriticDecoderCfg(
            actor=DecoderCollectionCfg(
                learning_rate=1.0e-4,
                loss_weight=0.01,
                configs=[
                    VAEDecoderCfg(
                        target_indices=[0, 1, 2],
                        loss_weight=1.0,
                        hidden_dims=[128, 64],
                        activation="elu",
                        latent_dim=32,
                        kl_weight=1.0e-3,
                    )
                ],
            ),
            critic=DecoderCollectionCfg(
                learning_rate=1.0e-4,
                loss_weight=0.01,
                configs=[
                    VAEDecoderCfg(
                        target_indices=[0, 1, 2],
                        loss_weight=1.0,
                        hidden_dims=[128, 64],
                        activation="elu",
                        latent_dim=32,
                        kl_weight=1.0e-3,
                    )
                ],
            ),
        ),
    )


@configclass
class Go2WFlatRnnDecoderPPORunnerCfg(RecurrentDecoderPPORunnerCfg):
    def __post_init__(self) -> None:
        self.experiment_name = "omnibot_go2w_flat_rnn_v0"
        self.algorithm.desired_kl = 0.01


@configclass
class Go2WRoughRnnDecoderPPORunnerCfg(RecurrentDecoderPPORunnerCfg):
    def __post_init__(self) -> None:
        self.experiment_name = "omnibot_go2w_rough_rnn_v0"
        self.algorithm.desired_kl = 0.01
