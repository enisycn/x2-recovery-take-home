"""Compact RSL-RL PPO configuration for the X2 recovery task."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class X2RecoveryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 3000 envs x 12 steps is already a 36k-transition PPO batch.  Shorter
    # rollouts update the actor 2.67x more often and keep 400 iterations near
    # the expected desktop-GPU runtime without reducing environment diversity.
    num_steps_per_env = 12
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hrs_x2_recovery"
    seed = 42
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HumanUpHistoryActorCfg(RslRlMLPModelCfg):
    class_name: str = "x2_recovery_isaac.agents.humanup_history_model:HumanUpHistoryActor"
    proprio_dim: int = 98
    history_length: int = 10
    history_latent_dim: int = 20


@configclass
class X2HumanUpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """HumanUP code-release PPO and neural-network settings for X2."""

    num_steps_per_env = 24
    max_iterations = 50_001
    save_interval = 50
    experiment_name = "hrs_x2_humanup"
    seed = 1
    actor = HumanUpHistoryActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class X2HumanUpCurriculumPPORunnerCfg(X2HumanUpPPORunnerCfg):
    """Lower-noise actor for the explicit X2 balance/rise teacher phases."""

    experiment_name = "hrs_x2_humanup_curriculum"
    actor = HumanUpHistoryActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
