"""Default configuration presets."""

import torch
from MAPPO.config.config import ExperimentConfig, SumoConfig, NetworkConfig, MAPPOConfig, TrainingConfig, LoggingConfig


def _best_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_config() -> ExperimentConfig:
    """Get default configuration for cologne3."""
    return ExperimentConfig(
        sumo=SumoConfig(
                net_file="RESCO/cologne3/cologne3.net.xml",
                route_file="RESCO/cologne3/cologne3.rou.xml",
                # cologne3 vehicles depart 23512s–28798s; begin_time skips the empty
                # warm-up without touching the route file, preserving benchmark validity.
                begin_time=23400,   # 100s before first departure (23512s)
                num_seconds=5600,   # episode runs 23400→29000s, past last departure (28798s)
                use_gui=False,
                delta_time=5
            ),
        network=NetworkConfig(
            actor_hidden=[128, 128],
            critic_hidden=[256, 256],
            activation="relu",
            use_orthogonal_init=True
        ),
        mappo=MAPPOConfig(
            # lr_actor=1e-4.  A 3e-5 trial (run mappo_1778046430) caused
            # clip_frac to collapse toward 0 (mean 0.00245 vs 0.00905),
            # leaving the policy unable to escape the gridlock attractor for
            # 9 epochs.  1e-4 is the minimum rate needed to escape within 20
            # epochs while staying below the bimodal boundary-crossing threshold.
            lr_actor=1e-4,
            # lr_critic halved 1e-3 → 5e-4 to compensate for repeat_per_collect
            # doubling (2 → 4).  Twice as many gradient steps per epoch at the
            # same lr would double the effective per-epoch critic update magnitude,
            # making oscillation worse.  5e-4 keeps the per-epoch critic step size
            # the same as the original (repeat=2, lr=1e-3) baseline.
            lr_critic=5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            eps_clip=0.2,
            # value_clip re-enabled now that reward_normalization=True keeps
            # critic targets O(1).  Previously had to disable it because
            # unnormalized rewards made critic values O(100+), causing the
            # clamp range (±eps_clip=0.2) to saturate immediately and zero
            # all critic gradients.
            value_clip=True,
            # max_grad_norm lowered 0.5 → 0.3 as a second line of defence
            # against catastrophic policy jumps caused by noisy advantage
            # estimates (critic_loss 3–10 → RMS error 1.7–3.2 per step).
            max_grad_norm=0.3,
            # Normalize per-step rewards to zero-mean unit-variance using a
            # shared Welford running normalizer (clip ±10).  This reduces
            # critic targets from O(reward/(1-gamma)) ≈ O(100) to O(1),
            # which is the root-cause fix for the high critic loss (3–10)
            # and resulting inaccurate advantage estimates that caused
            # catastrophic policy flips between the cologne3 attractors.
            reward_normalization=True,
        ),
        training=TrainingConfig(
            max_epoch=100,
            step_per_epoch=10000,
            # episode_per_collect raised 30 → 50 (Fix 4) so the batch gradient
            # averages over more bimodal traffic realizations per update, giving
            # a more stable policy-gradient direction and reducing the chance of
            # the actor stepping toward gridlock due to an all-clear batch.
            episode_per_collect=50,
            batch_size=256,
            # repeat_per_collect raised 2 → 4: more gradient steps per collected
            # batch drive the critic closer to convergence each epoch, reducing
            # the epoch-to-epoch oscillation in critic_loss (mean |Δloss| 2.98
            # across both prior runs with repeat=2).  At episode_per_collect=50
            # this gives 50×4=200 mini-batch pass-throughs per epoch vs the
            # original 30×4=120, providing strictly more critic fitting without
            # additional environment interaction.
            repeat_per_collect=4,
            n_train_envs=4,
            # n_test_envs raised 5 → 20 (Fix 1): with only 5 fixed-seed eval
            # episodes a single seed flipping attractor state moves the reported
            # mean reward by ~275 points.  20 seeds reduces that sensitivity by
            # 2× (variance ∝ 1/√N), making the eval curve reflect true policy
            # quality rather than which seeds happened to be in gridlock.
            n_test_envs=20
        ),
        logging=LoggingConfig(
            project="sumo-mappo-traffic",
            group="cologne3",
            use_wandb=False  # Disabled by default - use --wandb flag or set to True to enable
        ),
        seed=42,
        device=_best_device()
    )


def get_debug_config() -> ExperimentConfig:
    """Get debug configuration (faster, headless by default)."""
    config = get_default_config()
    config.sumo.use_gui = False  # Changed to False - GUI requires X11/XQuartz on macOS
    # begin_time and num_seconds inherited from get_default_config() (23400 / 5600)
    config.training.max_epoch = 5
    config.training.step_per_epoch = 1000
    config.training.n_train_envs = 1
    config.training.n_test_envs = 1
    config.logging.use_wandb = False
    return config


def get_fast_test_config() -> ExperimentConfig:
    """Get configuration for fast testing (no GUI, shorter but with vehicles)."""
    config = get_default_config()
    # begin_time=23400, num_seconds=5600 inherited from get_default_config()
    config.training.max_epoch = 20
    config.training.step_per_epoch = 2000
    config.training.n_train_envs = 4
    config.training.test_interval = 1
    config.training.save_interval = 1
    config.logging.use_wandb = False
    return config
