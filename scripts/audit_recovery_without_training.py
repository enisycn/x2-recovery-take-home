"""Read-only policy/physics diagnostics: one robot, no optimizer or training."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('reports/no_training_audit.json'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
import x2_recovery_isaac
from x2_recovery_isaac import mdp
from x2_recovery_isaac.env_cfg import X2HumanUpRiseEnvCfg

try:
    cfg = X2HumanUpRiseEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.events.reset_back_pose.params['reference_probability_start'] = 0.0
    cfg.events.reset_back_pose.params['reference_probability_end'] = 0.0
    cfg.events.lift_assist = None
    env = gym.make('HRS-X2-Recovery-Play-v0', cfg=cfg)
    task = env.unwrapped
    robot = task.scene['robot']
    sensor = task.scene.sensors['contact_all']
    action_term = task.action_manager.get_term('joint_position')
    checkpoint = torch.load('reports/checkpoints/x2_humanup_selected_model750.pt',
                            map_location=task.device, weights_only=False)
    state = checkpoint['actor_state_dict']
    mean, std = state['obs_normalizer._mean'], state['obs_normalizer._std']
    result = {'training_updates': 0, 'simulated_envs': 1, 'reset_observations': [], 'brake': []}
    for seed in (101, 102, 103, 104, 105):
        observation, _ = env.reset(seed=seed)
        o = observation['policy']
        z = (o - mean) / (std + 0.01)
        result['reset_observations'].append({
            'seed': seed, 'roll_pitch': o[0, 3:5].tolist(),
            'normalized_roll_pitch': z[0, 3:5].tolist(),
            'max_abs_normalized': float(z.abs().max()),
            'history_matches_current': bool(torch.allclose(o[:, 168:].reshape(1, 10, 98), o[:, :98, None].transpose(1, 2).expand(-1, 10, -1))),
        })
    # Load the sensor with real floor forces, then inspect its reset operation
    # directly, before requesting another lazy update from PhysX.
    for _ in range(30):
        with torch.no_grad():
            env.step(torch.zeros((1, 31), device=task.device))
    history = sensor.data.force_matrix_w_history.torch
    before = history.clone()
    sensor.reset(torch.tensor([0], device=task.device))
    after = history.clone()
    result['contact_reset'] = {
        'history_peak_before_n': float(before.norm(dim=-1).max()),
        'history_peak_after_n': float(after.norm(dim=-1).max()),
        'history_identical_after_reset': bool(torch.equal(before, after)),
        'current_force_peak_after_n': float(sensor._data.force_matrix_w.torch.norm(dim=-1).max()),
    }
    # Teleport only to evaluate the mathematical action map; no rollout or
    # success claim is made from these artificially imposed poses.
    for height in (0.50, 0.55, 0.58, 0.60, 0.62, 0.68):
        pose = robot.data.root_pose_w.torch.clone()
        pose[:, 2] = height
        pose[:, 3:7] = torch.tensor([0., 0., 0., 1.], device=task.device)
        robot.write_root_pose_to_sim_index(root_pose=pose)
        mdp._invalidate_root_derived_buffers(robot)
        action_term.process_actions(torch.ones((1, 31), device=task.device))
        result['brake'].append({'height_m': height,
            'max_processed_delta_before_joint_clip_rad': float(action_term.processed_actions.abs().max())})
    # Exact supine is an Euler singularity; compare nearby orientations.
    euler_examples = []
    for offset in (-0.001, 0.001):
        zero = torch.zeros(1, device=task.device)
        q = quat_from_euler_xyz(zero, zero - torch.pi/2 + offset, zero)
        r, p, y = euler_xyz_from_quat(q)
        euler_examples.append({'pitch_offset_rad': offset, 'quaternion_xyzw': q[0].tolist(),
                               'roll_pitch_yaw_rad': [float(r[0]), float(p[0]), float(y[0])]})
    result['supine_euler_singularity'] = euler_examples
    result['joint_names'] = list(robot.joint_names)
    result['effort_limits_nm'] = robot.data.joint_effort_limits.torch[0].tolist()
    result['normalizer_count'] = int(state['obs_normalizer.count'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    env.close()
finally:
    launcher.app.close()
