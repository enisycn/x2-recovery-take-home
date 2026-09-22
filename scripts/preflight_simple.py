"""Short live PhysX checks before any v2 PPO run; no optimizer updates."""
import argparse, json, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--environment', choices=('simple_v2','symmetric_v3'), default='simple_v2')
parser.add_argument('--dt', type=float, default=.005)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import torch, gymnasium as gym
import x2_recovery_isaac
from x2_recovery_isaac import mdp
from x2_recovery_isaac.simple_cfg import X2SimpleRecoveryEnvCfg, X2SymmetricRecoveryEnvCfg
from x2_recovery_isaac.env_cfg import foot_contact_cfg, all_contact_cfg

def main():
    cfg=X2SymmetricRecoveryEnvCfg() if args.environment=="symmetric_v3" else X2SimpleRecoveryEnvCfg(); cfg.scene.num_envs=2; cfg.sim.device=args.device
    cfg.sim.dt=args.dt; cfg.decimation=round(.02/args.dt); cfg.episode_length_s=3.
    p=cfg.events.reset_back_pose.params
    p['reference_probability_start']=p['reference_probability_end']=0.
    env=gym.make('HRS-X2-Recovery-v0',cfg=cfg); task=env.unwrapped
    try:
        obs,_=env.reset(seed=42); robot=task.scene['robot']; sensor=task.scene.sensors['contact_all']
        initial=robot.data.projected_gravity_b.torch.clone()
        action_width=8 if args.environment=='symmetric_v3' else 31
        obs_width=106+2*action_width
        assert obs['policy'].shape==(2,obs_width) and torch.isfinite(obs['policy']).all()
        assert (initial[:,:2].norm(dim=-1)>.99).all()
        ids=torch.tensor([1],device=task.device,dtype=torch.int32)
        pose=robot.data.default_root_pose.torch[ids].clone();pose[:,:3]=task.scene.env_origins[ids];pose[:,2]+=.675
        pose[:,3:]=torch.tensor([0.,0.,0.,1.],device=task.device)
        robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        q=torch.zeros((1,31),device=task.device)
        lim=robot.data.soft_joint_pos_limits.torch[ids]
        q=q.clamp(min=lim[...,0],max=lim[...,1])
        robot.write_joint_state_to_sim_index(position=q,velocity=torch.zeros_like(q),env_ids=ids)
        mdp._invalidate_root_derived_buffers(robot)
        feet=foot_contact_cfg();allb=all_contact_cfg();feet.resolve(task.scene);allb.resolve(task.scene)
        stable=best=0; maxheight=0.; count=0
        task.terminal_observer=lambda t,e: {'ids':e.tolist(),'height':robot.data.root_pos_w.torch[e,2].tolist(),'length':t.episode_length_buf[e].tolist()}
        # Absolute zero target (clipped to soft limits) must continue to hold.
        a=torch.zeros((2,action_width),device=task.device)
        if action_width==8:
            from x2_recovery_isaac.synergy_action import GROUPS
            a[:]=torch.atanh(torch.tensor([-centre/span for _,centre,span in GROUPS],device=task.device).clamp(-.999,.999))
        for step in range(150):
            obs,r,terminated,truncated,_=env.step(a)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(r).all()
            if terminated.any() or truncated.any(): break
            strict=bool(mdp.strict_success(task,feet_cfg=feet,all_bodies_cfg=allb)[1])
            stable=stable+1 if strict else 0;best=max(best,stable);count+=1
            maxheight=max(maxheight,float(robot.data.root_pos_w.torch[0,2]))
            if step==25:
                before=sensor.data.force_matrix_w_history.torch.clone()
                assert before[1].abs().max()>0
                sensor.reset(env_ids=torch.tensor([0],device=task.device))
                assert sensor._data.force_matrix_w_history.torch[0].abs().max()==0
                assert torch.equal(sensor._data.force_matrix_w_history.torch[1],before[1])
                partial_reset=True
        # Control target stays sensitive to commands at standing height.
        term=task.action_manager.get_term('joint_position')
        term.process_actions(a);zero=term.processed_actions.clone()
        term.process_actions(a+.1); response=(term.processed_actions-zero).abs().max().item()
        report={'dt':args.dt,'observation_width':obs['policy'].shape[1],'all_finite':True,
                'partial_contact_history_reset':partial_reset,'supine_initial_gravity':initial.tolist(),
                'standing_strict_hold_s':best*.02,'supine_zero_action_max_height':maxheight,
                'action_response_rad':response,'terminal_capture':task.terminal_snapshot,
                'passed':best*.02>=.5 and response>.07 and task.terminal_snapshot is not None}
        path=Path(f'reports/{args.environment}_preflight_{args.dt}.json');path.write_text(json.dumps(report,indent=2)+'\n')
        print('PREFLIGHT',json.dumps(report),flush=True)
        assert report['passed']
    finally: env.close()
try:main()
except BaseException:traceback.print_exc();raise
finally:app.close()
