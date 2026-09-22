# Validation — selected relaxed_v4 policy

22 September 2026. Actual Isaac Lab/PhysX evaluation and ROS integration; CPU harness results are separate historical checks.

## Five supine episodes

Checkpoint: `reports/checkpoints/x2_relaxed_v4_model450.pt`. All final training, evaluation and ROS resets are true supine, with no external lift or reference starts. Each evaluation runs the full 10 seconds and reads the last state before automatic reset. Both feet must support the robot without any other ground support. Full thresholds and states are in [evaluation](relaxed_v4_evaluation.json).

| Seed | Recovery | Strict stance through end (s) | Relaxed stance through end (s) | Final 2 s satisfy both |
| --- | --- | ---: | ---: | --- |
| 101 | PASS | 8.94 | 8.86 | 100% |
| 102 | PASS | 8.54 | 8.54 | 100% |
| 103 | PASS | 8.64 | 8.18 | 100% |
| 104 | PASS | 8.74 | 8.48 | 100% |
| 105 | PASS | 8.88 | 7.78 | 100% |

The old v3 policy held its shoulders forward near -1.918 rad. The selected policy places the arms near the torso with slightly bent elbows. The extra posture criterion is each shoulder error <=0.30 rad from zero and each elbow error <=0.30 rad from -0.15 rad, together with strict stance. A directly rendered [GIF](gifs_relaxed_v4/x2_final_policy_attempt.gif) records this same checkpoint, seed 101: 250 frames, 40 ms/frame, 640×360. No scripted arm override is used.

## ROS build and runtime

Fresh build command (new empty build/install directories):

```bash
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 /usr/bin/colcon --log-base tmp/ros_fresh_log build \
  --symlink-install --base-paths src --build-base tmp/ros_fresh_build \
  --install-base tmp/ros_fresh_install --event-handlers console_direct+
```

One package built successfully in 1.03 s: [full build log](ros_fresh_build_v4.txt). The final integration command used that install:

```bash
HRS_ROS_INSTALL="$PWD/tmp/ros_fresh_install" \
  ./scripts/validate_ros_isaac_runtime.sh reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0
```

The recorded run used the byte-identical export before it was copied from `tmp/v4_supine450/exported_relaxed_v4/policy.pt` into its delivery path.

Acceptance latency was 0.001080 s; immediate second request was rejected. The concurrency trial received 85 simulator joint samples and IDLE → RUNNING → SUCCEEDED. A literal `ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'` received 100 samples and SUCCEEDED. `ros2 param set /x2_recovery timeout_sec 0.2` then forced a genuine Isaac episode to time out after 10 policy steps: RUNNING → FAILED. The telemetry node logged `status=RUNNING left_knee_joint=... rad`. [Complete runtime record](ros_isaac_relaxed_v4_validation.txt).

## Other checks and limitations

- 29 formula/contact/ROS unit tests passed. The new posture-reward check excludes supine/missing-support states and allows arm adjustment before the strict low-speed condition holds.
- TorchScript export matches the trained normalized actor exactly on its verification probe (maximum error 0.0).
- Training source revisions, environment/agent YAML, hashes and parent checkpoints are included under `configs/` and `checkpoints/`. Earlier mixed-reset exploratory pretraining is disclosed; only the final task experiment is entirely supine.
- The reward curve covers the selected final-stage prefix 400–450; previous stages and unsuccessful posture trials are documented in [method/trials](../docs/relaxed_v4.md).
- HRS used the existing Isaac Python and ROS system Python separately, nice 15, CPU set `<HOST_CPUSET>`. No robot-project, package, driver or CPU-isolation changes were made.
- Nominal simulation only; these five starts do not establish hardware safety, arbitrary-posture recovery or sim-to-real robustness.
- [PDF audit](../docs/pdf_compliance_audit.md): GitHub repository/publication remains pending. Genuine local development history is preserved; earlier ongoing pushes cannot be claimed retroactively.

Historical v3 results remain in `symmetric_v3_evaluation.json`; failed posture trials are in `arm_posture_trials/`. They are not mixed into the selected 5/5 result.
