# Test matrix

`29 passed` refers to the repository's fast pytest regression suite. It is supporting evidence for formulas, state handling and ROS control flow; it is not the five-episode Isaac performance result required by the assignment.

## What the 29 tests cover

| Group | Count | What is checked | Test file |
| --- | ---: | --- | --- |
| Frame and geometry | 3 | XYZW supine quaternion, collision-derived reset/standing clearances, and floor coverage for the parallel environment grid. | `isaaclab_ext/test/test_reward_formulas.py` |
| Reward, contact, control and safety | 18 | Height/upright formulas, inverted-pose rejection, strict two-foot success, other-body support rejection, self-collision filtering, current versus stale contact, arm reward gating, action/joint clipping, soft limits, cache invalidation, unsafe termination and assistance schedules. | `isaaclab_ext/test/test_reward_formulas.py` |
| Reduced-order ROS harness | 4 | Supine reset, action/joint limits, scripted success predicate and zero-action timeout without false success. | `src/x2_recovery_ros/test/test_reduced_env.py` |
| Session and IPC behavior | 4 | Atomic busy gate, state publication with success, timeout to `FAILED`, and forwarding of joint state/result over the local Unix socket protocol. | `src/x2_recovery_ros/test/test_session.py` |
| **Total** | **29** | | |

Run them with:

```bash
export ISAAC_PYTHON=/path/to/isaac/python
export PYTHONPATH="$PWD/isaaclab_ext:$PWD/src/x2_recovery_ros"
"$ISAAC_PYTHON" -m pytest -q \
  isaaclab_ext/test/test_reward_formulas.py \
  src/x2_recovery_ros/test
```

The recorded result is `29 passed, 37 warnings`. The warnings are two Isaac Lab API deprecations and 35 PyTorch TorchScript deprecations; there is no failed or skipped test in the recorded run. See `reports/unit_tests_v4.txt`.

## Assignment validation is separate

The HRS task asks for system-level evidence that cannot be established by these fast tests alone:

| Assignment check | Evidence |
| --- | --- |
| Five simulator recovery episodes | `reports/relaxed_v4_evaluation.json`: seeds 101-105, 5/5 in real Isaac/PhysX. |
| Upright, both feet, no other support | The evaluator measures height, projected gravity, base velocities and ground-filtered contact for all 32 bodies over every episode. |
| Fresh ROS 2 build | `reports/ros_fresh_build_v4.txt`. |
| One launch, CLI Trigger request and live JointState | `reports/ros_isaac_relaxed_v4_validation.txt`. |
| Busy rejection | The same real ROS-Isaac validation records one concurrent request accepted and the other rejected. |
| Timeout reaches `FAILED` | The real runtime is tested with `timeout_sec=0.2`. |

The concise meeting statement is: **the 29 tests protect implementation contracts; the 5/5 Isaac evaluation establishes recovery behavior; and the live ROS-Isaac validation establishes end-to-end communication and error handling.**
