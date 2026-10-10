# Go2W actuator model limits

The common Isaac Lab asset, deployment torque caps and MuJoCo
actuators use user-selected conservative rounding of the per-joint limits in the
[official Go2-W URDF](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/go2w_description/urdf/go2w_description.urdf).

| Joint group | Effort limit (Nm) | Velocity limit (rad/s) |
| --- | ---: | ---: |
| Hip | 23.5 | 30 |
| Thigh | 23.5 | 30 |
| Calf | 35 | 20 |
| Wheel | 23.5 | 30 |

The corresponding official URDF values remain 23.7/30.1 for hip/thigh/wheels and
35.55/20.07 for calves. The source URDF is not rewritten. Our rounded limits are
below those model limits, not confirmed continuous hardware ratings.
The [Go2-W product page](https://www.unitree.com/cn/go2-w/) lists approximately
45 Nm for the largest joint motor and explicitly states that the 16 motors have
different maximum torques. Do not apply 45 Nm to every joint or treat it as a
continuous rating. Real-robot operating limits still require device-specific
firmware, thermal and manufacturer confirmation.

The October 1 correction replaces uniform 23.5 Nm / 30 rad/s training limits,
removes the stale M20 velocity override, and fixes MuJoCo's previous 45.43 Nm
calf and 15 Nm wheel caps. The existing `legs` actuator group is retained;
joint order, PD gains, observations, action scales and network shapes do not change.
This common-asset change applies to future Go2W task instances, including rough,
and flat. The retained Jump task explicitly uses model_8500's historical
23.5 Nm / 30 rad/s leg simulation limits for checkpoint compatibility; see
[the Jump recipe](go2w_jump.md). This exception does not change hardware limits.
It does not alter already running training processes. Older
checkpoints and TO references were generated with the previous actuator model;
their original dynamics/replay metrics are historical, not a validation under
the corrected model. Re-solve TO and re-evaluate/resume policies before reuse.
Do not increase real-robot torque caps and deploy an old policy without separate
hardware safety review and validation. No training or hardware control is started
by changing these repository configurations.

## Verification

The correction passed 73 focused regression tests (one optional Python-MuJoCo
test skipped because those bindings are absent), Isaac Lab native 64-env /
120-step inference checking every joint's actual simulation effort and velocity
limits, and a separately compiled native MuJoCo check of all 16 actuator limits,
positive/negative saturated forces and 100 finite-state physics steps. The latter
uses the already installed MuJoCo library and does not connect to hardware.
These checks do not establish old-policy performance or revalidate legacy TO.
The obsolete ForwardJump experiments and their dedicated solver/smoke scripts
were removed when the user retained only the model_8500 Jump baseline.

```bash
python -m pytest tests/test_go2w_actuator_limits.py -q
g++ -std=c++17 tests/smoke_go2w_mujoco_limits.cpp \
  -I"$MUJOCO_DIR/include" -L"$MUJOCO_DIR/lib" -Wl,-rpath,"$MUJOCO_DIR/lib" \
  -lmujoco -o /tmp/smoke_go2w_mujoco_limits
/tmp/smoke_go2w_mujoco_limits deploy/src/robot_description/go2w_description/mjcf/go2w.xml
```
