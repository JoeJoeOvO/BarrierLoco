# Go2W RL 部署系统

将 IsaacLab 训练的强化学习策略部署到 Unitree Go2W 轮足机器人。

## 目录结构

```
deploy/
├── build.sh                    # 构建脚本
├── policy/go2w/               # 策略配置
│   ├── base.yaml              # 机器人基础参数
│   └── robot_lab/
│       ├── config.yaml        # 策略配置
│       └── policy.onnx        # 训练好的模型 (需手动放置)
├── src/
│   ├── rl_deploy/             # 核心部署代码
│   │   ├── src/
│   │   │   ├── rl_real_go2.cpp    # 真机部署
│   │   │   └── rl_sim_mujoco.cpp  # MuJoCo 仿真
│   │   └── library/core/      # 核心库
│   ├── robot_description/     # 机器人 URDF
│   ├── robot_joint_controller/
│   └── robot_msgs/
└── scripts/                   # 辅助脚本
```

## 快速开始

### 1. 安装依赖

```bash
sudo apt-get install -y libtbb-dev libglfw3-dev libyaml-cpp-dev
```

### 2. 构建

```bash
cd /path/to/deploy

# MuJoCo 仿真构建 (推荐先测试)
./build.sh -mj

# 或者硬件部署构建 (无仿真)
./build.sh -m

# 或者完整 ROS2 构建 (需要先 source ROS)
source /opt/ros/humble/setup.bash
./build.sh
```

### 3. 放置策略模型

将训练好的 ONNX 模型复制到 `policy/go2w/robot_lab/`:

```bash
cp /path/to/your/exported/policy.onnx policy/go2w/robot_lab/
```

### 4. 运行

```bash
# MuJoCo 仿真
./cmake_build/bin/rl_sim_mujoco go2w scene

# 真机部署 (需连接机器人)
./cmake_build/bin/rl_real_go2 go2w
```

## 构建模式

| 命令 | 用途 | 输出 |
|------|------|------|
| `./build.sh -mj` | MuJoCo 仿真测试 | `rl_sim_mujoco` |
| `./build.sh -m` | 纯硬件部署 | `rl_real_go2` |
| `./build.sh` | ROS2 完整构建 | ROS2 节点 |
| `./build.sh -c` | 清理构建 | - |

## 键盘控制

### 状态切换
| 按键 | 功能 |
|------|------|
| `0` | 切换到 GetUp (站立) |
| `1` | 切换到 RL Locomotion |
| `9` | 切换到 GetDown (趴下) |
| `P` | 切换到 Passive (被动) |

### 运动控制
| 按键 | 功能 |
|------|------|
| `W/S` | 前进/后退 |
| `A/D` | 左移/右移 |
| `Q/E` | 左转/右转 |
| `Space` | 停止 |

### 仿真控制
| 按键 | 功能 |
|------|------|
| `Enter` | 暂停/继续仿真 |
| `R` | 重置仿真 |

## 配置说明

### base.yaml - 机器人基础参数

```yaml
go2w:
  dt: 0.005              # 仿真时间步长
  decimation: 4          # 控制频率 = 1/(dt*decimation) = 50Hz
  num_of_dofs: 16        # 自由度数量 (12腿 + 4轮)
  wheel_indices: [12, 13, 14, 15]  # 轮子关节索引
  default_dof_pos: [...]  # 默认关节位置
  fixed_kp: [...]         # 固定 PD 增益 (非 RL 模式)
  fixed_kd: [...]
```

### config.yaml - 策略配置

```yaml
go2w/robot_lab:
  model_name: "policy.onnx"
  num_observations: 53
  observations: ["ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions"]

  # 关节顺序映射 (SDK -> 训练)
  obs_mapping: [...]      # 观测重排
  action_mapping: [...]   # 动作重排

  rl_kp: [...]            # RL 控制时的 PD 增益
  rl_kd: [...]
  action_scale: [...]     # 动作缩放
```

## 从训练到部署

### 1. 训练 (IsaacLab)
```bash
cd /path/to/locolab
python scripts/train.py --task Go2W-Flat-v0
```

### 2. 导出 ONNX
训练完成后模型自动导出到 `logs/rsl_rl/.../exported/policy.onnx`

### 3. 部署
```bash
cp logs/.../exported/policy.onnx deploy/policy/go2w/robot_lab/
cd deploy
./build.sh -mj
./cmake_build/bin/rl_sim_mujoco go2w scene
```

## 关节顺序说明

**重要**: IsaacLab 训练和 SDK 部署使用不同的关节顺序！

| 顺序类型 | 排列方式 |
|---------|---------|
| **训练 (IsaacLab)** | 按类型分组: 4×hip, 4×thigh, 4×calf, 4×wheel |
| **部署 (SDK)** | 按腿分组: FR(3), FL(3), RR(3), RL(3), wheels(4) |

`config.yaml` 中的 `obs_mapping` 和 `action_mapping` 自动处理这个转换。
