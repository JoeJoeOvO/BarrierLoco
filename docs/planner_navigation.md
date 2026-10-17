# OCC 导航 Planner 与 PPO 训练

任务目标是：给定目标点，由 Planner 根据本体状态和局部 OCC 选择技能、产生该技能的 cmd，接近目标后刷新下一目标。当前实现包含感知、网络、冻结技能执行、导航环境和混合动作 PPO。训练流程已实现；接口与短训练通过不代表已有导航能力。

## 输入与网络

- 本体/执行状态 58 维：机体线速度 3、角速度 3、重力方向 3、腿关节相对位置 12、关节相对速度 16、上次底层动作 16、当前技能 one-hot 3、Jump 剩余时间 1、执行忙状态 1。线速度目前使用仿真真值，部署需提供本体状态估计。
- 目标 3 维：目标相对位置 XYZ，旋转到机体 yaw 坐标系，分别除以 8、8、2 米并裁剪。目标 Z 来自目标地面的高度。
- OCC：机体 yaw 对齐、以 base 为原点，X∈[-1.6,4.8)、Y∈[-2.4,2.4)、Z∈[-1.6,1.6) m，分辨率 0.2 m。张量为 `[N,2,16,24,32]`，两通道为 occupied 和 known。未知为 `[0,0]`，自由为 `[0,1]`，占用为 `[1,1]`。

OCC 来自实际地形网格上的首回波射线：360° 水平扫描、3° 间隔、24 层垂直射线（-85° 至 35°）、8 m 最大量程。沿射线以半体素步长标记已观测区域，命中体素标记占用，遮挡后方保持未知。这不是高度图挤出的假三维体素，也不读取目标技能或地形类别作为感知输入。当前是无噪声单帧静态地形扫描；未模拟机器人自遮挡、动态物体或多帧建图。无回波与无效测量分别处理。

选择 0.2 m 是初始显存/计算预算折中，不能精确表达 0.2 m 宽的最窄沟。应在实测小障碍通过率后决定是否增加局部高分辨率分支。目标可以在 OCC 范围外；策略只能利用当前局部地图，尚无全局地图或循环记忆，复杂死胡同可能需要后续扩展。

网络结构：

1. OCC → 三层 3D Conv（2→8→16→32，kernel 3，stride 2）→保留空间布局的展平→128 维。
2. 本体状态＋目标 → MLP 128→64。
3. 拼接后共享 MLP 256→128。
4. 技能 categorical 头 3 维；独立 cmd 头 10/3/4 维；value 头 1 维。激活为 ELU。

连续动作使用 tanh-squashed Gaussian。概率只包含技能选择及被选中的 cmd 头，并包含 tanh Jacobian；未选中的头不计入联合 log probability。Jump 忙状态下冻结 cmd，其连续动作似然不计入更新。调用者需保存并重用采样时的技能可执行 mask。value 头使用同一观测，目前不是性能 barrier 或安全 critic。

## 执行接口

Planner 每 0.1 s 更新；每次执行 5 个 0.02 s 底层控制步。三个技能参数冻结（Multiskill/GoalGrid 使用仓库内 TorchScript；HighPlatform 直接从保留的 checkpoint 重建 actor，不依赖本地临时导出）：

| 技能 | 权重 | Planner cmd |
| --- | --- | --- |
| Multiskill | multiskill_scratch80k/policy.pt | vx、vy、wz、身高、频率、三个步态相位参数、支撑比例、摆腿高度 |
| HighPlatform | highplatform_rough15k_batch196608/model_19998.pt | vx、vy、wz |
| GoalGrid | goalgrid_scratch5k/policy.pt | 前向速度、峰值高度、固定起跳坐标系下落点 X/Y |

每个技能分别重建原 actor 的 64/53/57 维输入，保留原始观测缩放、关节次序及 Multiskill 时钟。GoalGrid 选择事件就是起跳触发；捕获当前朝向作为起跳方向，1.8 s 执行＋0.4 s 恢复期间冻结选择与目标，速度指令转到当前 yaw 坐标系。当前使用定时结束，尚未加入接触确认和失败恢复技能。按用户决定保留选择即触发，不强加助跑流程；由 Planner 根据机器人当前速度和姿态学习起跳时机。站立直接触发的效果较差属于已知限制。

初始 cmd 范围见 `locolab/planner/skills.py`，它们只是探索范围，不是已验证能力范围。Multiskill 暂用 vx ±3 m/s；相位和支撑比例可连续变化，但非训练步态附近的组合有泛化风险。HighPlatform 的全向 cmd 也没有据此完成能力验证。GoalGrid 初始 H=0.15–0.25 m、X=0.4–0.8 m、Y=±0.15 m，框内组合并非全部有成功参考。

统一物理机器人暂沿用 GoalGrid 关节限值和自碰撞，但求解器仍为基础场景的 4/1 次，GoalGrid 原任务为 8/4 次；三个技能原始配置不变。Multiskill 与 HighPlatform 原有限值/自碰撞不同，接口接通并不证明它们在这个机器人上保持原有性能，需要单独验证和必要的适配。

## 连续导航目标

围绕机器人当前位置采样 4–20 m 的二维目标，允许跨地块、跨地形类别移动，仅限制在完整生成地图内。每个候选从全图最高出生面以上 10 m 向下射线，取实际地形表面为目标 Z。0.6 m 十字足迹高差小于 0.35 m，自定义沟底不作为目标。拒绝失败时目标无效，不显示标记、不产生到达奖励。地面支持不代表路线一定可达。

目标 XY 距离小于 0.5 m、机身比目标表面高 0.05–0.65 m，连续两个 Planner 步且 Jump 不忙，记到达并换目标，不重置机器人。训练保留 30 s 超时和全图边界终止，取消出生地块边界；回放仅摔倒/掉沟重置。橙色标记只显示当前机器人的有效目标，半径 8 cm，底部接触目标表面。

注意：`model_04004.pt` 是此前地块内目标规则训练的模型。当前回放使用新的跨地形规则，不能将其表现视为已在新任务上完成训练。旧运行的源码快照保留原始配置。

训练奖励：目标距离减少量＋到达 5 分－失败 5 分－每步 0.01。距离奖励在目标刷新前计算，失败重置不产生虚假进度。首轮使用这一基础导航奖励，其效果需由长期训练和独立评测判断。

## 检查入口

```bash
scripts/skill.sh omnijump python locolab/scripts/locomotion/preview_navigation.py --headless --num_envs 6 --steps 4 --report /tmp/planner-navigation-check.json
/home/joey/miniconda3/envs/env_isaaclab/bin/python -m pytest tests/test_navigation_planner.py tests/test_planner_terrain.py -q
```

默认脚本强制覆盖三类技能，检查 OCC、网络输出、动作输入和注入近目标后的刷新逻辑；不是导航成功率测试。`--random-planner` 才使用随机初始化网络的选择，不应把它当作训练效果展示。

## 首轮训练配置与复现

```bash
scripts/skill.sh omnijump python locolab/scripts/locomotion/train_planner.py --headless --num-envs 160 --horizon 32 --max-iterations 20000 --minibatch-size 256 --seed 42 --output logs/planner/<new_run_name>
```

- 160 个并行环境，10 类地形等列分配，初始化覆盖全部 6 档难度，不做自动难度升降。
- 从零初始化导航网络；初始技能 logits 为 [1,0,-1]，cmd 均值初始化在常用策略指令附近。这是探索先验，不按地形指定技能。三个底层策略始终冻结。
- 每轮采样 32 个高层步，即 5120 个导航转移；每个高层步执行 5 个底层步。
- PPO 4 epochs、minibatch 256、clip 0.2、初始学习率 3e-4（基于 KL 自适应）、gamma 0.99、GAE lambda 0.95、熵系数 0.005、梯度范数上限 1。
- 选择概率只包含当前可执行技能和选中 cmd 的密度；Jump 忙状态下无效 cmd 不参与概率更新。OCC 缓存用 uint8，网络输入时转换为 float。
- 高层每 0.1 s 检查失败和 30 s 超时。终止状态在物理重置前保存；失败不 bootstrap，超时使用终止观测价值 bootstrap，两者都切断跨回合 GAE。到达后刷新目标不切断回合。
- 每 100 轮及停止/结束时保存 checkpoint；支持 SIGTERM/SIGINT 在当前更新结束后保存并停止。`--resume <checkpoint>` 恢复网络和优化器，仿真从新 reset 开始，不声称逐帧复现原续训轨迹。
- 输出包含 `config.json`、`env.yaml`、相关源码快照、checkpoint、`metrics.jsonl`、TensorBoard 和原子更新的 `status.json`。除总奖励外，记录每地形到达/失败/超时、无效目标比例、实际技能占用比例与可决策时刻的技能选择比例。

短训练检查（会注入近目标、超时、越界以验证逻辑，不是导航成功率测试）：

```bash
scripts/skill.sh omnijump python locolab/scripts/locomotion/train_planner.py --headless --verify --num-envs 20 --horizon 4 --max-iterations 3 --minibatch-size 80 --save-interval 1 --output logs/planner/<new_preflight_name>
python -m pytest tests/test_planner_ppo.py tests/test_navigation_planner.py tests/test_planner_terrain.py -q
```

这是普通 PPO 导航基线，尚不包含 BPO 性能约束、部署 safety filter 或持续学习机制。成功实验记录只在后续独立评测确认完整任务效果后建立。

## 已训练 Planner 的交互回放

```bash
scripts/skill.sh omnijump python locolab/scripts/locomotion/play_planner.py --checkpoint <planner_checkpoint.pt>
```

每类地形放置一个机器人，默认观察楼梯地形（`--terrain-index 4`）、中等难度（第 3/6 行）和确定性策略输出。启动时对实际碰撞网格逐类向下射线检测，日志 `TERRAIN_AUDIT` 给出高度范围。相机跟随当前机器人；橙色球是目标。右侧 `Planner Navigation` 与 Stage 同组停靠，显示当前技能、实际 cmd、目标距离、到达/失败计数及速度跟踪曲线。Previous/Next terrain 切换观察对象，Easier/Harder 切换难度并重置，New goals 仅重新采样目标，不重置机器人，Toggle mean/sampled policy 切换确定性与采样策略。回放不更新网络，不强制指定技能。

交互回放只因摔倒姿态（倾斜超过 1 rad）或掉入沟底重置，不因 30 秒超时或越过当前地块边界重置。到达目标也只换目标。主动切换难度会重新放置机器人；训练保留超时，但边界改为全图边界。


## 连续混合地形训练选项

`train_planner.py --continuous-mixed --jump-always-active --episode-seconds 180 --progress-reward 3 --arrival-reward 20 --step-cost 0.02` 启用新版任务；不传 `--resume` 时从随机 Planner 权重开始，底层技能冻结。回放自动读取 checkpoint 中这些选项。

- 目标距当前位置 8–16 m，延续上一目标方向并在 ±60° 内改变航向；采样多次失败时扩大方向搜索，允许跨地块。目标仍通过表面射线和足迹支持检查，未保证全局路线一定可达。
- 进入 1.2 m 范围、水平速度低于 1.0 m/s、机身位于目标面上方 0.05–0.65 m，且 Jump 不忙时即可刷新目标，无需静止或持续停留。回合上限 180 s，训练保留摔倒/掉沟/全图越界终止。
- Jump actor 的模式输入始终为 1；每次选择锁定 1.8 s，取消旧版 0.4 s inactive 恢复段。并非基于落地事件的退出。全新的混合训练和纯 Gap 训练都用近似均等的初始技能选择先验。
- 平台高度 0.15–0.65 m、长度 1–3 m、起点 3–4.5 m；沟宽 0.2–0.6 m，纯沟起点 4–11 m，混合地形沟起点 9–12 m。各地块采用可复现的独立参数组合，均在本地块坐标下描述。楼梯单级高度范围 0.04–0.25 m，其余 Rough 类型保留原参数范围。地形在运行初始化时生成，不随每次 reset 重建。
- 奖励为水平距离进展 ×3、到达 +20、失败 −5、每高层步 −0.02。没有技能选择奖励。
- `spawn_terrain/*` 指按出生地形分组的统计，不表示跨地形运动时的当前所在地，也不是独立地形通过率。旧训练使用各自原始代码快照复现。


`--continuous-mixed --diverse-gaps` 使用五种纯沟场景（单沟、双沟、斜沟、折线沟、变宽沟），覆盖 96×80 m，保留连续方向目标及新版到达条件。沟宽 0.15–0.9 m、深度 0.6–1.4 m，各地块独立生成。所有目标排除沟底；纯沟场景机身中心下降到岸面以下（z<0）判为掉沟；机身低于岸面以上 0.12 m 持续 0.3 s，或机身接触力超过 10 N 持续 0.3 s，也判为塌陷/触地失败。几何由闭合棱柱组成，沟下无地表高度支撑板。训练/回放从 checkpoint 识别该场景。该设置允许绕行，不强制选择 Jump，宽沟也不保证在冻结技能能力范围内。

回放输出 `NAV_RESET`，记录重置前高度、掉沟、倾斜、持续塌低/机身触地和超时原因。此补充判定在 `model_12961.pt` 训练结束后加入，旧 checkpoint 的历史失败统计不含这些事件；旧运行源码快照保留原规则。
