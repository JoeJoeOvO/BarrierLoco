# Go2W Parkour：保留的单阶段跳跃技能

最终任务：`OmniBot-Go2W-Parkour-FixedTargetTuned-MLP-v0`。保留其原名称用于已有checkpoint；
已移除早期固定1m/s、两阶段Speed、未优化FixedTarget三套实验任务。此次整理不改变当前训练行为。

## 训练与接口

- 从随机初始化训练5000次，4096环境，纯MLP：57 → 512 → 256 → 128 → 16，ELU。
- actor为57维，critic为312维。actor无高度/竖直速度/跳跃相位状态；critic包含相位和动态落点误差。
- 7维command：`[vx_yaw, vy_yaw, yaw_rate, jump, height, target_dx, target_dy]`。
  vx是固定世界前进方向速度转换至当前航向坐标系；heading rate为0。
  height、dx、dy描述TO初始坐标系中的目标，在接近和跳跃期间固定，不是当前位置到目标的剩余位移。
- vx范围[0.2,1.5]m/s：70%均匀采样，15%固定0.2，15%固定1.5。
- 同时训练正常滚动和7种跳跃：三个平地落点、40/60cm沟壑、左右20°横坡斜台。
  物理地形和每条TO一一对应；不对TO时间、几何或控制尺度做速度缩放。
- 重置总体占比：30%普通滚动、50%地面跳跃接近、20%参考相位练习。
  地面完整起跳单独记账，不把空中重置计为真正起跳成功。
- episode上限10s，跳跃参考时序固定2.2s；平地/沟壑成功后再按vx滚动1s，斜台在目标处停稳。
- Go2W action scale保持腿0.25、轮10；腿部力矩上限23.5/23.5/35Nm，速度上限30/30/20rad/s。
- 训练保留原有随机化；Play关闭随机化、扰动和观测噪声。

## 保留的奖励与判定

跳跃追踪实测TO的关节位置/速度、机身位置/速度、姿态、角速度及接触时序。
上升速度+100、高度误差平方−200、高度tanh+20、真实四轮腾空+10。
非跳跃状态单独追踪速度、高度和站姿，限制多余竖直/角运动。
本轮保留的优化包括：接近速度跟踪+20、速度误差−10、TO速度跟踪+15；
XY位置误差反馈仅影响速度奖励目标；腾空落点预测−60、恢复位置误差−80。
第一次着地位置误差按`−100×XY误差²`只计一次；RSI辅助事件按1/6缩放。
成功/失败完整事件分别+30/−30；动作变化、关节限位、空中轮速等正则化保留。

完整成功要求真实四轮离地并达到目标高度、首次落地XY误差≤20cm，
终点3D位置误差<20cm、速度误差<0.3m/s、姿态与四轮稳定着地；
斜台另要求轮子在所选台面内。没有放宽这些门限。

## 最终产物

`deploy/policy/go2w/parkour_mlp_20261012/`保存最终`model_4999.pt`、TorchScript、ONNX、
原始训练参数、随机初始化证明及导出校验信息。
TO训练库保留在`locolab/source/locomotion/mdp/reference_motions/go2w_parkour_bank.json.gz`，
SHA256：`45b202e86d29022f312f8e911f108f11e7a716f7b35decbd5ba553fafd1966a5`。
该库只包含用户选定的7条实测回放，保留原始验收状态，不将人工选用改写为严格验收通过。

同口径最终测试：四档速度×7场景×3次，84次真实起跳，80次完整成功（95.2%）。
成功跳跃机身相对起跳点抬升14.9–22.4cm；这不是轮底净离地高度或可跨障碍高度。
0.2m/s右斜台3次碰撞，1m/s左斜台1次末端位置误差20.7cm。普通滚动36/36完成10s。
这些是无随机化的小样本测试，不构成实机或域随机化鲁棒性认证。

## 复现与验证

在现有Isaac Lab环境中，从仓库根目录：

```bash
cd locolab
python scripts/locomotion/train.py --task OmniBot-Go2W-Parkour-FixedTargetTuned-MLP-v0 \
  --from_scratch --num_envs 4096 --max_iterations 5000 --logger wandb --video
python scripts/locomotion/play.py --task OmniBot-Go2W-Parkour-FixedTargetTuned-MLP-v0 \
  --checkpoint ../deploy/policy/go2w/parkour_mlp_20261012/model_4999.pt --num_envs 10 --video
python scripts/locomotion/evaluate_parkour_ground.py \
  --checkpoint ../deploy/policy/go2w/parkour_mlp_20261012/model_4999.pt --output /tmp/parkour_evaluation.json --headless
```

CPU导出及一致性校验（仓库根目录）：

```bash
python deploy/policy/go2w/parkour_mlp_20261012/export_policy.py --verify-only
python -m pytest -q tests/test_parkour_command.py tests/test_parkour_reference_bank.py tests/test_parkour_scratch_initialization.py
```

TO求解/回放工具及其几何与验收测试保留，不将求解器代码当作训练中间产物删除。
原已提交的Go2W Jump技能及其54维网络不在本轮清理范围。

**部署边界：**这个网络是57维观测/7维command，不能直接替换已有54维Jump网络。
ONNX/JIT导出正确不代表现有sim2sim或实机command调度已适配；本次没有改C++运行时。

## 本轮整理与验证

最终代码、网络和评估摘要集中在本分支；评估及视频见
`artifacts/go2w_parkour/20261012_final_4999/`。
原始训练源文件以压缩快照保留在网络包的`params/training_source/`，与训练时哈希逐一核对。
清理后的Train、Play及PPO有效配置与原版逐项相等，网络与TO库哈希未改变。
214项TO/指令/初始化测试和25项共享配置/旧Jump回归测试、30环境320步原生接触事件审计、
385组ONNX/JIT一致性校验及
本次修改文件的pre-commit检查均通过。原生验证检查到21次首次落地事件，无重复计罚。
归档网络加载压缩参数后的120环境回放再次得到80/84跳跃成功和36/36滚动完成，
每个episode的完整记录与整理前完全一致。详见网络包中的`cleanup_manifest.json`。

已删除本地14个旧实验目录、服务器14个旧工作区，以及当前训练的smoke、
中间checkpoint、训练过程录像和重复原始录像；完整评估轨迹压缩保留，最终录像保留。
W&B线上历史、非跑酷任务、原已提交的Jump技能及其归档不变。
10月12日整理时未启动新训练，也未commit或push。

## 10月13日发布范围

保留的7条TO技能、已验证的`model_4999.pt`、ONNX/JIT、参数快照、评估和录像与
GoalGrid扩展一起发布在`codex/go2w-parkour-goal-grid`分支。
GoalGrid追加23条独立验收通过的平地TO，保留原7条不变；详见`docs/go2w_parkour_goal_grid.md`。
上述网络仍对应7条TO任务，不是正在服务器训练的30条TO策略；未将中间checkpoint作为最终网络发布。
网络、较大的TO库和完整评估/视频通过局部Git LFS规则保存，克隆后需`git lfs pull`。
