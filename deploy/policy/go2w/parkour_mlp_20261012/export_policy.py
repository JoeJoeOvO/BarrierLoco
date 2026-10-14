"""Export and verify the retained 57-observation Go2W Parkour MLP without Isaac Sim."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[4]
BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "locolab"))
from rsl_rl.modules import ActorCritic

MODEL_SHA256 = "2fe50b758224b45bfa2f689a37931a3e9d6e4cef670e5de675b9a4b7584eeb50"
BANK_SHA256 = "45b202e86d29022f312f8e911f108f11e7a716f7b35decbd5ba553fafd1966a5"
BANK = ROOT / "locolab/source/locomotion/mdp/reference_motions/go2w_parkour_bank.json.gz"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy():
    if sha256(BUNDLE / "model_4999.pt") != MODEL_SHA256 or sha256(BANK) != BANK_SHA256:
        raise ValueError("Retained checkpoint or TO bank checksum mismatch.")
    agent = yaml.safe_load((BUNDLE / "params/agent.yaml").read_text())
    initial = yaml.safe_load((BUNDLE / "params/initialization.yaml").read_text())
    with gzip.open(BUNDLE / "params/env.yaml.gz", "rt") as stream:
        env = yaml.load(stream, Loader=yaml.BaseLoader)
    assert not agent["resume"] and agent["max_iterations"] == 5000
    assert initial["mode"] == "scratch" and not initial["checkpoint_loaded"]
    assert initial["initial_iteration"] == initial["optimizer_state_entries"] == 0
    assert env["training_stage"] == "fixed_target_single_stage_tuned"
    assert int(env["command_semantics_version"]) == 3 and int(env["critic_command_semantics_version"]) == 2
    assert int(env["reward_recipe_version"]) == 2
    assert env["commands"]["parkour"]["landing_command_mode"] == "fixed"
    checkpoint = torch.load(BUNDLE / "model_4999.pt", map_location="cpu", weights_only=True)
    assert checkpoint["iter"] == 4999
    assert all(torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values())
    assert {float(v["step"]) for v in checkpoint["optimizer_state_dict"]["state"].values()} == {100000.0}
    config = dict(agent["policy"])
    assert config.pop("class_name") == "ActorCritic" and not agent["empirical_normalization"]
    assert not config["actor_obs_normalization"]
    obs = TensorDict({"policy": torch.zeros(1, 57), "critic": torch.zeros(1, 312)}, batch_size=[1])
    policy = ActorCritic(obs, agent["obs_groups"], num_actions=16, **config)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return policy.eval()


def export(policy):
    package = importlib.util.find_spec("isaaclab_rl")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("Use the existing Isaac Lab Python environment.")
    path = Path(next(iter(package.submodule_search_locations))) / "rsl_rl/exporter.py"
    spec = importlib.util.spec_from_file_location("parkour_isaac_exporter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.export_policy_as_jit(policy, policy.actor_obs_normalizer, str(BUNDLE))
    module.export_policy_as_onnx(policy, str(BUNDLE), policy.actor_obs_normalizer)
    return dict(implementation="isaaclab_rl/rsl_rl/exporter.py", sha256=sha256(path))


def verify(policy):
    onnx.checker.check_model(str(BUNDLE / "policy.onnx"), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(BUNDLE / "policy.onnx"), options, providers=["CPUExecutionProvider"])
    assert [(x.name, x.shape, x.type) for x in session.get_inputs()] == [("obs", [1, 57], "tensor(float)")]
    assert [(x.name, x.shape, x.type) for x in session.get_outputs()] == [("actions", [1, 16], "tensor(float)")]
    jit = torch.jit.load(str(BUNDLE / "policy.pt"), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(42)
    samples = torch.cat([torch.zeros(1, 57), *[torch.randn(128, 57, generator=generator) * s for s in (0.1, 1, 10)]])
    errors = {"torchscript": 0.0, "onnx": 0.0}
    with torch.inference_mode():
        for sample in samples.split(1):
            expected = policy.act_inference(TensorDict({"policy": sample}, batch_size=[1])).numpy()
            outputs = {"torchscript": jit(sample).numpy(), "onnx": session.run(["actions"], {"obs": sample.numpy()})[0]}
            for name, value in outputs.items():
                assert value.shape == (1, 16) and np.isfinite(value).all()
                np.testing.assert_allclose(value, expected, rtol=1e-5, atol=2e-4)
                errors[name] = max(errors[name], float(np.abs(value - expected).max()))
    return dict(samples=len(samples), seed=42, max_absolute_error=errors, rtol=1e-5, atol=2e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = load_policy()
    exporter = None if args.verify_only else export(policy)
    validation = verify(policy)
    files = {
        name: sha256(BUNDLE / name)
        for name in (
            "model_4999.pt",
            "policy.pt",
            "policy.onnx",
            "params/agent.yaml",
            "params/env.yaml.gz",
            "params/initialization.yaml",
            "export_policy.py",
        )
    }
    if args.verify_only:
        report = json.loads((BUNDLE / "export_manifest.json").read_text())
        if report["files_sha256"] != files:
            raise ValueError("Exported bundle differs from its manifest.")
    else:
        report = dict(
            task="OmniBot-Go2W-Parkour-FixedTargetTuned-MLP-v0",
            checkpoint_iteration=4999,
            total_training_updates=5000,
            random_initialization=True,
            actor_observations=57,
            critic_observations=312,
            actions=16,
            actor_hidden_dims=[512, 256, 128],
            activation="elu",
            observation_normalization=False,
            actor_command_version=3,
            critic_command_version=2,
            external_command_and_phase_logic_required=True,
            bank_sha256=BANK_SHA256,
            onnx_interface={"input": "obs", "input_shape": [1, 57], "output": "actions", "output_shape": [1, 16]},
            exporter=exporter,
            validation=validation,
            validation_scope=(
                "CPU export equivalence; native simulation reviewed separately; "
                "not configured for existing 54-input Jump deployment; no real-robot validation."
            ),
            versions={"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
            files_sha256=files,
        )
        (BUNDLE / "export_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
