"""Export and verify the bundled Go2W Jump checkpoint on CPU, without Isaac Sim."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import re
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

SOURCE_HASHES = {
    "model_4999.pt": "173b0d9e22e2b451c317b7892b326a341aa85358a1c280e6f39cd36ec50f71fb",
    "model_9998.pt": "64dc3fe4396e79afd0bc3e8a2496f6cc262ffa78466d409ff7c1e454d74dc550",
    "params/agent.yaml": "1e0187540c50b499ecb2253801beff6300bc733fa66d90d9f615912a462f893f",
    "params/env.yaml": "6199e03dfd99ecaf90bc430fe48a030d7ca426f2166a4181b1dc497e50f87182",
    "params/skill/agent.yaml": "e570798a043d5086ab5d5c6472c209e6f4276778af1d21512579c7cbd9d6d302",
    "params/skill/env.yaml": "7d65c22ecd54c9a307ac105bf15b377624017bb0c7d5e071a0dd220cfb6d06d0",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy():
    for name in ("model_4999.pt", "model_9998.pt", "params/agent.yaml", "params/skill/agent.yaml"):
        if sha256(BUNDLE / name) != SOURCE_HASHES[name]:
            raise ValueError(f"Source checksum mismatch: {name}. Run git lfs pull if this is a pointer file.")
    for name in ("params/env.yaml", "params/skill/env.yaml"):
        with gzip.open(BUNDLE / (name + ".gz"), "rb") as file:
            if hashlib.sha256(file.read()).hexdigest() != SOURCE_HASHES[name]:
                raise ValueError("Original environment configuration checksum mismatch.")

    agent = yaml.safe_load((BUNDLE / "params/agent.yaml").read_text())
    skill_agent = yaml.safe_load((BUNDLE / "params/skill/agent.yaml").read_text())
    if skill_agent["resume"] or skill_agent["max_iterations"] != 5000:
        raise ValueError("Expected a randomly initialized 5000-update skill stage.")
    checkpoint = torch.load(BUNDLE / "model_9998.pt", map_location="cpu", weights_only=True)
    if checkpoint["iter"] != 9998 or agent["max_iterations"] != 5000:
        raise ValueError("Expected the final checkpoint after 5000 additional Jump updates.")
    if not agent["resume"] or re.fullmatch(agent["load_checkpoint"], "model_4999.pt") is None:
        raise ValueError("Expected the bundled skill-stage MLP resume configuration.")
    with gzip.open(BUNDLE / "params/env.yaml.gz", "rt") as file:
        environment = yaml.load(file, Loader=yaml.BaseLoader)
    jump = environment["commands"]["jump_command"]
    if [float(jump[key]) for key in ("takeoff_time", "apex_time", "landing_time", "sequence_duration")] != [
        0.38,
        0.62,
        0.88,
        2.2,
    ]:
        raise ValueError("Unexpected fixed jump schedule.")
    if float(jump["heading_hold_duration"]) != 2.2 or environment["training_stage"] != "speed":
        raise ValueError("Expected speed-stage heading hold through the entire jump.")
    first = torch.load(BUNDLE / "model_4999.pt", map_location="cpu", weights_only=True)
    if first["iter"] != 4999 or "optimizer_state_dict" not in first:
        raise ValueError("Invalid skill-stage resume checkpoint.")
    config = dict(agent["policy"])
    if config.pop("class_name") != "ActorCritic" or agent["empirical_normalization"]:
        raise ValueError("This exporter is scoped to the bundled non-recurrent Jump MLP.")
    observations = TensorDict({"policy": torch.zeros(1, 54), "critic": torch.zeros(1, 309)}, batch_size=[1])
    policy = ActorCritic(observations, agent["obs_groups"], num_actions=16, **config)
    policy.load_state_dict(first["model_state_dict"], strict=True)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    return policy


def export(policy):
    # Load the same exporter as play.py without importing Isaac's environment wrappers.
    package = importlib.util.find_spec("isaaclab_rl")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("Use the existing Isaac Lab Python environment; isaaclab_rl was not found.")
    path = Path(next(iter(package.submodule_search_locations))) / "rsl_rl/exporter.py"
    spec = importlib.util.spec_from_file_location("jump_isaac_exporter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.export_policy_as_jit(policy, policy.actor_obs_normalizer, str(BUNDLE))
    module.export_policy_as_onnx(policy, str(BUNDLE), policy.actor_obs_normalizer)
    return {"implementation": "isaaclab_rl/rsl_rl/exporter.py", "sha256": sha256(path)}


def verify(policy):
    onnx.checker.check_model(str(BUNDLE / "policy.onnx"), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(BUNDLE / "policy.onnx"), options, providers=["CPUExecutionProvider"])
    if [(item.name, item.shape, item.type) for item in session.get_inputs()] != [("obs", [1, 54], "tensor(float)")]:
        raise ValueError("Unexpected ONNX observation interface.")
    if [(item.name, item.shape, item.type) for item in session.get_outputs()] != [
        ("actions", [1, 16], "tensor(float)")
    ]:
        raise ValueError("Unexpected ONNX action interface.")
    jit = torch.jit.load(str(BUNDLE / "policy.pt"), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(42)
    samples = torch.cat([torch.zeros(1, 54), *[torch.randn(128, 54, generator=generator) * s for s in (0.1, 1, 10)]])
    errors = {"torchscript": 0.0, "onnx": 0.0}
    by_scale = {name: dict(errors) for name in ("zero", "0.1", "1", "10")}
    with torch.inference_mode():
        for index, sample in enumerate(samples.split(1)):
            scale = "zero" if index == 0 else ("0.1", "1", "10")[(index - 1) // 128]
            observation = TensorDict({"policy": sample, "critic": torch.zeros(1, 309)}, batch_size=[1])
            expected = policy.act_inference(observation).numpy()
            outputs = {
                "torchscript": jit(sample).numpy(),
                "onnx": session.run(["actions"], {"obs": sample.numpy()})[0],
            }
            for name, output in outputs.items():
                if output.shape != (1, 16) or not np.isfinite(output).all():
                    raise ValueError(f"Invalid {name} inference output.")
                # FP32 accumulation differs on the deliberately out-of-distribution 10x inputs.
                np.testing.assert_allclose(output, expected, rtol=1e-5, atol=2e-4)
                errors[name] = max(errors[name], float(np.abs(output - expected).max()))
                by_scale[scale][name] = max(by_scale[scale][name], float(np.abs(output - expected).max()))
    return {
        "samples": len(samples),
        "seed": 42,
        "max_absolute_error": errors,
        "max_absolute_error_by_input_std": by_scale,
        "rtol": 1e-5,
        "atol": 2e-4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify committed networks without overwriting them."
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = load_policy()
    exporter = None if args.verify_only else export(policy)
    validation = verify(policy)
    files = {
        name: sha256(BUNDLE / name)
        for name in (
            "model_4999.pt",
            "model_9998.pt",
            "policy.pt",
            "policy.onnx",
            "params/agent.yaml",
            "params/env.yaml.gz",
            "params/skill/agent.yaml",
            "params/skill/env.yaml.gz",
            "params/source_files_sha256.json",
            "training_provenance.json",
        )
    }
    if args.verify_only:
        manifest = json.loads((BUNDLE / "export_manifest.json").read_text())
        if files != manifest["files_sha256"]:
            raise ValueError("Bundled network/configuration checksum does not match the manifest.")
    else:
        manifest = {
            "task": "OmniBot-Go2W-Jump-MLP-v0",
            "provenance": json.loads((BUNDLE / "training_provenance.json").read_text()),
            "source_config_sha256": SOURCE_HASHES,
            "iteration": 9998,
            "additional_training_iterations": 5000,
            "total_training_updates": 10000,
            "command_schedule_is_external": True,
            "actor_observations": 54,
            "critic_observations": 309,
            "actions": 16,
            "actor_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "deterministic_mean_actions": True,
            "observation_normalization": False,
            "onnx_interface": {"input": "obs", "input_shape": [1, 54], "output": "actions", "output_shape": [1, 16]},
            "exporter": exporter,
            "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
            "validation": validation,
            "validation_scope": (
                "CPU export equivalence; separate final native Isaac Lab evaluation; no real-robot validation"
            ),
            "files_sha256": files,
        }
        (BUNDLE / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
