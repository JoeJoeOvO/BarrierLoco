"""Export and verify the bundled Go2W HighPlatform checkpoint on CPU, without Isaac Sim."""

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

SOURCE_HASHES = {
    "model_9998.pt": "85a13a13d35bd2ec7ff1eb5a1a25aaa3640668a5918bdac4072774a7b097c46a",
    "params/agent.yaml": "182992edaa70665569e95d9668a0976a3fc64c32318d8e1c339b983a0992749f",
    "params/env.yaml": "c02ea5986273c926a5378ee2ec9900a02d1bf40d911ab62be4eff5c33c0030f0",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy():
    for name in ("model_9998.pt", "params/agent.yaml"):
        if sha256(BUNDLE / name) != SOURCE_HASHES[name]:
            raise ValueError(f"Source checksum mismatch: {name}. Run git lfs pull if this is a pointer file.")
    with gzip.open(BUNDLE / "params/env.yaml.gz", "rb") as file:
        if hashlib.sha256(file.read()).hexdigest() != SOURCE_HASHES["params/env.yaml"]:
            raise ValueError("Original environment configuration checksum mismatch.")

    agent = yaml.safe_load((BUNDLE / "params/agent.yaml").read_text())
    checkpoint = torch.load(BUNDLE / "model_9998.pt", map_location="cpu", weights_only=True)
    if checkpoint["iter"] != 9998 or agent["max_iterations"] != 5000:
        raise ValueError("Expected the final checkpoint after 5000 additional HighPlatform updates.")
    if not agent["resume"] or agent["load_checkpoint"] != "model_4999.pt":
        raise ValueError("Expected the Rough MLP resume configuration.")
    config = dict(agent["policy"])
    if config.pop("class_name") != "ActorCritic" or agent["empirical_normalization"]:
        raise ValueError("This exporter is scoped to the bundled non-recurrent HighPlatform MLP.")
    observations = TensorDict({"policy": torch.zeros(1, 53), "critic": torch.zeros(1, 307)}, batch_size=[1])
    policy = ActorCritic(observations, agent["obs_groups"], num_actions=16, **config)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    return policy


def export(policy):
    # Load the same exporter as play.py without importing Isaac's environment wrappers.
    package = importlib.util.find_spec("isaaclab_rl")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("Use the existing Isaac Lab Python environment; isaaclab_rl was not found.")
    path = Path(next(iter(package.submodule_search_locations))) / "rsl_rl/exporter.py"
    spec = importlib.util.spec_from_file_location("highplatform_isaac_exporter", path)
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
    if [(item.name, item.shape, item.type) for item in session.get_inputs()] != [("obs", [1, 53], "tensor(float)")]:
        raise ValueError("Unexpected ONNX observation interface.")
    if [(item.name, item.shape, item.type) for item in session.get_outputs()] != [
        ("actions", [1, 16], "tensor(float)")
    ]:
        raise ValueError("Unexpected ONNX action interface.")
    jit = torch.jit.load(str(BUNDLE / "policy.pt"), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(42)
    samples = torch.cat([torch.zeros(1, 53), *[torch.randn(128, 53, generator=generator) * s for s in (0.1, 1, 10)]])
    errors = {"torchscript": 0.0, "onnx": 0.0}
    with torch.inference_mode():
        for sample in samples.split(1):
            observation = TensorDict({"policy": sample, "critic": torch.zeros(1, 307)}, batch_size=[1])
            expected = policy.act_inference(observation).numpy()
            outputs = {
                "torchscript": jit(sample).numpy(),
                "onnx": session.run(["actions"], {"obs": sample.numpy()})[0],
            }
            for name, output in outputs.items():
                if output.shape != (1, 16) or not np.isfinite(output).all():
                    raise ValueError(f"Invalid {name} inference output.")
                # FP32 accumulation differs on the deliberately out-of-distribution 10x inputs.
                # Observed worst case: 1.69e-4; unit-scale inputs stay below 2.39e-5.
                np.testing.assert_allclose(output, expected, rtol=1e-5, atol=2e-4)
                errors[name] = max(errors[name], float(np.abs(output - expected).max()))
    return {"samples": len(samples), "seed": 42, "max_absolute_error": errors, "rtol": 1e-5, "atol": 2e-4}


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
            "model_9998.pt",
            "policy.pt",
            "policy.onnx",
            "params/agent.yaml",
            "params/env.yaml.gz",
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
            "task": "OmniBot-Go2W-HighPlatform-MLP-v0",
            "provenance": json.loads((BUNDLE / "training_provenance.json").read_text()),
            "source_config_sha256": SOURCE_HASHES,
            "iteration": 9998,
            "additional_training_iterations": 5000,
            "actor_observations": 53,
            "critic_observations": 307,
            "actions": 16,
            "actor_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "deterministic_mean_actions": True,
            "observation_normalization": False,
            "onnx_interface": {"input": "obs", "input_shape": [1, 53], "output": "actions", "output_shape": [1, 16]},
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
