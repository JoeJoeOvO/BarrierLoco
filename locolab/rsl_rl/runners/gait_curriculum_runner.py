"""Preserve command curriculum progress alongside normal PPO checkpoints."""

from .on_policy_runner import OnPolicyRunner


class GaitCurriculumRunner(OnPolicyRunner):
    def _command(self):
        return self.env.unwrapped.command_manager.get_term("base_velocity")

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos["wtw_curriculum"] = self._command().curriculum_state_dict()
        return super().save(path, infos)

    def load(self, path, load_optimizer=True, map_location=None):
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        if load_optimizer:
            if not infos or "wtw_curriculum" not in infos:
                raise ValueError("Use --init_checkpoint for a model without curriculum state.")
            self._command().load_curriculum_state_dict(infos["wtw_curriculum"])
            self.current_learning_iteration += 1
            self.env.reset()
        return infos
