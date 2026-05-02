# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_cnn import ActorCriticCNN
from .actor_critic_dreamwaq import ActorCriticDreamWaQ
from .actor_critic_recurrent import ActorCriticRecurrent
from .decoder import DecoderCollection
from .dreamwaq_estimator import DreamWaQVAE
from .rnd import RandomNetworkDistillation, resolve_rnd_config
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .symmetry import resolve_symmetry_config

__all__ = [
    "ActorCritic",
    "ActorCriticCNN",
    "ActorCriticDreamWaQ",
    "ActorCriticRecurrent",
    "DecoderCollection",
    "DreamWaQVAE",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "resolve_rnd_config",
    "resolve_symmetry_config",
]
