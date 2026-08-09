"""
Load and prepare UMA models for libTorch export.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from fairchem.core.calculate.pretrained_mlip import (
    get_predict_unit,
    get_reference_energies,
    pretrained_checkpoint_path_from_name,
)
from fairchem.core.units.mlip_unit import InferenceSettings, load_predict_unit
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
from fairchem.core.units.mlip_unit.utils import (
    get_backbone_class_from_checkpoint,
    load_inference_model,
)

from common import disable_derivative_regression, phase0_inference_settings

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.models.base import HydraModel
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint


def _build_overrides(
    checkpoint: MLIPInferenceCheckpoint,
    settings: InferenceSettings,
    user_overrides: dict | None,
) -> dict:
    overrides = {} if user_overrides is None else dict(user_overrides)
    if "backbone" not in overrides:
        overrides["backbone"] = {}
    backbone_cls = get_backbone_class_from_checkpoint(checkpoint)
    overrides["backbone"].update(backbone_cls.build_inference_settings(settings))
    return overrides


def load_prepared_hydra_model(
    checkpoint_path: str,
    sample_data: AtomicData,
    settings: InferenceSettings | None = None,
    device: str = "cpu",
) -> tuple[HydraModel, MLIPInferenceCheckpoint, dict | None]:
    """
    Load checkpoint, run prepare_for_inference, return the inner HydraModel.
    """
    if settings is None:
        settings = phase0_inference_settings()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    overrides = _build_overrides(
        checkpoint,
        settings,
        {"backbone": {"execution_mode": settings.execution_mode or "general"}},
    )

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(settings.base_precision_dtype)
    try:
        averaged, checkpoint = load_inference_model(
            checkpoint_path,
            use_ema=True,
            overrides=overrides,
            preloaded_checkpoint=checkpoint,
        )
        model: HydraModel = averaged.module
        model.setup_tasks(checkpoint.tasks_config)
    finally:
        torch.set_default_dtype(prev_dtype)

    model.eval()
    model.prepare_for_inference(sample_data, settings)
    disable_derivative_regression(model)
    model.to(device)
    return model, checkpoint, None


def load_oracle_from_checkpoint(
    checkpoint_path: str,
    settings: InferenceSettings | None = None,
    device: str = "cpu",
) -> MLIPPredictUnit:
    if settings is None:
        settings = phase0_inference_settings()
    return load_predict_unit(
        checkpoint_path,
        inference_settings=settings,
        device=device,
    )


def load_oracle_predict_unit(
    model_name: str,
    settings: InferenceSettings | None = None,
    device: str = "cpu",
) -> MLIPPredictUnit:
    if settings is None:
        settings = phase0_inference_settings()
    return get_predict_unit(
        model_name,
        inference_settings=settings,
        device=device,
    )


def get_atom_refs(model_name: str) -> dict | None:
    try:
        return get_reference_energies(model_name, "atom_refs")
    except Exception as exc:
        logging.warning("Could not load atom_refs for %s: %s", model_name, exc)
        return None
