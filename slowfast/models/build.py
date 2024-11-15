#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry
from torch.distributed.algorithms.ddp_comm_hooks import (
    default as comm_hooks_default,
)

import slowfast.utils.logging as logging
import slowfast.utils.distributed as du

logger = logging.get_logger(__name__)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.solver.num_gpus <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.num_gpus == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.network.model_name
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.BN.NORM_TYPE == "sync_batchnorm_apex":
        try:
            import apex
        except ImportError:
            raise ImportError("APEX is required for this model, pelase install")

        logger.info("Converting BN layers to Apex SyncBN")
        process_group = apex.parallel.create_syncbn_process_group(
            group_size=cfg.BN.NUM_SYNC_DEVICES
        )
        model = apex.parallel.convert_syncbn_model(
            model, process_group=process_group
        )

    if cfg.solver.num_gpus:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.solver.num_gpus == 1:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cur_device],find_unused_parameters=True)

    if cfg.solver.num_gpus > 1:
        model = model.cuda(device=cur_device)
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=True
            if cfg.network.model_name == "ContrastiveModel"
            or cfg.network.model_name == "ClipImage"
            or cfg.network.model_name == "BasicClip"
            or cfg.network.model_name == "TemporalClipVideo"
            else False,
            static_graph=cfg.network.static_graph
        )
        if cfg.network.fp16_allreduce:
            model.register_comm_hook(
                state=None, hook=comm_hooks_default.fp16_compress_hook
            )
    return model
