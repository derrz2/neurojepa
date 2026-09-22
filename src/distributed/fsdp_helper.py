# src/utils/fsdp_helper.py
import functools
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from ..models.layer.vit_components import Block 


def wrap_submodule_fsdp(submodule, reduce_fp32=False, use_mixed_precision=True): # wrapper module
    mp_policy = None
    if use_mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32 if reduce_fp32 else torch.bfloat16,
            buffer_dtype=torch.float32,
        )
    
    wrap_policy = ModuleWrapPolicy({Block}) if not reduce_fp32 else None
    
    return FSDP(
        submodule,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        use_orig_params=True
    )


def wrap_pretrain_modules_fsdp(model):
    wrap_specs = (
        ("backbone", True, True),
        ("encoder", False, True),
        ("decoder", True, True),
        # Keep projector BatchNorm/SyncBatchNorm in fp32; FSDP still shards it.
        ("proj", True, False),
        # ("probe", True, False),
    )
    wrapped_modules = []

    for attr_name, reduce_fp32, use_mixed_precision in wrap_specs:
        submodule = getattr(model, attr_name, None)
        if not isinstance(submodule, nn.Module) or isinstance(submodule, FSDP):
            continue
        setattr(
            model,
            attr_name,
            wrap_submodule_fsdp(
                submodule,
                reduce_fp32=reduce_fp32,
                use_mixed_precision=use_mixed_precision,
            ),
        )
        wrapped_modules.append(attr_name)

    return wrapped_modules

def get_fsdp_wrapper(model, use_amp=True): # wrapper entire model

    if use_amp:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,   
            reduce_dtype=torch.float32,  
            buffer_dtype=torch.float32,  
        )
    else:
        mp_policy = None

    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, 
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        limit_all_gathers=True,
    )

    return fsdp_model

def save_fsdp_checkpoint(model, rank, checkpoint_dir, filename):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    
    if rank == 0:
        torch.save(cpu_state, f"{checkpoint_dir}/{filename}")
        print(f"FSDP model saved to {filename}")
