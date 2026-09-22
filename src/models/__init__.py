from .backbone import build_backbone

from ..utils.checkpoint import load_checkpoint

from importlib import import_module

# Inference should not import every optional pretraining method.
_METHODS = {
    'dino': ('DINO', 'DINO_VIT'), 'byol': ('BYOL', 'BYOLModel'),
    'lejepa': ('LeJEPA_model', 'LEJEPA_VIT'),
    'lejepa_sliced_prior': ('LeJEPASlicedPrior', 'LEJEPASlicedPrior_VIT'),
    'leworld': ('LeWorldModel', 'LeWorldModel'),
    'lpjepa': ('LpJEPA', 'RectifiedLpJEPA'), 'ijepa': ('IJEPA', 'IJEPAModel'),
    'mae': ('MAE', 'MAE_VIT'), 'simmim': ('SimMIM', 'SimMIM_VIT'),
    'simclr': ('SimCLR', 'SimCLRModel'), 'simsiam': ('SimSiam', 'SimSiamModel'),
    'vicreg': ('VICReg', 'VICRegModel'), 'swav': ('SwAV', 'SwAVModel'),
}


def __getattr__(name):
    for module, class_name in _METHODS.values():
        if name == class_name:
            return getattr(import_module('.' + module, __name__), class_name)
    raise AttributeError(name)


import logging
logger = logging.getLogger("neurojepa")


def build_pretrain_model(args):
    if args.model_chose not in _METHODS:
        raise ValueError(f"Unsupported model_chose: {args.model_chose}")
    module, class_name = _METHODS[args.model_chose]
    return getattr(import_module('.' + module, __name__), class_name)(args)


def create_model(config):
    task_config = config.get('task', {})
    exp_config = config['experiment']

    model_config = config['model']
    pretrained_checkpoint_path = exp_config.get('pretrained_checkpoint', None)
    skip_bias_in_pretrained = bool(exp_config.get('skip_bias_in_pretrained', False))
    num_classes = int(task_config.get('num_classes', model_config.get('num_classes', 1)))

    model = build_backbone(config, downstream=True, num_classes=num_classes)

    defer_pretrained_load = bool(exp_config.get('defer_pretrained_load', False))
    if pretrained_checkpoint_path and not defer_pretrained_load:
        load_checkpoint(
            checkpoint_path=pretrained_checkpoint_path,
            model=model,
            mode="pretrained",
            target="backbone",
            strict=False,
            skip_bias=skip_bias_in_pretrained,
        )
    elif pretrained_checkpoint_path and defer_pretrained_load:
        logger.info("Deferring pretrained checkpoint load for caller-managed distributed loading: %s", pretrained_checkpoint_path)
    else:
        logger.error("No pretrained checkpoint specified. Initializing model with random weights.")

    return model
