import torch
import torch.nn as nn
import copy
import logging
from dlc_implementation.model_dependencies import (
    WeightInitialization
)
from dlc_implementation.models.backbones import HRNet, BaseBackbone
from dlc_implementation.models.criterions import WeightedMSECriterion, WeightedHuberCriterion, WeightedLossAggregator
from dlc_implementation.models.generators import HeatmapGaussianGenerator
from dlc_implementation.models.predictor import HeatmapPredictor
from dlc_implementation.models.heads import HeatmapHead, BaseHead

class PoseModel(nn.Module):
    """A pose estimation model

    A slight change on deeplabcut's `model.py`, mainly just that the model always sets the neck
    to None, as the model I am using `SuperAnimal` does not need it and I just want to test this
    a physical device.

    A pose estimation model is composed of a backbone, optionally a neck, and an
    arbitrary number of heads. Outputs are computed as follows:
    """

    def __init__(
            self,
            cfg: dict,
            backbone: BaseBackbone,
            heads: dict[str, BaseHead],
            neck: None = None,
    ) -> None:
        """
        Args:
            cfg: configuration dictionary for the model.
            backbone: backbone network architecture.
            heads: the heads for the model
            neck: neck network architecture (default is None). Defaults to None.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.neck = neck
        self.output_features = False

        self._strides = {
            name: _model_stride(self.backbone.stride, head.stride)
            for name, head in heads.items()
        }

    def forward(self, x: torch.Tensor, **backbone_kwargs) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass of the PoseModel.

        Args:
            x: input images

        Returns:
            Outputs of head groups
        """
        if x.dim() == 3:
            x = x[None, :]
        features = self.backbone(x, **backbone_kwargs)
        if self.neck:
            features = self.neck(features)

        outputs = {}
        if self.output_features:
            outputs["backbone"] = dict(features=features)

        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)
        return outputs

    def get_loss(
            self,
            outputs: dict[str, dict[str, torch.Tensor]],
            targets: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        total_losses = []
        losses: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            head_losses = head.get_loss(outputs[name], targets[name])
            total_losses.append(head_losses["total_loss"])
            for k, v in head_losses.items():
                losses[f"{name}_{k}"] = v

        # TODO: Different aggregation for multi-head loss?
        losses["total_loss"] = torch.mean(torch.stack(total_losses))
        return losses

    def get_target(
            self,
            outputs: dict[str, dict[str, torch.Tensor]],
            labels: dict,
    ) -> dict[str, dict]:
        """Summary:
        Get targets for model training.

        Args:
            outputs: output of each head group
            labels: dictionary of labels

        Returns:
            targets: dict of the targets for each model head group
        """
        return {
            name: head.target_generator(self._strides[name], outputs[name], labels)
            for name, head in self.heads.items()
        }

    def get_predictions(self, outputs: dict[str, dict[str, torch.Tensor]]) -> dict:
        """Abstract method for the forward pass of the Predictor.

        Args:
            outputs: outputs of the model heads

        Returns:
            A dictionary containing the predictions of each head group
        """
        predictions = {
            name: head.predictor(self._strides[name], outputs[name])
            for name, head in self.heads.items()
        }
        if self.output_features:
            predictions["backbone"] = outputs["backbone"]

        return predictions

    def get_stride(self, head: str) -> int:
        """
        Args:
            head: The head for which to get the total stride.

        Returns:
            The total stride for the outputs of the head.

        Raises:
            ValueError: If there is no such head.
        """
        return self._strides[head]

    @staticmethod
    def build(
            cfg: dict,
            weight_init: None | WeightInitialization = None,
            pretrained_backbone: bool = False,
            snapshot:str=None,
            device:str="cpu"
    ) -> "PoseModel":
        """
        Args:
            cfg: The configuration of the model to build.
            weight_init: How model weights should be initialized. If None, ImageNet
                pre-trained backbone weights are loaded from Timm.
            pretrained_backbone: Whether to load an ImageNet-pretrained weights for
                the backbone. This should only be set to True when building a model
                which will be trained on a transfer learning task.
            snapshot: path to model snapshot to load in.

        Returns:
            the built pose model
        """
        cfg["backbone"]["pretrained"] = pretrained_backbone
        args = cfg['backbone'].copy()
        if args.pop('type') != "HRNet":
            raise Exception("Config passed does not call for HRNEt backbone.")
        backbone = HRNet(**args)

        heads = {}
        for name, head_cfg in cfg["heads"].items():
            head_cfg = copy.deepcopy(head_cfg)
            weights = {}
            criterions = {}
            for loss_name, criterion_cfg in head_cfg["criterion"].items():
                weights[loss_name] = criterion_cfg.get("weight", 1.0)
                criterion_cfg = {
                    k: v for k, v in criterion_cfg.items() if k != "weight"
                }
                args = criterion_cfg.copy()
                criterion_type = args.pop('type')
                match criterion_type:
                    case 'WeightedMSECriterion':
                        criterion = WeightedMSECriterion()
                    case 'WeightedHuberCriterion':
                        criterion = WeightedHuberCriterion()
                    case _:
                        raise ValueError(f"Can't find loss type {criterion_type}")
                criterions[loss_name] = criterion

                aggregator_cfg = {"type": "WeightedLossAggregator", "weights": weights}
                aggregator_type = aggregator_cfg.pop('type')
                match aggregator_type:
                    case 'WeightedLossAggregator':
                        aggregator = WeightedLossAggregator(**aggregator_cfg)
                    case _:
                        raise ValueError(f"Can't find aggregator type {aggregator_type}")
                head_cfg["aggregator"] = aggregator
                head_cfg["criterion"] = criterions

            args = head_cfg["target_generator"].copy()
            generator_type = args.pop('type')
            if generator_type != 'HeatmapGaussianGenerator':
                raise ValueError(f"Can't find generator type: {generator_type}")
            head_cfg["target_generator"] = HeatmapGaussianGenerator(**args)

            args = head_cfg["predictor"].copy()
            predictor_type = args.pop('type')
            if predictor_type != 'HeatmapPredictor':
                raise ValueError(f"Can't find generator type: {predictor_type}")
            head_cfg["predictor"] = HeatmapPredictor(**args)

            print(head_cfg)
            args = head_cfg.copy()
            head_type = args.pop('type')
            if head_type != 'HeatmapHead':
                raise ValueError(f"Can't find head type: {head_type}")
            heads[name] = HeatmapHead(**args)

        model = PoseModel(cfg=cfg, backbone=backbone, neck=None, heads=heads)

        if weight_init is not None:
            logging.info(f"Loading pretrained model weights: {weight_init}")
            logging.info(f"The pose model is loading from {weight_init.snapshot_path}")
            snapshot = torch.load(weight_init.snapshot_path, map_location=device)
            state_dict = snapshot["model"]

            # load backbone state dict
            model.backbone.load_state_dict(filter_state_dict(state_dict, "backbone"))

            # if there's a neck, load state dict
            if model.neck is not None:
                model.neck.load_state_dict(filter_state_dict(state_dict, "neck"))

            # load head state dicts
            if weight_init.with_decoder:
                all_head_state_dicts = filter_state_dict(state_dict, "heads")
                conversion_tensor = torch.from_numpy(weight_init.conversion_array)
                for name, head in model.heads.items():
                    head_state_dict = filter_state_dict(all_head_state_dicts, name)

                    # requires WeightConversionMixin
                    if not weight_init.memory_replay:
                        head_state_dict = head.convert_weights(
                            state_dict=head_state_dict,
                            module_prefix="",
                            conversion=conversion_tensor,
                        )

                    head.load_state_dict(head_state_dict)

        if snapshot is not None:
            model.load_state_dict(torch.load(snapshot, map_location=device,weights_only=True)['model'])

        return model


def filter_state_dict(state_dict: dict, module: str) -> dict[str, torch.Tensor]:
    """
    Filters keys in the state dict for a module to only keep a given prefix. Removes
    the module from the keys (e.g. for module="backbone", "backbone.stage1.weight" will
    be converted to "stage1.weight" so the state dict can be loaded into the backbone
    directly).

    Args:
        state_dict: the state dict
        module: the module to keep, e.g. "backbone"

    Returns:
        the filtered state dict, with the module removed from the keys

    Examples:
        state_dict = {"backbone.conv.weight": t1, "head.conv.weight": t2}
        filtered = filter_state_dict(state_dict, "backbone")
        # filtered = {"conv.weight": t1}
        model.backbone.load_state_dict(filtered)
    """
    return {
        ".".join(k.split(".")[1:]): v  # remove 'backbone.' from the keys
        for k, v in state_dict.items()
        if k.startswith(module)
    }


def _model_stride(backbone_stride: int | float, head_stride: int | float) -> float:
    """Computes the model stride from a backbone and a head"""
    if head_stride > 0:
        return backbone_stride / head_stride

    return backbone_stride * -head_stride
