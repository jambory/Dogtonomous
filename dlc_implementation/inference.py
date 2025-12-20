import yaml
from dlc_implementation.PoseSuperAnimalModel import PoseModel
from dlc_implementation.tranform import build_transforms
from dlc_implementation.preprocessor import build_top_down_preprocessor
from dlc_implementation.postprocessor import build_top_down_postprocessor
import numpy as np
import torch

class PoseInferenceRunner:

    def __init__(self, model_config, model_snapshot_path, device='mps'):

        if isinstance(model_config, str):
            with open(model_config, 'r') as file:
                self.model_config = yaml.safe_load(file)
        elif not isinstance(model_config, dict):
            raise TypeError(f"Parameter `model_config` is not str or dict")
        else:
            self.model_config = model_config
        self.device = device
        self.model = PoseModel.build(self.model_config['model'], snapshot=model_snapshot_path)
        self.model.to(device)
        self.model.eval()

        transform = build_transforms(self.model_config["data"]["inference"])
        crop_cfg = self.model_config["data"]["inference"].get("top_down_crop", {})
        width, height = crop_cfg.get("width", 256), crop_cfg.get("height", 256)
        margin = crop_cfg.get("margin", 0)

        self.pose_preprocessor = build_top_down_preprocessor(
            color_mode=self.model_config["data"]["colormode"],
            transform=transform,
            top_down_crop_size=(width, height),
            top_down_crop_margin=margin,
            top_down_crop_with_context=crop_cfg.get("crop_with_context", True),
        )

        max_individuals = len(self.model_config["metadata"]["individuals"])
        num_bodyparts = len(self.model_config["metadata"]["bodyparts"])
        num_unique_bodyparts = len(self.model_config["metadata"]["unique_bodyparts"])

        self.pose_postprocessor = build_top_down_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique_bodyparts,
        )

    def inference(self, context, frame):
        pre_inputs, updated_context = self.pose_preprocessor(frame, context)
        with torch.inference_mode():
            inputs = pre_inputs.to(self.device)
            model_outputs = self.model(inputs)
            raw_predictions = self.model.get_predictions(model_outputs)

        batch_size = len(model_outputs)
        predictions = [
            {
                head: {
                    pred_name: pred[b].cpu().detach().numpy()
                    for pred_name, pred in head_outputs.items()
                }
                for head, head_outputs in raw_predictions.items()
            }
            for b in range(batch_size)
        ]

        outputs = self.pose_postprocessor(predictions, updated_context)

        return outputs



