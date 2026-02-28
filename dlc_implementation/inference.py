import yaml
from dlc_implementation.PoseSuperAnimalModel import PoseModel
from dlc_implementation.tranform import build_transforms
from dlc_implementation.preprocessor import build_top_down_preprocessor
from dlc_implementation.postprocessor import build_top_down_postprocessor
import numpy as np
import torch

class PoseInferenceRunner:

    def __init__(self, model_config, model_snapshot_path,device='mps'):

        if isinstance(model_config, str):
            with open(model_config, 'r') as file:
                self.model_config = yaml.safe_load(file)
        elif not isinstance(model_config, dict):
            raise TypeError(f"Parameter `model_config` is not str or dict")
        else:
            self.model_config = model_config
        self.device = device
        
        # Check if the model is an ONNX or TensorRT engine
        if str(model_snapshot_path).endswith('.onnx'):
            import onnxruntime as ort
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(model_snapshot_path, providers=providers)
            self.model = None
            print(f"Loaded ONNX model with providers: {self.ort_session.get_providers()}")
        elif str(model_snapshot_path).endswith('.engine'):
            # Placeholder for direct TensorRT engine support if needed
            raise NotImplementedError("Direct .engine support for Pose DLC requires a custom TRT runner.")
        else:
            self.model = PoseModel.build(self.model_config['model'], snapshot=model_snapshot_path,device=device)
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
        
        if self.model is not None:
            # PyTorch Inference
            with torch.inference_mode():
                inputs = pre_inputs.to(self.device)
                model_outputs = self.model(inputs)
                raw_predictions = self.model.get_predictions(model_outputs)
            
            # Move all predictions to CPU in a single batch operation
            raw_predictions_cpu = {
                head: {
                    pred_name: pred.cpu().detach().numpy()
                    for pred_name, pred in head_outputs.items()
                }
                for head, head_outputs in raw_predictions.items()
            }
        else:
            # ONNX Inference
            # Pre-inputs is a torch tensor (B, C, H, W) from the preprocessor
            inputs_numpy = pre_inputs.numpy()
            input_name = self.ort_session.get_inputs()[0].name
            # The ONNX model should return a list of outputs corresponding to the heads
            ort_outputs = self.ort_session.run(None, {input_name: inputs_numpy})
            
            # Map ONNX outputs back to the expected raw_predictions_cpu format
            # This assumes the ONNX output order matches the expected head/pred names
            # In a real scenario, we'd use output names to map these correctly.
            # For SuperAnimal/DLC, it's typically [heatmap, locref]
            raw_predictions_cpu = {
                "head": {
                    "heatmap": ort_outputs[0],
                    "locref": ort_outputs[1] if len(ort_outputs) > 1 else None
                }
            }

        # Determine batch size from any of the output tensors
        first_head = next(iter(raw_predictions_cpu.values()))
        first_pred = next(iter(first_head.values()))
        if first_pred is None: # Handle case where locref might be missing
             first_pred = next(item for item in first_head.values() if item is not None)
        batch_size = first_pred.shape[0]

        # Re-structure into the list-of-dicts format expected by the post-processor
        predictions = [
            {
                head: {
                    pred_name: head_outputs[pred_name][b] if head_outputs[pred_name] is not None else None
                    for pred_name in head_outputs
                }
                for head, head_outputs in raw_predictions_cpu.items()
            }
            for b in range(batch_size)
        ]

        outputs = self.pose_postprocessor(predictions, updated_context)

        return outputs



