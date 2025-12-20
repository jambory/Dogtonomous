from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, Callable
import albumentations as A
import numpy as np
import torch
import cv2
Image = TypeVar("Image", torch.Tensor, np.ndarray, str, Path)
Context = TypeVar("Context", dict[str, Any], None)

class Preprocessor(ABC):
    """
    Class to preprocess an image and turn it into a batch of inputs before running
    inference.

    As an example, a pre-processor can load an image, use a "bboxes" key from context
    to crop bounding boxes for individuals (going from a (h, w, 3) array to a
    (num_individuals, h, w, 3) array), and convert it into a tensor ready for inference.
    """

    @abstractmethod
    def __call__(self, image: Image, context: Context) -> tuple[Image, Context]:
        """Pre-processes an image

        Args:
            image: an image (containing height, width and channel dimensions) or a
                batch of images linked to a single input (containing an extra batch
                dimension)
            context: the context for this image or batch of images (such as bounding
                boxes, conditional pose, ...)

        Returns:
            the pre-processed image (or batch of images) and their context
        """
        pass

def build_top_down_preprocessor(
    color_mode: str,
    transform: A.BaseCompose,
    top_down_crop_size: tuple[int, int],
    top_down_crop_margin: int = 0,
    top_down_crop_with_context: bool = True,
):
    """Creates a preprocessor for top-down pose estimation

    Creates a preprocessor that loads an image, crops all bounding boxes given as a
    context (through a "bboxes" key), runs some transforms on each cropped image (such
    as normalization), creates a tensor from the numpy array (going from
    (num_ind, h, w, 3) to (num_ind, 3, h, w)).

    Args:
        color_mode: whether to load the image as an RGB or BGR
        transform: the transform to apply to the image
        top_down_crop_size: the (width, height) to resize cropped bboxes to
        top_down_crop_margin: the margin to add around detected bboxes for the crop
        top_down_crop_with_context: whether to keep context when applying the top-down crop

    Returns:
        A default top-down Preprocessor
    """
    return ComposePreprocessor(
        components=[
            LoadImage(color_mode),
            TopDownCrop(
                output_size=top_down_crop_size,
                margin=top_down_crop_margin,
                with_context=top_down_crop_with_context,
            ),
            AugmentImage(transform),
            ToTensor(),
        ]
    )

class ComposePreprocessor(Preprocessor):
    """
    Class to preprocess an image and turn it into a batch of
    inputs before running inference
    """

    def __init__(self, components: list[Preprocessor]) -> None:
        self.components = components

    def __call__(self, image: Image, context: Context) -> tuple[Image, Context]:
        for preprocessor in self.components:
            image, context = preprocessor(image, context)
        return image, context

class LoadImage(Preprocessor):
    """Loads an image from a file, if not yet loaded"""

    def __init__(self, color_mode: str = "RGB") -> None:
        self.color_mode = color_mode

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        if isinstance(image, (str, Path)):
            image = load_image(image, color_mode=self.color_mode)

        h, w = image.shape[:2]
        context["image_size"] = w, h
        return image, context

class AugmentImage(Preprocessor):
    """

    Adds an offset and scale key to the context:
        offset: (x, y) position of the pixel in the top left corner of the augmented
            image in the original image
        scale: size of the original image divided by the size of the new image

    This allows to map the position of predictions in the transformed image back to the
    original image space.
        p_original = p_transformed * scale + offset
        p_transformed = (p_original - offset) / scale
    """

    def __init__(self, transform: A.BaseCompose) -> None:
        self.transform = transform

    @staticmethod
    def get_offsets_and_scales(
        h: int,
        w: int,
        output_bboxes: list[tuple[float, float, float, float]],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        offsets, scales = [], []
        for bbox in output_bboxes:
            x_origin, y_origin, w_out, h_out = bbox
            x_scale, y_scale = w / w_out, h / h_out
            x_offset = -x_origin * x_scale
            y_offset = -y_origin * y_scale
            offsets.append((x_offset, y_offset))
            scales.append((x_scale, y_scale))

        return offsets, scales

    @staticmethod
    def update_offset(
        offset: tuple[float, float],
        scale: tuple[float, float],
        new_offset: tuple[float, float],
    ) -> tuple[float, float]:
        return (
            scale[0] * new_offset[0] + offset[0],
            scale[1] * new_offset[1] + offset[1],
        )

    @staticmethod
    def update_scale(
        scale: tuple[float, float], new_scale: tuple[float, float]
    ) -> tuple[float, float]:
        return scale[0] * new_scale[0], scale[1] * new_scale[1]

    @staticmethod
    def update_offsets_and_scales(context, new_offsets, new_scales) -> tuple:
        """
        x = x' * scale' + offset'
        x' = x'' * scale'' + offset''
        -> x = x'' * (scale' * scale'') + (scale' * offset'' + offset')
        """
        # scales and offsets are either both lists or both tuples
        offsets = context.get("offsets", (0, 0))
        scales = context.get("scales", (1, 1))
        if isinstance(offsets, tuple):
            if isinstance(new_offsets, list):
                updated_offsets = [
                    AugmentImage.update_offset(offsets, scales, new_offset)
                    for new_offset in new_offsets
                ]
                updated_scales = [
                    AugmentImage.update_scale(scales, new_scale)
                    for new_scale in new_scales
                ]
            else:
                if not len(offsets) == len(new_offsets):
                    raise ValueError("Cannot rescale lists when not same length")

                updated_offsets = AugmentImage.update_offset(
                    offsets, scales, new_offsets
                )
                updated_scales = AugmentImage.update_scale(scales, new_scales)
        else:
            if isinstance(new_offsets, list):
                if not len(offsets) == len(new_offsets):
                    raise ValueError("Cannot rescale lists when not same length")

                updated_offsets = [
                    AugmentImage.update_offset(offset, scale, new_offset)
                    for offset, scale, new_offset in zip(offsets, scales, new_offsets)
                ]
                updated_scales = [
                    AugmentImage.update_scale(scale, new_scale)
                    for scale, new_scale in zip(scales, new_scales)
                ]
            else:
                updated_offsets = [
                    AugmentImage.update_offset(offset, scale, new_offsets)
                    for offset, scale in zip(offsets, scales)
                ]
                updated_scales = [
                    AugmentImage.update_scale(scale, new_scales) for scale in scales
                ]
        return updated_offsets, updated_scales

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        # If the image is a batch, process each entry
        if len(image.shape) == 4:
            batch_size, h, w, _ = image.shape
            if batch_size == 0:
                # no images in top-down when no detections
                offsets, scales = (0, 0), (1, 1)
            else:
                transformed = [
                    self.transform(
                        image=img,
                        keypoints=[],
                        class_labels=[],
                        bboxes=[[0, 0, w, h]],
                        bbox_labels=["image"],
                    )
                    for img in image
                ]
                image = np.stack([t["image"] for t in transformed])
                output_bboxes = [t["bboxes"][0] for t in transformed]
                offsets, scales = self.get_offsets_and_scales(h, w, output_bboxes)
        else:
            h, w, _ = image.shape
            transformed = self.transform(
                image=image,
                keypoints=[],
                class_labels=[],
                bboxes=[[0, 0, w, h]],
                bbox_labels=["image"],
            )
            image = transformed["image"]
            output_bboxes = [transformed["bboxes"][0]]
            offsets, scales = self.get_offsets_and_scales(h, w, output_bboxes)
            offsets = offsets[0]
            scales = scales[0]

        offsets, scales = self.update_offsets_and_scales(context, offsets, scales)
        context["offsets"] = offsets
        context["scales"] = scales
        return image, context


class ToTensor(Preprocessor):
    """Transforms lists and numpy arrays into tensors"""

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        image = torch.tensor(image, dtype=torch.float)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        else:
            image = image.permute(2, 0, 1)
        return image, context

class TopDownCrop(Preprocessor):
    """Crops bounding boxes out of images for top-down pose estimation

    Args:
        output_size: The (width, height) of crops to output
        margin: The margin to add around detected bounding boxes before cropping
        with_context: Whether to keep context in the top-down crop
    """

    def __init__(
        self,
        output_size: int | tuple[int, int],
        margin: int = 0,
        with_context: bool = True,
    ) -> None:
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.output_size = output_size
        self.margin = margin
        self.with_context = with_context

    def __call__(
        self, image: np.ndarray, context: Context
    ) -> tuple[np.ndarray, Context]:
        """TODO: numpy implementation"""
        if "bboxes" not in context:
            raise ValueError(f"Must include bboxes to CropDetections, found {context}")

        images, offsets, scales = [], [], []
        for bbox in context["bboxes"]:
            crop, offset, scale = top_down_crop(
                image,
                bbox,
                self.output_size,
                margin=self.margin,
                crop_with_context=self.with_context,
            )
            images.append(crop)
            offsets.append(offset)
            scales.append(scale)

        context["offsets"] = np.array(offsets)
        context["scales"] = np.array(scales)

        # can have no bounding boxes if detector made no detections
        if len(images) == 0:
            images = np.zeros((0, *image.shape))
        else:
            images = np.stack(images, axis=0)

        context["top_down_crop_size"] = self.output_size
        return images, context

if __name__ == "__main__":
    import yaml
    model_cfg_path = "/Users/cobyw/Documents/projects/DeepLab/projects/new_bodyparts_project-cwilcox-2025-11-30/dlc-models-pytorch/iteration-0/new_bodyparts_projectNov30-trainset95shuffle4/train/pytorch_config.yaml"
    with open(model_cfg_path, 'r') as file:
        model_config = yaml.safe_load(file)

    crop_cfg = model_config["data"]["inference"].get("top_down_crop", {})
    width, height = crop_cfg.get("width", 256), crop_cfg.get("height", 256)
    margin = crop_cfg.get("margin", 0)
    from dlc_implementation.tranform import build_transforms
    transform = build_transforms(model_config["data"]["inference"])

    pose_preprocessor = build_top_down_preprocessor(
        color_mode=model_config["data"]["colormode"],
        transform=transform,
        top_down_crop_size=(width, height),
        top_down_crop_margin=margin,
        top_down_crop_with_context=crop_cfg.get("crop_with_context", True),
    )

def load_image(filepath: str | Path, color_mode: str = "RGB") -> np.ndarray:
    """Loads an image from a file using cv2

    Args:
        filepath: the path of the file containing the image to load
        color_mode: {'RGB', 'BGR'} the color mode to load the image with

    Returns:
        the image as a numpy array
    """
    image = cv2.imread(str(filepath))
    if color_mode == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif not color_mode == "BGR":
        raise ValueError(f"Unsupported `color_mode`: {color_mode}")

    return image

def top_down_crop(
    image: np.ndarray,
    bbox: np.ndarray,
    output_size: tuple[int, int],
    margin: int = 0,
    center_padding: bool = False,
    crop_with_context: bool = True,
) -> tuple[np.array, tuple[int, int], tuple[float, float]]:
    """
    Crops images around bounding boxes for top-down pose estimation. Computes offsets so
    that coordinates in the original image can be mapped to the cropped one;

        x_cropped = (x - offset_x) / scale_x
        x_cropped = (y - offset_y) / scale_y

    Bounding boxes are expected to be in COCO-format (xywh).

    Args:
        image: (h, w, c) the image to crop
        bbox: (4,) the bounding box to crop around
        output_size: the (width, height) of the output cropped image
        margin: a margin to add around the bounding box before cropping
        center_padding: whether to center the image in the padding if any is needed
        crop_with_context: Whether to keep context around the bounding box when cropping

    Returns:
        cropped_image, (offset_x, offset_y), (scale_x, scale_y)
    """
    image_h, image_w, c = image.shape
    out_w, out_h = output_size
    x, y, w, h = bbox

    cx = x + w / 2
    cy = y + h / 2
    w += 2 * margin
    h += 2 * margin

    if crop_with_context:
        input_ratio = w / h
        output_ratio = out_w / out_h
        if input_ratio > output_ratio:  # h/w < h0/w0 => h' = w * h0/w0
            h = w / output_ratio
        elif input_ratio < output_ratio:  # w/h < w0/h0 => w' = h * w0/h0
            w = h * output_ratio

    # cx,cy,w,h will now give the right ratio -> check if padding is needed
    x1, y1 = int(round(cx - (w / 2))), int(round(cy - (h / 2)))
    x2, y2 = int(round(cx + (w / 2))), int(round(cy + (h / 2)))

    # pad symmetrically - compute total padding across axis
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if x1 < 0:
        pad_left = -x1
        x1 = 0
    if x2 > image_w:
        pad_right = x2 - image_w
        x2 = image_w
    if y1 < 0:
        pad_top = -y1
        y1 = 0
    if y2 > image_h:
        pad_bottom = y2 - image_h
        y2 = image_h

    w, h = x2 - x1, y2 - y1
    if not crop_with_context:
        input_ratio = w / h
        output_ratio = out_w / out_h
        if input_ratio > output_ratio:  # h/w < h0/w0 => h' = w * h0/w0
            w_pad = int(w - h * output_ratio) // 2
            pad_top += w_pad
            pad_bottom += w_pad

        elif input_ratio < output_ratio:  # w/h < w0/h0 => w' = h * w0/h0
            h_pad = int(h - (w / output_ratio)) // 2
            pad_left += h_pad
            pad_right += h_pad

    pad_x = pad_left + pad_right
    pad_y = pad_top + pad_bottom
    if center_padding:
        pad_left = pad_x // 2
        pad_top = pad_y // 2

    # crop the pixels we care about
    image_crop = np.zeros((h + pad_y, w + pad_x, c), dtype=image.dtype)
    image_crop[pad_top:pad_top + h, pad_left:pad_left + w] = image[y1:y2, x1:x2]

    # resize the cropped image
    image = cv2.resize(image_crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # compute scale and offset
    offset = x1 - pad_left, y1 - pad_top
    scale = (w + pad_x) / out_w, (h + pad_y) / out_h
    return image, offset, scale