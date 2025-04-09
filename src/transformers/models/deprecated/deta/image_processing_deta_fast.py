# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Image processor class for DeTa."""

from typing import Dict, List, Optional, Union

from ....image_processing_utils import BatchFeature
from ....image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    SizeDict,
)
from ....image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    AnnotationFormat,
    ChannelDimension,
    PILImageResampling,
)
from ....utils import TensorType, add_start_docstrings, is_torch_available, is_torchvision_available, logging


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms import functional as F


SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


class DetaFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Additional kwargs for DetaImageProcessorFast.

    Args:
        do_pad (`bool`, *optional*):
            Whether to pad the image. If `True`, padding will be applied to the bottom and right of
            the image with zeros. If `pad_size` is provided, the image will be padded to the specified
            dimensions. Otherwise, the image will be padded to the maximum height and width of the batch.
        pad_size (`Dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
            provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
            height and width in the batch.
        format (`str`, *optional*):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_convert_annotations (`bool`, *optional*):
            Whether to convert the annotations to the format expected by the model. Converts the bounding
            boxes from the format `(top_left_x, top_left_y, width, height)` to `(center_x, center_y, width, height)`
            and in relative coordinates.
        return_segmentation_masks (`bool`, *optional*):
            Whether to return segmentation masks.
    """

    do_pad: Optional[bool]
    pad_size: Optional[Dict[str, int]]
    format: Optional[str]
    do_convert_annotations: Optional[bool]
    return_segmentation_masks: Optional[bool]


@add_start_docstrings(
    "Constructs a fast DeTa image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class DetaImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast DeTa image processor.

    This processor inherits from [`BaseImageProcessorFast`] which contains most of the main methods. Users should refer
    to this superclass for more information regarding those methods.
    """

    valid_kwargs = DetaFastImageProcessorKwargs

    # Default values from the slow image processor
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    format = "coco_detection"
    do_convert_annotations = True
    return_segmentation_masks = False

    def __init__(
        self,
        do_resize=True,
        size=None,
        resample=PILImageResampling.BILINEAR,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_pad=True,
        pad_size=None,
        format="coco_detection",
        do_convert_annotations=True,
        return_segmentation_masks=False,
        **kwargs,
    ):
        super().__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.format = format
        self.do_convert_annotations = do_convert_annotations
        self.return_segmentation_masks = return_segmentation_masks

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool = None,
        size: SizeDict = None,
        resample: Optional["F.InterpolationMode"] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = None,
        pad_size: Optional[Dict[str, int]] = None,
        annotations: Optional[List[Dict]] = None,
        do_convert_annotations: bool = None,
        format: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.

        Args:
            images (`List[torch.Tensor]`):
                List of images to preprocess.
            do_resize (`bool`, *optional*):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*):
                Size of the image after resizing.
            resample (`F.InterpolationMode`, *optional*):
                Resampling filter to use when resizing the image.
            do_rescale (`bool`, *optional*):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*):
                Mean to use when normalizing the image.
            image_std (`float` or `List[float]`, *optional*):
                Standard deviation to use when normalizing the image.
            do_pad (`bool`, *optional*):
                Whether to pad the image.
            pad_size (`Dict[str, int]`, *optional*):
                Size to pad the image to.
            annotations (`List[Dict]`, *optional*):
                List of annotations in COCO format.
            do_convert_annotations (`bool`, *optional*):
                Whether to convert the annotations to the format expected by the model.
            format (`str`, *optional*):
                Format of the annotations.
            return_tensors (`str` or `TensorType`, *optional*):
                Type of tensors to return.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.

        Returns:
            `BatchFeature`: A `BatchFeature` with the following fields:
                - **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
                    Pixel values normalized and resized to the model's expected input size.
                - **pixel_mask** (`torch.Tensor` of shape `(batch_size, height, width)`):
                    Pixel mask indicating which pixels are padding (0) and which are not (1).
                - **labels** (`Dict[str, torch.Tensor]`, *optional*):
                    Labels for the image, if annotations are provided.
        """
        # Process images
        processed_images = []
        original_sizes = []
        target_sizes = []
        pixel_masks = []

        for image in images:
            # Get original size
            original_size = {"height": image.shape[-2], "width": image.shape[-1]}
            original_sizes.append(original_size)

            # Resize image
            if do_resize:
                image = self.resize(image, size=size, resample=resample)

            # Get target size
            target_size = {"height": image.shape[-2], "width": image.shape[-1]}
            target_sizes.append(target_size)

            # Rescale and normalize image
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            processed_images.append(image)

        # Pad images if needed
        if do_pad:
            if pad_size is not None:
                # Use provided pad_size
                for i, image in enumerate(processed_images):
                    processed_images[i] = self.pad_image(image, pad_size)
                    # Create pixel mask
                    pixel_mask = torch.ones((pad_size["height"], pad_size["width"]), dtype=torch.bool)
                    h, w = target_sizes[i]["height"], target_sizes[i]["width"]
                    pixel_mask[h:, :] = 0
                    pixel_mask[:, w:] = 0
                    pixel_masks.append(pixel_mask)
            else:
                # Pad to the largest image in the batch
                max_height = max(target_size["height"] for target_size in target_sizes)
                max_width = max(target_size["width"] for target_size in target_sizes)
                pad_size = {"height": max_height, "width": max_width}

                for i, image in enumerate(processed_images):
                    processed_images[i] = self.pad_image(image, pad_size)
                    # Create pixel mask
                    pixel_mask = torch.ones((pad_size["height"], pad_size["width"]), dtype=torch.bool)
                    h, w = target_sizes[i]["height"], target_sizes[i]["width"]
                    pixel_mask[h:, :] = 0
                    pixel_mask[:, w:] = 0
                    pixel_masks.append(pixel_mask)

        # Stack images and pixel masks
        data = {"pixel_values": torch.stack(processed_images, dim=0)}
        if do_pad:
            data["pixel_mask"] = torch.stack(pixel_masks, dim=0)

        # Process annotations if provided
        if annotations is not None and do_convert_annotations:
            labels = []
            for i, annotation in enumerate(annotations):
                if format == "coco_detection":
                    # Convert COCO detection annotations
                    if "annotations" in annotation:
                        annotation_data = annotation["annotations"]
                    else:
                        annotation_data = annotation
                    label = self.convert_coco_annotations_to_tensor_format(
                        annotation_data, original_sizes[i], target_sizes[i]
                    )
                    labels.append(label)
                elif format == "coco_panoptic":
                    # For panoptic segmentation, we would need to implement additional logic
                    # This is a placeholder for future implementation
                    labels.append({})
                else:
                    labels.append({})

            # Add labels to data
            if len(labels) > 0:
                data["labels"] = labels

        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)
        return encoded_inputs

    def pad_image(
        self, image: "torch.Tensor", pad_size: Dict[str, int], data_format: Optional[Union[str, ChannelDimension]] = None
    ) -> "torch.Tensor":
        """
        Pad an image to the specified size.

        Args:
            image (`torch.Tensor`):
                Image to pad.
            pad_size (`Dict[str, int]`):
                Size to pad to.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.

        Returns:
            `torch.Tensor`: Padded image.
        """
        output_height, output_width = pad_size["height"], pad_size["width"]
        input_height, input_width = image.shape[-2], image.shape[-1]

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width

        # Ensure padding is non-negative
        pad_bottom = max(0, pad_bottom)
        pad_right = max(0, pad_right)

        # Apply padding
        padded_image = F.pad(image, [0, pad_right, 0, pad_bottom], fill=0)

        return padded_image

    def convert_coco_annotations_to_tensor_format(
        self, annotations: List[Dict], original_size: Dict[str, int], target_size: Dict[str, int]
    ) -> Dict[str, "torch.Tensor"]:
        """
        Convert COCO annotations to tensor format.

        Args:
            annotations (`List[Dict]`):
                List of annotations in COCO format.
            original_size (`Dict[str, int]`):
                Original size of the image as a dictionary with keys "height" and "width".
            target_size (`Dict[str, int]`):
                Target size of the image as a dictionary with keys "height" and "width".

        Returns:
            `Dict[str, torch.Tensor]`: Annotations in tensor format.
        """
        if len(annotations) == 0:
            return {}

        # Convert to tensor format
        boxes = []
        classes = []
        areas = []
        for annotation in annotations:
            # Convert bbox from [x, y, w, h] to [center_x, center_y, w, h] and normalize
            box = annotation["bbox"]
            box_center_x = box[0] + box[2] / 2
            box_center_y = box[1] + box[3] / 2
            box_w = box[2]
            box_h = box[3]

            # Normalize coordinates
            box_center_x /= original_size["width"]
            box_center_y /= original_size["height"]
            box_w /= original_size["width"]
            box_h /= original_size["height"]

            boxes.append([box_center_x, box_center_y, box_w, box_h])
            classes.append(annotation["category_id"])
            areas.append(annotation["area"])

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32)

        return {"boxes": boxes, "class_labels": classes, "area": areas}


__all__ = ["DetaImageProcessorFast"]
