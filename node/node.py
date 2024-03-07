import torch
from PIL import Image
import numpy as np

from typing import List, Union

def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
    batch_count = tensor.size(0) if len(tensor.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2np(tensor[i]))
        return out

    return [np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)]

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ColorListMaskToImage:
    """Converts a mask (alpha) to an RGB image with a color and background"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "colorlist": ("STR",),
                "background": ("COLOR", {"default": "#000000"}),
            }
        }

    CATEGORY = "meshmesh"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_mask"
    OUTPUT_NODE = True

    idx = 0
    def select_next_color(self, colorlist):
        print('colorlist', colorlist)
        colors = colorlist["string"].split(",")
        print('colors', colors)
        color = colors[self.idx]
        if self.idx == len(colors) - 1:
            self.idx = 0
        else:
            self.idx += 1
        return color

    def render_mask(self, mask, colorlist, background):
        
        masks = tensor2np(mask)
        images = []
        print('colorlist', colorlist)
        
        idx = 0
        for m in masks:
            _mask = Image.fromarray(m).convert("L")
            color = self.select_next_color(colorlist)
            print('selected color', color)
            image = Image.new("RGBA", _mask.size, color=color)
            # apply the mask
            image = Image.composite(
                image, Image.new("RGBA", _mask.size, color=background), _mask
            )

            images.append(image.convert("RGB"))


        return (pil2tensor(images),)

class FlattenAndCombineMaskImages:
    """Flattens/combines mask images from ColorListMaskToImage"""
    def __init__(self):
        self.images = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1})
            }
        }

    CATEGORY = "meshmesh"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    def alpha_composite(self,bottom, top):
        # Assumes both tensors are of shape [C, H, W] with C=4 (RGBA)
        alpha_top = top[3:4]
        alpha_bottom = bottom[3:4]
        composite = (alpha_top * top[:3] + alpha_bottom * (1 - alpha_top) * bottom[:3]) / (alpha_top + alpha_bottom * (1 - alpha_top))
        composite_alpha = alpha_top + alpha_bottom * (1 - alpha_top)
        return torch.cat((composite, composite_alpha), dim=0)

    def run(self, image, count):
        self.images.append(image)
        
        # Check if we've received all images
        if len(self.images) == count: # All images received
            composite = self.images[0].squeeze(0)  # Remove the batch dimension
            for _image in self.images[1:]:
                _image_squeezed = _image.squeeze(0)  # Remove the batch dimension
                composite = self.alpha_composite(composite, _image_squeezed)

            # Reset images for the next batch
            self.images = []

            # Return the composite image
            return (composite,)
        else:
            # Return None or appropriate signal to indicate waiting for more images
            return (None,)


NODE_CLASS_MAPPINGS = {
    "ColorListMaskToImage": ColorListMaskToImage,
    "FlattenAndCombineMaskImages": FlattenAndCombineMaskImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorListMaskToImage": "ColorListMaskToImage",
    "FlattenAndCombineMaskImages": "FlattenAndCombineMaskImages"
}