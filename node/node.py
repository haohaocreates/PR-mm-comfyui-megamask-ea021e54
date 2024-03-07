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

    def select_next_color(self, colorlist, idx):
        print('colorlist', colorlist)
        colors = colorlist["string"].split(",")
        print('colors', colors)
        color = colors[idx]
        return color

    def render_mask(self, mask, colorlist, background):
        
        masks = tensor2np(mask)
        images = []
        print('colorlist', colorlist)
        
        idx = 0
        for m in masks:
            _mask = Image.fromarray(m).convert("L")
            color = self.select_next_color(colorlist, idx)
            idx += 1
            print('selected color', color)
            image = Image.new("RGBA", _mask.size, color=color)
            # apply the mask
            image = Image.composite(
                image, Image.new("RGBA", _mask.size, color=background), _mask
            )

            images.append(image.convert("RGB"))


        return (pil2tensor(images),)

NODE_CLASS_MAPPINGS = {
    "ColorListMaskToImage": ColorListMaskToImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorListMaskToImage": "ColorListMaskToImage"
}