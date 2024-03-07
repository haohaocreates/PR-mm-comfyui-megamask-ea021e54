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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    CATEGORY = "meshmesh"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(self, image):
        newimage = torch.cat([img.unsqueeze(0) for img in image], dim=0)
        return (newimage,)


NODE_CLASS_MAPPINGS = {
    "ColorListMaskToImage": ColorListMaskToImage,
    "FlattenAndCombineMaskImages": FlattenAndCombineMaskImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorListMaskToImage": "ColorListMaskToImage",
    "FlattenAndCombineMaskImages": "FlattenAndCombineMaskImages"
}