import os
import sys
import math
import copy
import torch
import numpy as np
import cv2
from tqdm import trange
import folder_paths

import comfy.model_management
from comfy.utils import ProgressBar
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger('comfyui_image_matting')

def FB_blur_fusion_foreground_estimator_1(image, alpha, r=90):
    alpha = alpha[:, :, None]
    return FB_blur_fusion_foreground_estimator(image, F=image, B=image, alpha=alpha, r=r)[0]


def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(
        image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * \
        (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B

matting_model_dir_name = "matting_models"
matting_model_list = {
    "vitmatte_small (103 MB)": {
        "model_url": "hustvl/vitmatte-small-composition-1k"
    },
    "vitmatte_base (387 MB)": {
        "model_url": "hustvl/vitmatte-base-composition-1k"
    },
}
def list_matting_models():
    return list(matting_model_list.keys())

def load_matting_model(model_name):
    model_url = matting_model_list[model_name]["model_url"]
    matting_model_checkpoint_path = get_local_filepath_(
        matting_model_list[model_name]["model_url"], matting_model_dir_name,f"{model_url}")
    matting_model = VitMatteForImageMatting.from_pretrained(matting_model_checkpoint_path)
    preprocessor = VitMatteImageProcessor.from_pretrained(matting_model_checkpoint_path)
    matting_model_device = comfy.model_management.get_torch_device()
    matting_model = matting_model.to(matting_model_device)
    matting_model.eval()
    return (matting_model, preprocessor)

def get_local_filepath_(url, dirname, local_file_name=None):

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        
        # download_url_to_file(url, destination)
        logger.warn(f'downloading {url} to {destination}')
        VitMatteForImageMatting.from_pretrained(url).save_pretrained(destination)
        VitMatteImageProcessor.from_pretrained(url).save_pretrained(destination)
    return destination

class MattingModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_matting_models(), ),
            }
        }
    CATEGORY = "image_matting"
    FUNCTION = "main"
    RETURN_TYPES = ("MATTING_MODEL", "MATTING_PREPROCESSOR")

    def main(self, model_name):
        matting_model,matting_preprocessor = load_matting_model(model_name)
        return (matting_model, matting_preprocessor)
def gen_trimap(alpha, kernel_size=25):
        
        k_size = kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (k_size, k_size))
        dilated = cv2.dilate(alpha, kernel)
        eroded = cv2.erode(alpha, kernel)
        trimap = np.zeros(alpha.shape)
        trimap.fill(128)
        trimap[eroded > 254.5] = 255
        trimap[dilated < 0.5] = 0

        return trimap

class CreateTrimap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ('MASK', {}),
                "kernel_size": ("INT", {
                    "default": 9,
                    "min": 0,
                    "max": 50,
                    "step": 0.01
                }),
            }
        }
    CATEGORY = "image_matting"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)

    def main(self, mask, kernel_size):
        res_masks = []
        for mask_ in mask:
            
            mask_ = mask_.cpu().numpy()*255
            mask_ = mask_.astype(np.uint8)
            mask_ = gen_trimap(mask_, kernel_size=kernel_size)/255.
            
            res_masks.extend([torch.from_numpy(mask_).unsqueeze(0)])
        return (torch.cat(res_masks, dim=0),)


    
class ApplyMatting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matting_model": ('MATTING_MODEL', {}),
                "matting_preprocessor": ('MATTING_PREPROCESSOR', {}),
                "image": ('IMAGE', {}),
                "trimap": ("MASK", {}),
            }
        }
    CATEGORY = "image_matting"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, matting_model,matting_preprocessor, image, trimap):
        res_images = []
        res_masks = []
        for item, tri in zip(image, trimap):
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            print(tri.shape)
            tri = Image.fromarray(
                np.clip(255. * tri.cpu().numpy(), 0, 255).astype(np.uint8)).convert('L')
            
            alpha = matting_predict(
                matting_model,
                matting_preprocessor,
                item,
                tri
            )
            alpha = (alpha*255).astype(np.uint8)
            item_shape = item.size
            alpha = cv2.resize(alpha, (item_shape[0], item_shape[1]), interpolation=cv2.INTER_NEAREST)
            ffe_foreground = FB_blur_fusion_foreground_estimator(
                np.array(item.convert("RGB"))/255.,np.array(item.convert("RGB"))/255.,np.array(item.convert("RGB"))/255., alpha[:,:,None]/255., r=90
            )[0]*255
            ffe_foreground = ffe_foreground.astype(np.uint8)
            
            #H,W,C
            alpha = alpha.astype(np.uint8)[..., None] 
            print(ffe_foreground.shape, alpha.shape)
            #make it rgba with alpha
            ffe_foreground = np.concatenate([ffe_foreground, alpha], axis=2)
            
            #resize image from HWC to CHW
            res_images.extend([torch.from_numpy(ffe_foreground).unsqueeze(0)/255.])
            res_masks.extend([torch.from_numpy(alpha[:,:,0]).unsqueeze(0)/255.])
        
        if len(res_images) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))

def matting_predict(
    matting_model,
    matting_preprocessor,
    image,
    tri
):
    def load_matting_image(image_pil, tri, image_preprocessor):

        inputs = image_preprocessor(images=image_pil, trimaps=tri, return_tensors="pt")
        return inputs

    inputs = load_matting_image(image.convert("RGB"), tri, matting_preprocessor)
    for key in inputs:
        inputs[key] = inputs[key].to(comfy.model_management.get_torch_device())
    with torch.no_grad():
        alphas = matting_model(**inputs).alphas[0][0].cpu().numpy()
    return alphas

NODE_CLASS_MAPPINGS = {
    "MattingModelLoader": MattingModelLoader,
    "ApplyMatting": ApplyMatting,
    "CreateTrimap": CreateTrimap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MattingModelLoader": "Load Matting Model",
    "ApplyMatting": "Apply Matting",
    "CreateTrimap": "Create Trimap",
}