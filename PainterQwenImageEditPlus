import node_helpers
import comfy.utils
import math
import torch
import comfy.model_management
import torch.nn.functional as F


class PainterQwenImageEditPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "mode": (["1_image", "2_image", "3_image", "4_image", "5_image", 
                         "6_image", "7_image", "8_image", "9_image", "10_image"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1_mask": ("MASK",),  # Moved above image1
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Pixel-perfect Qwen image editing with dynamic image inputs"

    def encode(self, clip, prompt, mode, batch_size, vae=None, 
               image1=None, image2=None, image3=None, image4=None, image5=None,
               image6=None, image7=None, image8=None, image9=None, image10=None,
               image1_mask=None, width=1024, height=1024):
        
        target_latent_h = height // 8
        target_latent_w = width // 8
        
        all_images = [image1, image2, image3, image4, image5, 
                      image6, image7, image8, image9, image10]
        count = int(mode.split("_")[0])
        images = [img for i, img in enumerate(all_images[:count]) if img is not None]
        
        ref_latents = []
        vl_images = []
        noise_mask = None
        
        ref_longest_edge = max(width, height)
        
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            samples = image.movedim(-1, 1)
            current_total = samples.shape[3] * samples.shape[2]
            
            vl_total = int(384 * 384)
            vl_scale_by = math.sqrt(vl_total / current_total)
            vl_width = round(samples.shape[3] * vl_scale_by)
            vl_height = round(samples.shape[2] * vl_scale_by)
            
            s_vl = comfy.utils.common_upscale(samples, vl_width, vl_height, "area", "center")
            vl_image = s_vl.movedim(1, -1)
            vl_images.append(vl_image)
            
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
            
            if vae is not None:
                _, img_h, img_w, _ = image.shape
                input_ar = img_w / img_h
                target_ar = width / height
                pixel_perfect = abs(input_ar - target_ar) < 0.01
                
                ori_longest_edge = max(samples.shape[2], samples.shape[3])
                scale_by = ori_longest_edge / ref_longest_edge
                scaled_width = int(round(samples.shape[3] / scale_by))
                scaled_height = int(round(samples.shape[2] / scale_by))
                
                if pixel_perfect:
                    canvas_width = math.ceil(scaled_width / 8.0) * 8
                    canvas_height = math.ceil(scaled_height / 8.0) * 8
                    
                    canvas = torch.zeros(
                        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                        dtype=samples.dtype,
                        device=samples.device
                    )
                    
                    crop = "center"
                    resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, "lanczos", crop)
                    resized_width_actual = resized_samples.shape[3]
                    resized_height_actual = resized_samples.shape[2]
                    
                    canvas[:, :, :resized_height_actual, :resized_width_actual] = resized_samples
                    s = canvas
                    
                    if i == 0 and image1_mask is not None:
                        mask = image1_mask
                        if mask.dim() == 2:
                            mask_samples = mask.unsqueeze(0).unsqueeze(0)
                        elif mask.dim() == 3:
                            mask_samples = mask.unsqueeze(1)
                        else:
                            print(f"Warning: Unexpected mask shape {mask.shape}, skipping")
                            mask_samples = None
                        
                        if mask_samples is not None:
                            resized_mask = comfy.utils.common_upscale(mask_samples, resized_width_actual, resized_height_actual, "area", crop)
                            noise_mask = resized_mask.squeeze(1)
                else:
                    crop = "center"
                    target_w = round(scaled_width / 8.0) * 8
                    target_h = round(scaled_height / 8.0) * 8
                    s = comfy.utils.common_upscale(samples, target_w, target_h, "lanczos", crop)
                    
                    if i == 0 and image1_mask is not None:
                        mask = image1_mask
                        if mask.dim() == 2:
                            mask_samples = mask.unsqueeze(0).unsqueeze(0)
                        elif mask.dim() == 3:
                            mask_samples = mask.unsqueeze(1)
                        else:
                            print(f"Warning: Unexpected mask shape {mask.shape}, skipping")
                            mask_samples = None
                        
                        if mask_samples is not None:
                            m = comfy.utils.common_upscale(mask_samples, target_w, target_h, "area", crop)
                            noise_mask = m.squeeze(1)
                
                image = s.movedim(1, -1)
                ref_latent = vae.encode(image[:, :, :, :3])
                ref_latents.append(ref_latent)
        
        tokens = clip.tokenize(image_prompt + prompt, images=vl_images, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        negative_tokens = clip.tokenize("")
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        
        if len(images) > 0 and vae is not None:
            base_img = images[0]
            base_samples = base_img.movedim(-1, 1)
            base_resized = comfy.utils.common_upscale(base_samples, target_latent_w * 8, target_latent_h * 8, "lanczos", "center")
            base_resized = base_resized.movedim(1, -1)
            latent_samples = vae.encode(base_resized[:, :, :, :3])
            
            if latent_samples.dim() == 3:
                latent_samples = latent_samples.unsqueeze(0)
            
            device = latent_samples.device
            dtype = latent_samples.dtype
            
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            negative_conditioning = node_helpers.conditioning_set_values(negative_conditioning, {"reference_latents": ref_latents}, append=True)
        else:
            device = comfy.model_management.intermediate_device()
            dtype = torch.float32
            latent_samples = torch.zeros(1, 4, target_latent_h, target_latent_w, device=device, dtype=dtype)
        
        latent_out = {"samples": latent_samples}
        
        if noise_mask is not None:
            if noise_mask.dim() == 2:
                noise_mask = noise_mask.unsqueeze(0).unsqueeze(0)
            elif noise_mask.dim() == 3:
                noise_mask = noise_mask.unsqueeze(1)
            
            if noise_mask.shape[2] != target_latent_h or noise_mask.shape[3] != target_latent_w:
                noise_mask = F.interpolate(
                    noise_mask.float(), 
                    size=(target_latent_h, target_latent_w), 
                    mode='bilinear',
                    align_corners=False
                )
            
            noise_mask = noise_mask.squeeze(1)
            
            if batch_size > 1 and noise_mask.shape[0] == 1:
                noise_mask = noise_mask.repeat(batch_size, 1, 1)
            
            latent_out["noise_mask"] = noise_mask
        
        if batch_size > 1:
            conditioning = conditioning * batch_size
            negative_conditioning = negative_conditioning * batch_size
            
            samples = latent_out["samples"]
            if samples.shape[0] != batch_size:
                target_shape = [batch_size] + [1] * (samples.dim() - 1)
                samples = samples.repeat(*target_shape)
            latent_out["samples"] = samples
        
        return (conditioning, negative_conditioning, latent_out)


NODE_CLASS_MAPPINGS = {
    "PainterQwenImageEditPlus": PainterQwenImageEditPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterQwenImageEditPlus": "Painter Qwen Image Edit Plus",
}
