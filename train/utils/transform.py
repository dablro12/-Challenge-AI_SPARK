from torchvision.transforms import functional as F
from torchvision import transforms
import torch 
import random

class AddGaussianNoise:
    """ Adding Gaussian Noise as an augmentation """ 
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class ColorJitterTransform:
    """ Color Space Transformation """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        
    def __call__(self, img):
        img = F.to_pil_image(img)
        img = self.color_jitter(img)
        return F.to_tensor(img)

class RandomDropout:
    """ Regularization Techniques """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if torch.rand(1) < self.p:
            mask = torch.rand(img.size()) > self.p
            img = img * mask
        return img
    
class RandomColorJitterTransform(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        # Randomly adjust color jitter parameters
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        
        # Apply color jitter transformation
        img = transforms.functional.adjust_brightness(img, brightness_factor)
        img = transforms.functional.adjust_contrast(img, contrast_factor)
        img = transforms.functional.adjust_saturation(img, saturation_factor)
        img = transforms.functional.adjust_hue(img, hue_factor)

        return img