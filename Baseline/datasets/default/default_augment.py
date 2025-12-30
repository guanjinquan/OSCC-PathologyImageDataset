from torchvision.transforms import Compose, Normalize, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomAutocontrast, RandomAdjustSharpness
import torch
            
class TrainTransforms:
    def __init__(self, mean_std, args):
        self.transforms = Compose([
            Normalize(mean=mean_std[0], std=mean_std[1]),
            RandomResizedCrop(scale=(args.crop_scale, 1), size=(512, 512)),
            RandomVerticalFlip(p=0.5), 
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=(-45, 45)),
            RandomAutocontrast(p=0.5), 
            RandomAdjustSharpness(sharpness_factor=3, p=0.5)  
        ])
    
    def __call__(self, ensemble_data):
        ensemble_data = torch.from_numpy(ensemble_data).float()
        return self.transforms(ensemble_data)


class TestTransforms:
    def __init__(self, mean_std):
        self.transforms = Normalize(mean=mean_std[0], std=mean_std[1])

    def __call__(self, ensemble_data):
        ensemble_data = torch.from_numpy(ensemble_data).float()
        return self.transforms(ensemble_data)