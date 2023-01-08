import albumentations as A

from albumentations.pytorch import ToTensorV2
import cv2


def get_transform(load_size, fine_size, is_train, apply_strong=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    prob = 0.3
    if is_train:
        pre_process = [
            A.SmallestMaxSize(load_size, always_apply=True, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.CenterCrop(fine_size, fine_size, always_apply=True),
        ]
    else:
        pre_process = [
            A.SmallestMaxSize(load_size, always_apply=True, interpolation=cv2.INTER_CUBIC),
            A.CenterCrop(fine_size, fine_size, always_apply=True)]

    very_rare_prob = 0.05
    rare_prob = 0.1
    medium_prob = 0.2
    normal_prob = 0.3
    often_prob = 0.6
    compression_prob = 0.35
    if apply_strong:
        strong = [
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=normal_prob),
                A.MotionBlur(p=rare_prob),
                A.Downscale(scale_min=0.6, scale_max=0.8, interpolation=cv2.INTER_CUBIC, p=rare_prob),
            ], p=rare_prob),
            A.OneOf([
                A.ImageCompression(quality_lower=39, quality_upper=60, p=compression_prob),
                A.MultiplicativeNoise(multiplier=[0.92, 1.08], elementwise=True, per_channel=True, p=compression_prob),
                A.ISONoise(p=medium_prob)
            ], p=normal_prob),
            A.OneOf([
                A.ToGray(p=often_prob),
                A.ToSepia(p=medium_prob)
            ], p=rare_prob),
            A.OneOf([
                A.CLAHE(p=rare_prob),
                A.Equalize(by_channels=False, p=rare_prob),
            ], p=normal_prob),
            A.OneOf([
                A.RandomGamma(p=medium_prob),
                A.RGBShift(p=medium_prob),
            ], p=medium_prob),
            A.OneOf([
                A.HueSaturationValue(p=medium_prob),
                A.RandomBrightnessContrast(p=medium_prob, brightness_limit=(-0.3, 0.2)),
            ], p=normal_prob),

        ]
    else:
        strong = []

    post_process = [A.Normalize(mean,
                                std),
                    ToTensorV2()]

    if not is_train:
        strong = []

    return A.Compose([*pre_process, *strong, *post_process])
