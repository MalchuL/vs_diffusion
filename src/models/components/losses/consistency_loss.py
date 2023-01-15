import random

import torch
import torch.nn as nn


# From https://arxiv.org/pdf/2209.11224.pdf
class ConsistencyLoss(nn.Module):
    def __init__(self, min_bbox_size_percent=0.8, max_bbox_size_percent=0.9, must_divided=32, test_iters=10):
        super().__init__()
        self.must_divided = must_divided
        self.min_bbox_size_percent = min_bbox_size_percent
        self.max_bbox_size_percent = max_bbox_size_percent
        self.loss = nn.MSELoss()

        self.test_iters = test_iters
        assert test_iters >= 1

    def generate_crop(self, W_base, H_base, bbox_size_percent):
        W_base, H_base = int(W_base), int(H_base)
        H, W, = round(H_base * bbox_size_percent) // self.must_divided * self.must_divided, \
                round(W_base * bbox_size_percent) // self.must_divided * self.must_divided
        crop_y = random.randint(0, H_base - H + self.must_divided - 1) // self.must_divided * self.must_divided
        crop_x = random.randint(0, W_base - W + self.must_divided - 1) // self.must_divided * self.must_divided
        assert H < H_base, W < W_base
        assert crop_y + H <= H_base, f'{crop_y, H, crop_y + H}'
        assert crop_x + W <= W_base, f'{crop_x, H, crop_x + W}'
        x1, y1, x2, y2 = crop_x, crop_y, crop_x + W, crop_y + H
        return x1, y1, x2, y2

    def overlap_crop(self, first_crop, second_crop):
        x1 = max(first_crop[0], second_crop[0])
        y1 = max(first_crop[1], second_crop[1])
        x2 = min(first_crop[2], second_crop[2])
        y2 = min(first_crop[3], second_crop[3])
        return x1, y1, x2, y2

    def offset_crop(self, crop, overlap_crop):
        return [overlap_crop[0] - crop[0], overlap_crop[1] - crop[1], overlap_crop[2] - crop[0],
                overlap_crop[3] - crop[1]]

    def crop_square(self, crop):
        inter_area = abs(max((crop[2] - crop[0], 0)) * max((crop[3] - crop[1]), 0))
        return inter_area

    def rand_range(self, min, max):
        return min + random.random() * (max - min)
    def forward(self, image, model):
        N_base, C_base, H_base, W_base = image.shape

        for i in range(self.test_iters):
            rand_bbox_size_percent = self.rand_range(self.min_bbox_size_percent, self.max_bbox_size_percent)
            first_crop = self.generate_crop(W_base, H_base, rand_bbox_size_percent)
            alpha = (i / self.test_iters)  # Increasing while don't crops
            second_bbox_size_percent = alpha * 1.0 + (1 - alpha) * rand_bbox_size_percent
            second_crop = self.generate_crop(W_base, H_base, second_bbox_size_percent)
            overlap = self.overlap_crop(first_crop, second_crop)
            if self.crop_square(overlap) > 0 and first_crop != second_crop:
                break

        overlapped_first_crop = self.offset_crop(first_crop, overlap)
        overlapped_second_crop = self.offset_crop(second_crop, overlap)
        assert all([value % self.must_divided == 0 for value in overlapped_first_crop])
        assert all([value % self.must_divided == 0 for value in overlapped_second_crop])

        first_cropped_img = image[:, :, first_crop[1]:first_crop[3], first_crop[0]:first_crop[2]]
        second_cropped_img = image[:, :, second_crop[1]:second_crop[3], second_crop[0]:second_crop[2]]
        first_cropped_features = model.forward_features(first_cropped_img)
        second_cropped_features = model.forward_features(second_cropped_img)
        losses = []
        for first_cropped_feature, second_cropped_feature in zip(first_cropped_features, second_cropped_features):
            downsampled_coef = second_cropped_img.shape[2] // second_cropped_feature.shape[2]
            downsampled_first_crop = [value // downsampled_coef for value in overlapped_first_crop]
            downsampled_second_crop = [value // downsampled_coef for value in overlapped_second_crop]
            first_overplapped_feature_crop = first_cropped_feature[:, :,
                                             downsampled_first_crop[1]:downsampled_first_crop[3],
                                             downsampled_first_crop[0]:downsampled_first_crop[2]]
            second_overplapped_feature_crop = second_cropped_feature[:, :,
                                              downsampled_second_crop[1]:downsampled_second_crop[3],
                                              downsampled_second_crop[0]:downsampled_second_crop[2]]
            assert first_overplapped_feature_crop.shape == second_overplapped_feature_crop.shape, f'{first_cropped_feature.shape}, {second_cropped_feature.shape}'
            losses.append(self.loss(first_overplapped_feature_crop, second_overplapped_feature_crop))
        loss = sum(losses) / len(losses)
        return loss




class ConsistencyLossWithTarget(nn.Module):
    def __init__(self, bbox_size_percent=0.8, must_divided=32):
        super().__init__()
        self.must_divided = must_divided
        self.bbox_size_percent = bbox_size_percent
        self.loss = nn.MSELoss()

    def forward(self, image, model, target_features):
        N_base, C_base, H_base, W_base = image.shape
        H, W, = round((H_base * self.bbox_size_percent) // self.must_divided) * self.must_divided, \
                round((W_base * self.bbox_size_percent) // self.must_divided) * self.must_divided
        crop_y = round(random.randint(0, H_base - H - 1) // self.must_divided) * self.must_divided
        crop_x = round(random.randint(0, W_base - W - 1) // self.must_divided) * self.must_divided
        assert H < H_base, W < W_base
        assert crop_y + H < H_base, f'{crop_y, H, crop_y + H}'
        assert crop_x + W < W_base, f'{crop_x, H, crop_x + W}'
        cropped_img = image[:, :, crop_y:crop_y + H, crop_x:crop_x + W]
        cropped_features = model.forward_features(cropped_img)
        losses = []
        for cropped_feature, target_feature in zip(cropped_features, target_features):
            feat_H = target_feature.shape[2]
            downsampled_coef = H_base // feat_H
            target_crop_y = crop_y // downsampled_coef
            target_crop_x = crop_x // downsampled_coef
            target_H = H // downsampled_coef
            target_W = W // downsampled_coef
            target_feature_crop = target_feature[:, :, target_crop_y: target_crop_y + target_H,
                                  target_crop_x: target_crop_x + target_W]
            losses.append(self.loss(cropped_feature, target_feature_crop))
        loss = sum(losses) / len(losses)
        return loss
