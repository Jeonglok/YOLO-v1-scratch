"""
Implementation of Yolov1 Loss Functions from originial paper.
"""

import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.sse = nn.MSELoss(reduction='sum')

        """
        S : 이미지 split size (in paper 7)
        B : cell당 bbox 개수 (in paper 2)
        C : PASCAL VOC class 개수 (in paper 20)
        """
        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # First Output Predictions are shaped (BATCH_SIZE, S*S*(C+5B)) -> 따라서 (BATCH_SIZE, 7, 7, 30)으로 reshape 해준다.
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # 각 cell당 나오는 2개의 Bbox IOU 전체 행렬 계산
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # prediction B1 box 좌표 <-> GT box 좌표
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # prediction B2 box 좌표 <-> GT box 좌표
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # 각 cell당 Highest IoU 계산
        # iou_maxes : Max IoU 값, bestbox : Argmax 값 (0 or 1)
        iou_maxes, best_box = torch.max(ious, dim=0)

        # Iobj_i
        # (N, S, S, 1)
        exists_box = target[..., 20].unsqueeze(3)


        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # (N, S, S, 4)
        box_predictions = exists_box * (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])
        box_targets = exists_box * target[..., 21:25]

        # w, h 부분만 따로 제곱근 처리
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6)
        box_targets[..., 2:4] = torch.sqrt(box_predictions[..., 2:4])

        # (N*S*S, 4)
        box_loss = self.sse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # (N, S, S, 1)
        confidence_predictions = exists_box * (best_box * predictions[..., 25] + (1 - best_box) * predictions[..., 20])
        confidence_targets = exists_box * target[..., 20]

        # (N*S*S*1)
        object_loss = self.sse(
            torch.flatten(confidence_predictions),
            torch.flatten(confidence_targets)
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # (N, S, S, 1)
        no_object_predictions_b1 = (1 - exists_box) * predictions[..., 20]
        no_object_predictions_b2 = (1 - exists_box) * predictions[..., 25]
        no_object_targets = (1 - exists_box) * target[..., 20]

        # (N, S*S*1)
        no_object_loss = self.sse(
            torch.flatten(no_object_predictions_b1, no_object_targets, start_dim=1)
        )
        no_object_loss += self.see(
            torch.flatten(no_object_predictions_b2, no_object_targets, start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.sse(
            torch.flatten(predictions[..., :20], end_dim=-2),
            torch.flatten(target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss            # (N*S*S, 4)
            + object_loss                           # (N*S*S*1)
            + self.lambda_noobj * no_object_loss    # (N, S*S*1)
            + class_loss                            # (N*S*S, 20)
        )

        return loss