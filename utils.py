import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Parameters:
        boxes_preds (tensor)    : Prediction of Bbox (BATCH_SIZE, 4)
        boxes_labels (tensor)   : labels of Bbox (BATCH_SIZE, 4)
        box_format (str)        : "midpoint"/"corners". if boxes (x, y, w, h) or (x1, y1, x2, y2)
    
    Returns:
        tensor                  : Intersection over union for all examples
    """

    if box_format=='midpoint':
        box1_x1 = boxes_preds[0] - boxes_preds[2] / 2
        box1_y1 = boxes_preds[1] - boxes_preds[3] / 2
        box1_x2 = boxes_preds[0] + boxes_preds[2] / 2
        box1_y2 = boxes_preds[1] + boxes_preds[3] / 2
        box2_x1 = boxes_labels[0] - boxes_labels[2] / 2
        box2_y1 = boxes_labels[1] - boxes_labels[3] / 2
        box2_x2 = boxes_labels[0] + boxes_labels[2] / 2
        box2_y2 = boxes_labels[1] + boxes_labels[3] / 2

    elif box_format=='corners':
        box1_x1 = boxes_preds[..., 0]
        box1_y1 = boxes_preds[..., 1]
        box1_x2 = boxes_preds[..., 2]
        box1_y2 = boxes_preds[..., 3]
        box2_x1 = boxes_labels[..., 0]
        box2_y1 = boxes_labels[..., 1]
        box2_x2 = boxes_labels[..., 2]
        box2_y2 = boxes_labels[..., 3]
    
    # intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # torch.clamp: 해당 텐서 값들을 [min, max] 값의 범주 안에 들어가도록 값을 교환.
    # .clamp(0) : min=0 으로하여, 0보다 작은 값들은 모두 0으로 교체해줌. (don't intersection case 경우를 고려하기 위함)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format='midpoint'):
    """
    Parameters:
        bboxes (list)           : list of lists (각 리스트는 하나의 물체(또는 클래스)에 대한 bboxes)
                                  [[class_pred, confidence_score, x1, y1, x2, y2], [...], [...]]
        iou_threshold (float)   : best confidence bbox <-> other confidence bbox 비교하는 IoU threshold
        threshold (float)       : 처음에 confidence가 낮은 후보 bbox를 먼저 지울 때 사용하는 threshold (independent of IoU)
        box_format (str)        : 'midpoint'/'corners'

    Returns:
        list                    : NMS 연산 후에 나오는 최종 bboxes
    """

    assert type(bboxes) == list
    
    # bboxes = [[1, 0.9, x1, y1, x2, y2], [], []]

    bboxes = [box for box in bboxes if bboxes[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]  # 다른 class bbox면 남긴다.
            or intersection_over_union( # 같은 class일 때, iou threshold 보다 적으면 남긴다.
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='midpoint', num_classes=20):
    """
    Parameters:
        pred_boxes (list)       : list of lists (모든 bboxes를 담은 list)
                                  [[train_idx, class_pred, confidence, x1, y1, x2, y2], [,,,], [,,,]]
        true_boxes (list)       : pred_boxes 랑 비슷한 구조
                                  [[train_idx, class_pred, confidence(0/1), x1, y1, x2, y2], [,,,], [,,,]]
        iou_threshold (float)   : TP/FP 를 구분하는 iou threshold
        box_format (str)        : 'midpoint' / 'corners'
        num_classes (int)       : number of classes
    
    Returns:
        float                   : mAP 값 (주어진 specific IoU threshold에 대한)
    """

    average_precision = []  # 각 class에 대한 모든 AP
    epsilon = 1e-6

    # 1. 먼저 각 클래스에 대한 AP를 구한다.
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # 2. 각 클래스 c에 대한 pred_boxes, true_boxes들을 뽑아낸다
        for box in pred_boxes:
            if box[1] == c:
                detections.append(box)
        
        for box in true_boxes:
            if box[1] == c:
                ground_truths.append(box)

        # 3. 뽑아낸 각 클래스 별 모든 Ground Truths를, training example image(=idx)별로도 구별한다.
        # Image 0 -> GT 3개
        # Image 1 -> GT 5개 라고 가정했을 때,
        amount_bboxes = Counter([gt[0] for gt in ground_truths])    # amount_bboxes = {0:3, 1:5}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)   # amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        # amount_bboxes 는 뒤에 detection들과 비교할 때 매칭 유/무로 쓰인다. 
        # (1: 이미 다른 detection과 매칭 o / 0: detection과 매칭 x -> 매칭 가능)
        
        # 4. 예측된 모든(각 클래스c 별) bboxes들에 대해 confidence로 내림차순한다.
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_boxes = len(ground_truths)

        # 클래스 c에 대한 GT가 없으면 skip
        if total_true_boxes == 0:
            continue

        # 5. 각 예측 boxes들에 대한 TP/FP 를 구한다.
        for detection_idx, detection in enumerate(detections):

            # 5-1. 각 detection 당 같은 image 내의 모든 GT들을 뽑는다.
            gt_per_image = [
                gt for gt in ground_truths if gt[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = 0
            # 5-2. 같은 image 내의 각 GT들과 <-> detection 의 IoU 를 비교하여,
            # 가장 근접한 GT 1개를 찾는다. (detection : GT 는 1대1 대응이므로, 1개를 찾는다)
            for idx, gt in enumerate(gt_per_image):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # 5-3. 찾아진 GT와 iou threshold 비교를 하여, detection의 TP/FP를 구한다.
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        # 6. 모든 detections 에 대한 Precision/Recall & 클래스의 AP를 구한다.
        TP_cumsum = torch.cumsum(TP, dim=0) # e.g., [1, 1, 0, 1, 0] -> cumsum=[1, 2, 2, 3, 3]
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = torch.divide(TP_cumsum, TP_cumsum + FP_cumsum + epsilon)

        # 면적을 구하기 위해 start point를 설정해준다.
        recalls = torch.cat(torch.tensor([0]), recalls)         # recalls은 0에서 시작
        precisions = torch.cat(torch.tensor([1]), precisions)   # precision은 1에서 시작

        # torch.trapz == torch.trapezoid(y, x)
        # y, x 에 해당하는 사다리꼴 면적 공식 함수
        average_precision.append(torch.trapz(precisions, recalls))
    
    # 7. 모든 클래스에 대한 AP. 즉 mAP를 구한다
    return sum(average_precision) / len(average_precision)

def plot_image(image, boxes):
    """ Plots predicted Bboxes on Image """
    im = np.array(image)
    # h, w, c
    height, width, _ = im.shape

    # Create figure and axes
    # Axes : 축(axis), 눈금(tick), 텍스트(text), 다각형(polygon) 등 그래프의 다양한 구성요소를 포함하는 객체
    fig, ax = plt.subplots(1)
    # Display image
    ax.imshow(im)

    # boxes (list) : list of lists [[x, y, w, h]]

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "box에 x, y, w, h 이외에 다른 값들이 있다!"
        lower_left_x = box[0] - box[2] / 2
        lower_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (lower_left_x * width, lower_left_y * height), box[2]*width, box[3]*height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_aptch(rect)
    
    plt.show()

def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", box_format="midpoint", device="cpu"):
    all_pred_boxes = []
    all_true_boxes = []

    # 먼저, model을 eval모드로 전환
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        # 각 batch_size 안의 하나의 이미지에 대해
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format
            )

            # 사진 batch_size 하나면, nms_boxes 구한 이미지 출력해보기
            if batch_idx == 0 and idx == 0:
                plot_image(x[idx].permute(1,2,0).to('cpu'), nms_boxes)
                print(nms_boxes)
            
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred (???????)
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            
            train_idx += 1
    
    # model train 모드로 전환
    model.train()
    return all_pred_boxes, all_true_boxes

def converted_cellboxes(predictions, S=7):
    """
    분할 크기가 S인 cell 비율에 맞춰져 있는 Bbox를 전체 이미지 비율로 convert.
    vectorized로 convert하겠다.

    Parameters:
        predictions (list) : model output(predictions) (N, S*S*30) / target(labels) (N, S, S, 30)
    
    Returns:
        list : (N, S, S, 6) (BATCH_SIZE, S_y, S_x, [class, confidence, x, y, w, h])
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25] # (x, y, w, h)
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat( # (1, N, 7, 7) +(cat) (1, N, 7, 7) -> (2, N, 7, 7)
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )

    best_box = scores.argmax(0).unsqueeze(-1)   # (N, 7, 7) -> (N, 7, 7, 1)
    best_boxes = bboxes1 * (1-best_box) + bboxes2 * best_box # bboxes1 또는 bboxes2 둘 중 하나. # (N, 7, 7, 4)

    # torch.tensor([0,1,2,3,4,5,6]).repeat(N, 7, 1) -> [0,1,2,3,4,5,6] 7D vector가 dim=(N, 7, 1) 만큼 반복됨.
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)   # (7) -> (N, 7, 7) -> (N, 7, 7, 1)

    x = 1 / S * (best_boxes[..., :1] + cell_indices) # 전체 이미지의 0~1 비율로 만듦. x축만 생각하면, 1 / {S * (cell # + 기존 cell 비율)}
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0,2,1,3))    # y축. (N, 7(y), 7(x), 1) -> (N, 7(x), 7(y), 1)
    w_y = 1 / S * best_boxes[..., 2:4]

    converted_bboxes = torch.cat((x, y, w_y), dim=-1) # (N, 7, 7, 4)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1) # (N, 7, 7, 1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1) # (N, 7, 7, 1)

    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = converted_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bboxes_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bboxes_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filenames='my_checkpoint.pth.tar'):
    print("=> Saving Checkpoint")
    torch.save(state, filenames)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])