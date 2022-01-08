import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
# Weight decay 와 Dropout 을 없앤다. 전체 이미지 training이 아니라, 100개 예제 학습이라
# Overfitting을 내어, AP가 올라가는 것만 확인하기 때문에, Overfitting을 더 쉽게 하기 위해 제거.
EPOCHS = 100
NUM_WORKERS = 4 # GPU를 쓰기 위해, 즉 학습을 위해 쓰이는 CPU 코어 수. GPU는 기본적으로 CPU의 컨트롤을 받기 때문에.
PIN_MEMORY = True if torch.cuda.is_available() else False
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = 'data/images'
LABEL_DIR = 'data/labels'

# Image Transforms (toTensor, resize, data augmentation 등등)
# 보통 image Transforms 같은 경우, 간단한 함수호출 대신 호출가능한 클래스로 작성한다.
# 이렇게 한다면, 클래스가 호출될 때마다 Transform의 매개변수를 계속 전달할 필요가 없다. 클래스의 __call__ 함수로 구현하면 된다.
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    # 클래스 객체 호출 시, 실행되는 __call__함수 구현
    def __call__(self, img, bboxes):
        # 각각의 self.transform마다 img에서만 처리를 해준다.
        # 보통 custom transform에서는 bboxes 까지 같이 처리를 해주나 이 예시에서는 img만 전처리하기로 한다.
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

# 만든 custom transform 'Compose' 클래스 객체를 만들어서, 
# torchvision.transforms 메소드를 -> 'Compose' 객체의 멤버로 하는 객체를 만든다. 
# 앞으로는 'Compose' 객체로 접근해서 transform 사용한다.
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


# 전체 dataset을 받아와서 BATCH당 training을 하는 Train function 구현
def train_fn(train_loader, model, optimizer, loss_fn):
    # 1. train_loader를 통해 데이터 불러오기
    tqdm_train = tqdm(train_loader, leave=True) # 상태바를 보기위한 tqdm으로 train_loader 감싸기
    mean_loss = []

    # 2. train_loader의 BATCH당 학습 진행
    for batch_idx, (x, y) in tqdm_train:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)          # 2-1. model forwarding
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()   # 2-2. model optimizer 진행
        loss.backward()         # 2-3. model backwarding 진행
        optimizer.step()        # 2-4. 다음 step 학습 진행

        # Progress bar 업데이트
        tqdm_train.set_postfix(loss = loss.item())

    print(f"Mean loss : {sum(mean_loss)/len(mean_loss)}")


def main():
    # 1. model Initialization
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    # 2. optimizer Initialization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 3. Loss Initialization
    loss_fn = YoloLoss()

    # 4. Load model checkpoint
    if LOAD_MODEL:
        load_checkpoint(LOAD_MODEL_FILE, model, optimizer)
    
    # 5. Train / Test Dataset Load
    train_dataset = VOCDataset(
        'data/8examples.csv',
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform
    )

    test_dataset = VOCDataset('data/test.csv', img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    # 6. Start Training for loop in Epochs
    for epoch in range(EPOCHS):

        # 8개 사진 임의 출력
        # for x, y in train_loader:
        #     x = x.to(DEVICE)
        #     for idx in range(8):
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4)
        #         plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #     import sys
        #     sys.exit()

        # epoch 당 mAP 구해서 출력해보기.
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5)
        print(f"Train mAP for {epoch} epochs : {mean_avg_prec}")

        # 조건 만족하면 checkpoint 저장
        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filenames=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        # Training
        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()