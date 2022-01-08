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
# Weight decay 와 Dropout 을 없앤다. 전체 이미지 training이 아니라, 100개 예제 학습이라
# Overfitting을 내어, AP가 올라가는 것만 확인하기 때문에, Overfitting을 더 쉽게 하기 위해 제거.
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 4 # GPU를 쓰기 위해, 즉 학습을 위해 쓰이는 CPU 코어 수. GPU는 기본적으로 CPU의 컨트롤을 받기 때문에.
