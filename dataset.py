import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # label
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
        boxes = torch.tensor(boxes)

        # Image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        # Data Augmentation
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert original label(dependent to whole image) to cell dependent label
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            #row, column (cell 위치 나타냄)
            i, j = int(self.S * y), int(self.S * x) # (0~1) 범위 -> (0~7) 범위로 x7배 증가.
            cell_x, cell_y = self.S*x - j, self.S*y - i # cell 당 위치 비율 전환.
            cell_w, cell_h = self.S * width, self.S * height
            
            # If object found for specific cell (i, j)
            # fill in label_matrix
            if label_matrix[i, j, 20] == 0:
                # Iobj (S, S, 20)
                label_matrix[i, j, 20] = 1

                # class_label (S, S, 0:20)
                label_matrix[i, j, class_label] = 1

                # box coordinate (S, S, 21:25)
                box_coordinate = torch.tensor(
                    [cell_x, cell_y, cell_w, cell_h]
                )
                label_matrix[i, j, 21:25] = box_coordinate
        
        return image, label_matrix