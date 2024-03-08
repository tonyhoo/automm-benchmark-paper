from PIL import Image
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from sentence_transformers import util
from torch.utils.data import DataLoader
import os
import zipfile
from tqdm import tqdm
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
import numpy as np
import json
import argparse
import torch

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model = SentenceTransformer('clip-ViT-L-14')
num_epochs = 10
train_batch_size = 24
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
margin = 0.5

parser = argparse.ArgumentParser(description='PyTorch Training with Seed')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--dataname', type=str, default='aribnb')
parser.add_argument('--datapath', type=str, default='image_image_matching/airbnb')
parser.add_argument('--traindata', type=str, default='image_image_matching/airbnb/train_processed.csv')
parser.add_argument('--valdata', type=str, default='image_image_matching/airbnb/val_processed.csv')
parser.add_argument('--testdata', type=str, default='image_image_matching/airbnb/test_processed.csv')
parser.add_argument('--col1', type=str, default='image_1')
parser.add_argument('--col2', type=str, default='image_2')
parser.add_argument('--labelcol', type=str, default='label')
args = parser.parse_args()

set_seed(args.seed)

dataset_path = "image_image_matching/{}".format(args.dataname)
train_path = os.path.join(dataset_path, "train_processed.csv")
val_path = os.path.join(dataset_path, "val_processed.csv")
test_path = os.path.join(dataset_path, "test_processed.csv")
train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)
image_col_1 = "image1"
image_col_2 = "image2"
label_col = "label"
match_label = 1

model_save_path = "exp/st/image-image/{}/seed_{}".format(args.dataname, args.seed)
os.makedirs(model_save_path, exist_ok=True)

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for image_col in [image_col_1, image_col_2]:
    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    val_data[image_col] = val_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

train_samples = []
for idx, row in train_data.iterrows():
    image1 = Image.open(row[image_col_1])
    image2 = Image.open(row[image_col_2])
    sample = InputExample(texts=[image1, image2], label=int(row[label_col]))
    train_samples.append(sample)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.ContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=1000,
    output_path=model_save_path,
)

print("========= TESTING =========: ", model_save_path)
model = SentenceTransformer(model_save_path)
y_true = []
y_pred = []
sentences1 = []
sentences2 = []
for idx, row in test_data.iterrows():
    image1 = Image.open(row[image_col_1])
    image2 = Image.open(row[image_col_2])
    label = int(row[label_col])
    y_true.append(label)
    sentences1.append(image1)
    sentences2.append(image2)
print("== Finished Building Test Data ==")
embed1 = model.encode(sentences1, convert_to_tensor=True)
embed2 = model.encode(sentences2, convert_to_tensor=True)
cos_sim = util.cos_sim(embed1, embed2)
y_true = np.array(y_true)
y_pred = cos_sim.diag().cpu().numpy()
score = roc_auc_score(y_true, y_pred)
print("TEST ROC_AUC: ", score)
with open(os.path.join(model_save_path, "test_metrics.json"), "w") as fp:
    json.dump(score, fp)