from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
from zipfile import ZipFile
import random
import pandas as pd
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

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)
#### /print debug information to stdout


model = SentenceTransformer("all-mpnet-base-v2")
num_epochs = 10
train_batch_size = 64

# As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

# Negative pairs should have a distance of at least 0.5
margin = 0.5

parser = argparse.ArgumentParser(description='PyTorch Training with Seed')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--dataname', type=str, default='MRPC')
args = parser.parse_args()

set_seed(args.seed)
model_save_path = "exp/st/text-text/{}/seed_{}".format(args.dataname, args.seed)
dataset_path = "text_text_matching/{}".format(args.dataname)
train_path = os.path.join(dataset_path, "train_processed.csv")
val_path = os.path.join(dataset_path, "val_processed.csv")
test_path = os.path.join(dataset_path, "test_processed.csv")
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

os.makedirs(model_save_path, exist_ok=True)

######### Read train data  ##########
# Read train data
train_samples = []
for idx, row in df_train.iterrows():
    sample = InputExample(texts=[row["premise"], row["hypothesis"]], label=int(row["label"]))
    train_samples.append(sample)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.ContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

dev_sentences1 = []
dev_sentences2 = []
dev_labels = []

for idx, row in df_val.iterrows():
    dev_sentences1.append(row["premise"])
    dev_sentences2.append(row["hypothesis"])
    dev_labels.append(int(row["label"]))

binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=binary_acc_evaluator,
    epochs=num_epochs,
    warmup_steps=1000,
    output_path=model_save_path,
)

# test
print("========= TESTING =========: ", model_save_path)
model = SentenceTransformer(model_save_path)
y_true = []
y_pred = []
sentences1 = []
sentences2 = []
for idx, row in df_test.iterrows():
    s1, s2, label = row["premise"], row["hypothesis"], row["label"]
    y_true.append(label)
    sentences1.append(s1)
    sentences2.append(s2)
embed1 = model.encode(sentences1, convert_to_tensor=True)
embed2 = model.encode(sentences2, convert_to_tensor=True)
cos_sim = util.cos_sim(embed1, embed2)
y_true = np.array(y_true)
y_pred = cos_sim.diag().cpu().numpy()
score = roc_auc_score(y_true, y_pred)
print("TEST ROC_AUC: ", score)
with open(os.path.join(model_save_path, "test_metrics.json"), "w") as fp:
    json.dump(score, fp)