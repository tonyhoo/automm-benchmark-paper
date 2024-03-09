from PIL import Image
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from sentence_transformers import util
from torch.utils.data import DataLoader
import os
import pandas as pd
import random
import numpy as np
import json
import torch
import argparse
from utils import t2i, i2t, compute_sims, truncate_sentence

IMAGE_DUPLICATE = {
    'cub200': 10,
    'flickr30k': 5,
    'flower102': 10,
    'ImageParagraphCap': 1,
    'mscoco': 5,
}

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch Training with Seed')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--dataname', type=str, default='cub200')
args = parser.parse_args()

set_seed(args.seed)

model = SentenceTransformer('clip-ViT-L-14')
tokenizer = model._first_module().processor.tokenizer
num_epochs = 10
train_batch_size = 24
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
margin = 0.5

dataset_path = "image_text_matching/{}".format(args.dataname)
train_path = os.path.join(dataset_path, "train_processed.csv")
val_path = os.path.join(dataset_path, "val_processed.csv")
test_path = os.path.join(dataset_path, "test_processed.csv")
train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)
image_col = "image"
text_col = "caption"
image_dup = IMAGE_DUPLICATE[args.dataname]

model_save_path = "exp/st/image-text/{}/seed_{}".format(args.dataname, args.seed)
os.makedirs(model_save_path, exist_ok=True)

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
val_data[image_col] = val_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

train_data[text_col] = train_data[text_col].apply(lambda x: truncate_sentence(x, tokenizer))
val_data[text_col] = val_data[text_col].apply(lambda x: truncate_sentence(x, tokenizer))
test_data[text_col] = test_data[text_col].apply(lambda x: truncate_sentence(x, tokenizer))

train_samples = []
for idx, row in train_data.iterrows():
    image = Image.open(row[image_col])
    caption = row[text_col]
    sample = InputExample(texts=[image, caption], label=1)
    train_samples.append(sample)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesSymmetricRankingLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=1000,
    output_path=model_save_path,
)

print("========= TESTING: {} =========".format(model_save_path))

model = SentenceTransformer(model_save_path)

test_text_data = test_data[text_col]
test_image_data = test_data[image_col]
test_image_data = test_image_data.iloc[::image_dup]
text_emb = model.encode(test_text_data.tolist(), convert_to_tensor=True)
test_image_data_list = [Image.open(x) for x in test_image_data.tolist()]
image_emb = model.encode(test_image_data_list, convert_to_tensor=True)
num_text = len(test_text_data)
num_image = len(test_image_data)
sims = compute_sims(image_emb, text_emb)
print("==== Text to Image ====")
text2image_score = t2i(num_text, num_image, sims.T, image_dup)
print("==== Text 2 Image ====")
image2text_score = i2t(num_text, num_image, sims, image_dup)
### Paper R@K Metric ###

with open(os.path.join(model_save_path, "test2image_metrics_paper.json"), "w") as fp:
    json.dump(text2image_score, fp)
with open(os.path.join(model_save_path, "image2text_metrics_paper.json"), "w") as fp:
    json.dump(image2text_score, fp)

print(f"txt_to_img_scores: {text2image_score}")
print(f"img_to_txt_scores: {image2text_score}")