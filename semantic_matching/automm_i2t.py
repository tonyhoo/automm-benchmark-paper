import os
import pandas as pd
import json
import time
from autogluon.multimodal import MultiModalPredictor
import argparse
from utils import compute_sims, t2i, i2t

IMAGE_DUPLICATE = {
    'cub200': 10,
    'flickr30k': 5,
    'flower102': 10,
    'ImageParagraphCap': 1,
    'mscoco': 5,
}

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training with Seed')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--dataname', type=str, default='cub200')
    args = parser.parse_args()

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
    exp_log = "exp/automm/image-text/{}/seed_{}".format(args.dataname, args.seed)

    def path_expander(path, base_folder):
        path_l = path.split(';')
        return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    val_data[image_col] = val_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

    test_image_data = pd.DataFrame({image_col: test_data[image_col].unique().tolist()})
    test_text_data = pd.DataFrame({text_col: test_data[text_col].unique().tolist()})
    test_data_with_label = test_data.copy()
    test_label_col = "relevance"
    test_data_with_label[test_label_col] = [1] * len(test_data)

    predictor = MultiModalPredictor(
                presets='best_quality',
                path=exp_log,
                query=text_col,
                response=image_col,
                problem_type="image_text_similarity",
                eval_metric="recall",
            )

    predictor.fit(
                train_data=train_data,
                tuning_data=val_data,
                seed=args.seed,
            )

    test_text_data = test_data[text_col].to_frame()
    test_image_data = test_data[image_col].to_frame()
    test_image_data = test_image_data.iloc[::image_dup]
    text_emb = predictor.extract_embedding(test_text_data, as_tensor=True)
    image_emb = predictor.extract_embedding(test_image_data, as_tensor=True)
    sims = compute_sims(image_emb, text_emb)
    num_text = len(test_text_data)
    num_image = len(test_image_data)
    print("==== Text to Image ====")
    text2image_score = t2i(num_text, num_image, sims.T, image_dup)
    print("==== Image to Text ====")
    image2text_score = i2t(num_text, num_image, sims, image_dup)
    print(f"txt_to_img_scores: {text2image_score}")
    print(f"img_to_txt_scores: {image2text_score}")

    with open(os.path.join(predictor.path, "test_metrics.json"), "w") as fp:
        json.dump({"text2img_score:": text2image_score, "image2text_score:": image2text_score}, fp)   

if __name__ == '__main__':
    main()