import pandas as pd
import json
import time
import os
from autogluon.multimodal import MultiModalPredictor
import argparse


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training with Seed')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--dataname', type=str, default='aribnb')
    args = parser.parse_args()

    dataset_path = "image_image_matching/{}".format(args.dataname)
    train_path = os.path.join(dataset_path, "train_processed.csv")
    val_path = os.path.join(dataset_path, "val_processed.csv")
    test_path = os.path.join(dataset_path, "test_processed.csv")
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    exp_log = "exp/automm/image-image/{}/seed_{}".format(args.dataname, args.seed)
    image_col_1 = "image1"
    image_col_2 = "image2"
    label_col = "label"
    match_label = 1

    def path_expander(path, base_folder):
        path_l = path.split(';')
        return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

    for image_col in [image_col_1, image_col_2]:
        train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
        val_data[image_col] = val_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
        test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

    predictor = MultiModalPredictor(
            problem_type="image_similarity",
            presets="best_quality",
            path=exp_log,
            query=image_col_1, # the column name of the first image
            response=image_col_2, # the column name of the second image
            label=label_col, # the label column name
            match_label=match_label, # the label indicating that query and response have the same semantic meanings.
            eval_metric='auc', # the evaluation metric
        )
    
    # Fit the model
    fit_start = time.time()
    predictor.fit(
        train_data=train_data,
        tuning_data=val_data,
        seed=args.seed,
    )
    fit_end = time.time()

    evaluation_start = time.time()
    score = predictor.evaluate(test_data)
    evaluation_end = time.time()

    print("evaluation score: ", score)

    with open(os.path.join(predictor.path, "test_metrics.json"), "w") as fp:
        json.dump(score, fp)
    with open(os.path.join(predictor.path, "time_spent.json"), "w") as fp:
        json.dump({"fit_time": fit_end - fit_start, "evaluation_time": evaluation_end - evaluation_start}, fp)

if __name__ == '__main__':
    main()