import pandas as pd
import json
import time
import os
from autogluon.multimodal import MultiModalPredictor
import torch
import numpy as np
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training with Seed')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--dataname', type=str, default='MRPC')
    args = parser.parse_args()

    train_path = "text_text_matching/{}/train_processed.csv".format(args.dataname)
    val_path = "text_text_matching/{}/val_processed.csv".format(args.dataname)
    test_path = "text_text_matching/{}/test_processed.csv".format(args.dataname)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    exp_log = "exp/automm/text-text/{}/seed_{}".format(args.dataname, args.seed)

    # Initialize the model
    predictor = MultiModalPredictor(
            problem_type="text_similarity",
            presets="best_quality",
            path=exp_log,
            query="premise", # the column name of the first sentence
            response="hypothesis", # the column name of the second sentence
            label="label", # the label column name
            match_label=1, # the label indicating that query and response have the same semantic meanings.
            eval_metric='auc', # the evaluation metric
        )

    # Fit the model
    fit_start = time.time()
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        seed=args.seed,
    )
    fit_end = time.time()

    evaluation_start = time.time()
    score = predictor.evaluate(test_df)
    evaluation_end = time.time()

    print("evaluation score: ", score)

    with open(os.path.join(predictor.path, "test_metrics.json"), "w") as fp:
        json.dump(score, fp)
    with open(os.path.join(predictor.path, "time_spent.json"), "w") as fp:
        json.dump({"fit_time": fit_end - fit_start, "evaluation_time": evaluation_end - evaluation_start}, fp)

if __name__ == '__main__':
    main()