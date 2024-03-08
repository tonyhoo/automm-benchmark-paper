#!/bin/bash

# preparing text-text data
mkdir text_text_matching
cd text_text_matching
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/text-text/MRPC_processed.zip
unzip MRPC_processed.zip -d MRPC
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/text-text/MultiNLI_processed.zip
unzip MultiNLI_processed.zip -d MultiNLI
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/text-text/quora_duplicate_processed.zip
unzip quora_duplicate_processed.zip -d quora
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/text-text/SciTail_processed.zip
unzip SciTail_processed.zip -d SciTail
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/text-text/SNLI_processed.zip
unzip SNLI_processed.zip -d SNLI

# preparing image-image data
cd ..
mkdir image_image_matching
cd image_image_matching
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-image/airbnb_duplicate.zip
unzip airbnb_duplicate.zip -d airbnb
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-image/cub200.zip
unzip cub200.zip -d cub200
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-image/iPanda50.zip
unzip iPanda50.zip -d iPanda
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-image/sea_turtle400.zip
unzip sea_turtle400.zip -d sea_turtle
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-image/SOP.zip
unzip SOP.zip -d SOP

# preparing image-text data
cd ..
mkdir image_text_matching
cd image_text_matching
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-text/cub200.zip
unzip cub200.zip -d cub200
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-text/flickr30k.zip
unzip flickr30k.zip -d flickr30k
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-text/flower102.zip
unzip flower102.zip -d flower102
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-text/ImageParagraphCap.zip
unzip ImageParagraphCap.zip -d ImageParagraphCap
wget https://automl-mm-bench.s3.amazonaws.com/semantic_matching_datasets/image-text/mscoco.zip
unzip mscoco.zip -d mscoco

cd ..