# Reproducing the experiments
This folder contains the instructions for reproducing the result from the paper **AutoGluon-Multimodal (AutoMM): Empowering 1 Multimodal AutoML with Foundation Models** paper

## Multimodal Classification and Regression
### 1. Installation
### 2. Datasets Preparation
### 3. Run the Benchmark
### 4. Getting Results

## Semantic Matching
### 1. Installation
**AutoGluon:**
Please follow the instruction [here](https://auto.gluon.ai/stable/install.html) to install pytorch and AutoGluon with GPU support.
**Sentence Transformer:**
Please follow the instruction [here](https://www.sbert.net/docs/installation.html#install-sentencetransformers) to install Sentence Transformer.
### 2. Datasets Preparation
Please run the data downloading script to download all the data.
```
cd semantic_matching
bash download_data.sh
```
### 3. Run the Benchmark
To run the experiments of AutoMM, please use the following script. This will run AutoMM on text-text matching, image-image matching and image-text matching tasks with different seeds.
```
bash run_automm.sh
```
Similarly, to run the experiments of Sentence Transformer, please use the script
```
bash run_st.sh
```
### 4. Getting Results
The running scrips will create the `exp` folder for all the experimental results.

## Object Detection
### 1. Installation
### 2. Datasets Preparation
### 3. Run the Benchmark
### 4. Getting Results

## Semantic Segmentation
### 1. Installation
### 2. Datasets Preparation
### 3. Run the Benchmark
### 4. Getting Results


