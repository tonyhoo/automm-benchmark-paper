# Reproducing the experiments
This folder contains the instructions for reproducing the result from the paper **AutoGluon-Multimodal (AutoMM): Empowering 1 Multimodal AutoML with Foundation Models** paper

## Multimodal Classification and Regression
### 1. Installation
### 2. Datasets Preparation
### 3. Run the Benchmark
### 4. Getting Results

## Semantic Matching
### 1. Installation
### 2. Datasets Preparation
### 3. Run the Benchmark
### 4. Getting Results

## Object Detection
### 1. Installation
### 2. Datasets Preparation
### 3. Run the Benchmark
### 4. Getting Results

## Semantic Segmentation

### 1. Installation
**AutoGluon:**
Visit our [Installation Guide](https://auto.gluon.ai/stable/install.html) for detailed instructions, including GPU support, Conda installs, and optional dependencies.

### 2. Datasets Preparation
```python
python3 sem_seg/prepare_semantic_segmentation_datasets.py
```

### 3. Run the Benchmark
**AutoGluon:**
```python
python3 sem_seg/run_autogluon.py --dataset {dataset_name} --output_dir {output_dir} --seed {seed}
```

### 4. Getting Results
**AutoGluon:**
After running the benchmark, the evaluation results of test set are stored in "{output_dir}/metrics.txt".

You can also run the following command to evaluate a checkpoint:
```python
python3 sem_seg/run_autogluon.py --dataset {dataset_name} --output_dir {output_dir} --ckpt_path {ckpt_path} --eval
```



