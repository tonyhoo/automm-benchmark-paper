# Reproducing the experiments
This folder contains the instructions for reproducing the result from the paper **AutoGluon-Multimodal (AutoMM): Empowering 1 Multimodal AutoML with Foundation Models** paper

## Multimodal Classification and Regression
The benchmarks for this set of tasks can be run locally on each individual dataset, or run on AWS Batch on multiple datasets at the same time.
### 1. Installation
The below installation is based on a linux machine. It is recommended to run on a GPU machine. 

```
# clone and checkout to branch
git clone https://github.com/suzhoum/autogluon-bench.git
cd autogluon-bench
git checkout automl_paper

# create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```
### 2. Datasets Preparation
Dataloaders of different modality combinations have been provided for basica data cleaning, pre-processing and splitting, and they are stored in `sample_configs/dataloaders/{modality}_dataloader.py`. Extra processing of the data for different frameworks (e.g. loading data of different modalities in Auto-Keras) is done in `src/autogluon/bench/frameworks/{framework_name}/exec.py`. 
### 3. Run the Benchmark
1. Run the benchmark in `local` mode, only one dataset can be run each time.
    - update local config file `sample_configs/paper_{modality}_cloud_configs.yaml`
        - specify `module`, for benchmarking on `AutoGluon`, please specify `multimodal`; for benchmarking on `Auto-Keras`, please specify `autokeras`
        - specify `framework`
        - specify a `dataset_name`
    - run the following command:

```
agbench run paper_{modality}_local_configs.yaml
```

2. Run the benchmark in `AWS` mode, multiple instances will be started based on the number of datasets specified. Each instance runs benchmark on one dataset.
    - follow [installation guide](https://github.com/autogluon/autogluon-bench?tab=readme-ov-file#run-benchmarks-on-aws) to setup AWS CDK, install `awscliv2`
    - assume AWS credentials in your terminal
    - update cloud config file `sample_configs/paper_{modality}_local_configs.yaml`
        - update `CDK_DEPLOY_ACCOUNT`, `CDK_DEPLOY_REGION` and `METRICS_BUCKET` values
        - specify `module`, for benchmarking on `AutoGluon`, please specify `multimodal`; for benchmarking on `Auto-Keras`, please specify `autokeras`
        - specify `framework`
        - specify `constraint`
        - choose the list of `dataset_name`
    - run the following command:

```
agbench run paper_{modality}_cloud_configs.yaml
```

Choices of local config files:

- sample_configs/paper_image_local_configs.yaml
- sample_configs/paper_text_local_configs.yaml
- sample_configs/paper_text_tabular_local_configs.yaml

Choices of cloud config files:

- sample_configs/paper_image_cloud_configs.yaml
- sample_configs/paper_text_cloud_configs.yaml
- sample_configs/paper_text_tabular_cloud_configs.yaml


### 4. Getting Results
- `local` mode - the metrics will be saved under `{root_dir}/{benchmark_name}_{TIMESTAMP}/results`
- `aws` mode - the metrics will be saved under `s3://{METRICS_BUCKET}/{module}/{benchmark_name}_{TIMESTAMP}/`

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
```
mim install "mmcv==2.1.0"
pip install "mmdet==3.2.0"
pip install pycocotools
```
### 2. Datasets Preparation
```
wget https://automl-mm-bench.s3.amazonaws.com/object_detection/benchmark/AutoMLDetBench.zip
```
And unzip this somewhere as `<Data_Dir>`.
### 3. Run the Benchmark
```
python3 <Installed_Dir>/autogluon/examples/automm/object_detection/benchmarking.py \
   --benchmark_root <Data_Dir>/AutoMLDetBench \
   --dataset_name <dataset_name> \
   --seed <seed>
```
### 4. Getting Results
Results will be printed.

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



