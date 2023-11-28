# ukp_entrance_test
## 1. Environment Installation
### 1.1 Install python environment
Install environment with conda:
```bash
git clone https://github.com/ututono/ukp_entrance_test.git
conda env create -n nlp_mini python=3.11
conda activate nlp_mini
pip install -r requirements.txt 
```
Note that if you are using Windows, you might need to select suitable version of PyTorch from [its website](https://pytorch.org).

### 1.2 Create `.env` file to root directory
The `.env` file should contain the following variables:
```bash
ROOT_PATH = <path to root directory>
```
Note that if you are using Windows, you should use double backslash `\\` instead of single backslash `\` in the path.

### 1.3 Download embedding file
Download embedding file and unzip it to `pretrained` directory under root directory. 


The project should be structured as follows:
```txt
- root
  - pretrained
    - glove.6B.100d.txt
  - src
    - ...
  - .env
  - README.md
  - requirements.txt
```


## 2. Usage
Move to the root directory of this repository.
### 2.1 Train
Run the following command to train the model:
```bash
python -m src.main \
  --batch_size 1 \
  --epochs 20 \
  --learning_rate 0.001 \
  --optimizer adam \
  --loss cross_entropy \
  --mode train \
```
There is a template script in `scripts` directory. You can modify it and run it as follows:
```bash
bash scripts/train.sh
```
The model will be saved to a timestamped directory under `output/checkpoint`, for example `output/checkpoint/1970_01_01-00_00_00`.

### 2.2 Test
Run the following command to test the model:
```bash
python -m src.main \
  --batch_size 1 \
  --loss cross_entropy \
  --checkpoint 2023_11_27-17_16_04 \
  --mode test \
```
There is a template script in `scripts` directory. You can modify it and run it as follows:
```bash
bash scripts/evaluate.sh
```




