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

## Usage
### Train
