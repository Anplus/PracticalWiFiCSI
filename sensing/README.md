# Widar 3.0 WiFi based Gesture Identification

## Requirements

- MATLAB R2016a or later
- pytorch
- scipy
- numpy
- einops
- scikit-learn

## Data Generation

1. Download the Widar 3.0 TPAMI 2021 dataset from [here](https://tns.thss.tsinghua.edu.cn/widar3.0/). Extract the dataset to a folder on your local machine.

```bash
curl --ftp-pasv --ftp-skip-pasv-ip \
     -u widarftp:widar2019 \
     ftp://166.111.80.127:40121/CSI/CSI_20181109.zip \
     -o CSI_20181109.zip
```

2. Run the MATLAB script `DFSExtractionCode/generate_dataset.m`. Before running, open the script and set the `input_root` variable to the folder containing your `.dat` files. The script converts `.dat` → `.mat` and saves the results to the configured output folder.

3. Create a `shuffle.txt` file that lists all dataset file paths (one per line). Use relative paths if you plan to move the dataset.

## Training and testing

build environment and install dependencies:

```bash
conda create -n myenv python=3.11
conda activate myenv
conda install numpy scipy pandas matplotlib scikit-learn
conda install pytorch torchvision torchaudio
pip install -r requirements.txt
```

1. Run the training script `train.py` with the appropriate model type.
2. Run the testing script `test.py` with the appropriate model type.
