# FineWAVE
## FineWAVE: Fine-Grained Warning Verification of Bugs for Automated Static Analysis Tools

This repository contains the dataset and the source code of FineWAVE.

### To run the tool, please follow the instructions:
1. Install the dependencies: `pip install -r requirements.txt`
2. Modify the `config.py` file to set the path to the dataset and the path to the output directory.
3. Run the tool: `python main.py`

### If you only wants to run the test, we provide the pre-trained model and the test dataset. Please follow the instructions:
1. Install the dependencies: `pip install -r requirements.txt`
2. Download the pre-trained model from [here](https://drive.google.com/file/d/1otNmqfVk-rriPlrAgB9CQB9zdqPJjX97/view?usp=drive_link) and put it in the `model` directory.
2. Modify the `config.py` file to set the path to the test dataset and the path to the output directory.
3. Run the test: `python test.py`


