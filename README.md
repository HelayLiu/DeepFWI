# FineWAVE
## FineWAVE: Fine-Grained Warning Verification of Bugs for Automated Static Analysis Tools

This repository contains the dataset and the source code of FineWAVE.

### To run the tool, please follow the instructions:
1. Install the dependencies: `pip install -r requirements.txt`
2. Modify the `config.py` file to set the path to the dataset and the path to the output directory.
 （Data Download Link: [Data](https://drive.google.com/file/d/1BOzSatw2BouVfYroo9LQXXf1VCaY7G-f/view?usp=sharing)）
3. Run the tool: `python main.py`

### If you only want to run the test, we provide the pre-trained model and the test dataset. Please follow the instructions:
1. Install the dependencies: `pip install -r requirements.txt`
2. Download the pre-trained model from [here](https://drive.google.com/file/d/1BezwMMt7JIzAghUg2oFU6VCK7P1ocXfx/view?usp=sharing) and put it in the `model` directory.
3. Modify the `config.py` file to set the path to the test dataset and the path to the output directory.
4. Run the test: `python test.py`


