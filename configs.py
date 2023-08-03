import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
gpus = [0,1]
data_path=''
train_file='train.csv'
valid_file='valid.csv'
test_file='test.csv'
batch_size=64
save_path=''
debug=False
device = torch.device("cuda" if torch.cuda.is_available() and not debug else 'cpu')
save_dataset=False
alpha=0.05
gamma = 2