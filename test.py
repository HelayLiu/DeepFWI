import models
import torch
import prepare_data_feature
import train
import configs
import torch.optim as optim
import logging
torch.manual_seed(3407)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import os
if __name__ == '__main__':
    logging.info('start contruct data iterator')
    train_iterator, vaild_iterator, test_iterator = prepare_data_feature.prepare_data()
    logging.info('start contruct model')
    model = models.LSTMwithLSTM11(desc_dim=100002, desc_embedding_dim=512, desc_hidden_dim=512, desc_n_layers=1, 
                 code_dim=100002, code_embedding_dim=512, code_hidden_dim=512, code_n_layers=1,
                 dropout=0., hidden_dim=512, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    if not configs.debug:
        model.cuda()
        model = torch.nn.DataParallel(model.cuda(), device_ids=configs.gpus, output_device=configs.gpus[0])
    model.to(configs.device)                                       
    logging.info('start evaluate')
    train.evaluate(model=model, test_iterator=test_iterator,reload_from_checkpoint=True,
                load_path_checkpoint=os.path.join(configs.save_path,'model.pt'),optimizer=optimizer)