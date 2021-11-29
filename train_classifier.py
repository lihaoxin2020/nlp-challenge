#%%
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torchtext.legacy.data as lData
from sklearn.model_selection import KFold
import numpy as np
import time
import sys
from tqdm import tqdm
import wandb

from utils.utils import BasicDataset, reset_weights
from utils.utils import TextDataset
from utils.model import Classifier

import datetime
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

import logging
logfile = str('log/log-{}.txt'.format(run_start_time))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logfile),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

wandb.init(project="TextClassifier_11.23")
#%%
def get_fold_data(train_ds, fields, num_folds=10):
    """
    More details about 'fields' are available at 
    https://github.com/pytorch/text/blob/master/torchtext/datasets/imdb.py
    """
    
    kf = KFold(n_splits=num_folds, shuffle=True)
    train_data_arr = np.array(train_ds.examples)

    for train_index, val_index in kf.split(train_data_arr):
        yield(
            TextDataset(train_data_arr[train_index], fields=fields),
            TextDataset(train_data_arr[val_index], fields=fields),
        )

#%%
# tokonize_method = 'basic_english'
def tokenizer(word):
    return list(word)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k_folds = 10
#%%
TRAIN_DATA = "./challenge-data/train.txt"
#%%
train_dataset = BasicDataset(TRAIN_DATA).get_pairs()
print("Total data: ", len(train_dataset))

# train_ds, val_ds = train_test_split(train_dataset, test_size=0.2)
# print('train size: ', len(train_ds), '\tval size: ', len(val_ds))
#%%
TEXT = lData.Field(tokenize = tokenizer, include_lengths = True, batch_first=True)
LABEL = lData.LabelField(dtype = torch.float)

#%%
fields = [('text',TEXT), ('label',LABEL)]
train_ds = TextDataset.splits(fields, train_dataset)[0]

TEXT.build_vocab(train_ds)
LABEL.build_vocab(train_ds)

torch.save(TEXT.vocab, "./TEXT_vocab.pth")

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

learning_rate = 0.001
num_epochs = 10
BATCH_SIZE = 128

config = wandb.config
config.input_dim = INPUT_DIM
config.embedding_dim = EMBEDDING_DIM
config.hidden_dim = HIDDEN_DIM
config.output_dim = OUTPUT_DIM
config.n_layers = N_LAYERS
config.k_folds = k_folds
config.num_epochs = num_epochs
config.init_lr = learning_rate
config.batch_size = BATCH_SIZE

logger.info(f"INPUT_DIM: {INPUT_DIM}\n \
              EMBEDDING_DIM: {EMBEDDING_DIM}\n \
              HIDDEN_DIM: {HIDDEN_DIM}\n \
              N_LAYERS: {N_LAYERS}\n \
              DROPOUT: {DROPOUT}\n \
              KFOLD: {k_folds}\n \
              NUM_EPOCH: {num_epochs}\n \
              INIT_LR: {learning_rate}\n \
              BATCH_SIZE: {BATCH_SIZE}\n")

criterion = nn.BCEWithLogitsLoss()

t = time.time()

# %%
def train(model, iterator, optimizer):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in tqdm(iterator):
        # print(batch)
        text, text_lengths = batch.text
        
        # print('input dim: ', text.shape)
        # print('input labels: ', batch.label)

        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # print('predictions: ', predictions.cpu())
        # print('ite_loss: ', loss.item())
        # print('ite_acc: ', acc.item())
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def evaluate(model, iterator):
    
    epoch_acc = 0
    epoch_loss = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            
            epoch_acc += acc.item()
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def save(model, model_name, val_loss):
    path_last = model_name + "_last.pth"
    torch.save(model.state_dict(), path_last)
    if val_loss[len(val_loss)-1] == min(val_loss):
        torch.save(model.state_dict(), model_name+"_best.pth")
        logger.info("Best model saved. \n")

#%%
# logger.info(f'Embedding size: {TEXT.vocab.vectors.size()}.')

for fold, (train_data, val_data) in enumerate(get_fold_data(train_ds, fields)):
    print('Fold: ', fold)
    print(len(train_data), len(val_data))
    logger.info("***** Running Training *****")
    logger.info(f"Now fold: {fold + 1} / {k_folds}")
    train_iterator, valid_iterator = lData.BucketIterator.splits(
                                                                 (train_data, val_data), 
                                                                 batch_size = BATCH_SIZE,
                                                                 sort_within_batch = True,
                                                                 device = device, 
                                                                )
    # print('train: ', len(train_data))
    # print('val: ', len(val_data))

    model = Classifier(INPUT_DIM, 
                   EMBEDDING_DIM, 
                   HIDDEN_DIM, 
                   OUTPUT_DIM, 
                   N_LAYERS, 
                   BIDIRECTIONAL, 
                   DROPOUT, 
                   PAD_IDX)
    
    model.apply(reset_weights)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, min_lr=1e-6, verbose=True)
    
    loss=[]
    acc=[]
    val_acc=[]
    val_loss = []

    for epoch in range(num_epochs):
        wandb.watch(model)

        train_loss, train_acc = train(model, train_iterator, optimizer)
        valid_loss, valid_acc = evaluate(model, valid_iterator)

        lr_scheduler.step(valid_loss)
        
        logger.info(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% |')
        logger.info(f'| Val. Loss: {train_loss:.3f} | Val Acc: {valid_acc*100:.2f}% |')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}%')
        
        wandb.log({f'Train Loss fold{fold}': train_loss})
        wandb.log({f'Train Acc fold{fold}': train_acc})
        wandb.log({f'Valid Loss fold{fold}': valid_loss})
        wandb.log({f'Valid Acc fold{fold}': valid_acc})

        loss.append(train_loss)
        acc.append(train_acc)
        val_acc.append(valid_acc)
        val_loss.append(valid_loss)

        model_name = f"./runs/classifier_fold{fold}"
        save(model, model_name, val_loss)
        
    print(f'time:{time.time()-t:.3f}')
    
    logger.info('***** Cross Validation Result *****')
    logger.info(f'LOSS: {average(loss)}, ACC: {average(acc)}')
    logger.info(f'VAL LOSS: {average(val_loss)}, VAL ACC: {average(val_acc)}')


# %%

# %%

# %%
