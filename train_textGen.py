#%%
import torch
import torch.nn as nn
import torchtext.legacy.data as lData
from sklearn.model_selection import KFold
import numpy as np
import time
import sys
from tqdm import tqdm
import wandb

from utils.utils import TextDataset, reset_weights
from utils.utils import TextGenDataset
from utils.model import Encoder, Decoder, Seq2Seq

import datetime
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

import logging
logfile = str('./textGen/log/log-{}.txt'.format(run_start_time))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logfile),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

wandb.init(project="TextGenerator_11.29")
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
train_dataset = TextGenDataset(TRAIN_DATA).get_pairs()
print("Total data: ", len(train_dataset))

# train_ds, val_ds = train_test_split(train_dataset, test_size=0.2)
# print('train size: ', len(train_ds), '\tval size: ', len(val_ds))
#%%
TEXT = lData.Field(tokenize=tokenizer, init_token='<sos>', eos_token='<eos>')
LABEL = lData.Field(tokenize=tokenizer, init_token='<sos>', eos_token='<eos>')

#%%
fields = [('text',TEXT), ('label',LABEL)]
train_ds = TextDataset.splits(fields, train_dataset)[0]

TEXT.build_vocab(train_ds, min_freq=2)
LABEL.build_vocab(train_ds, min_freq=2)

torch.save(TEXT.vocab, "./textGen/TEXT_vocab.pth")
torch.save(LABEL.vocab, "./textGen/LABEL_vocab.pth")

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
BIDIRECTIONAL = True
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
TEXT_PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
LABEL_PAD_IDX = LABEL.vocab.stoi[LABEL.pad_token]

learning_rate = 0.001
num_epochs = 10
BATCH_SIZE = 64

config = wandb.config
config.input_dim = INPUT_DIM
config.enc_embedding_dim = ENC_EMB_DIM
config.dec_embedding_dim = DEC_EMB_DIM
config.hidden_dim = HIDDEN_DIM
config.output_dim = OUTPUT_DIM
config.n_layers = N_LAYERS
config.k_folds = k_folds
config.num_epochs = num_epochs
config.init_lr = learning_rate
config.batch_size = BATCH_SIZE

logger.info(f"INPUT_DIM: {INPUT_DIM}\n \
              ENC_EMB_DIM: {ENC_EMB_DIM}\n \
              DEC_EMB_DIM: {DEC_EMB_DIM}\n \
              HIDDEN_DIM: {HIDDEN_DIM}\n \
              N_LAYERS: {N_LAYERS}\n \
              ENC_DROPOUT: {ENC_DROPOUT}\n \
              DEC_DROPOUT: {DEC_DROPOUT}\n \
              KFOLD: {k_folds}\n \
              NUM_EPOCH: {num_epochs}\n \
              INIT_LR: {learning_rate}\n \
              BATCH_SIZE: {BATCH_SIZE}\n")

criterion = nn.CrossEntropyLoss(ignore_index = LABEL_PAD_IDX)

t = time.time()

# %%
def train(model, iterator, optimizer, clip):
    epoch_loss = 0    
    model.train()
    
    for batch in tqdm(iterator):
        # print(batch)
        text = batch.text
        label = batch.label

        # print(text)
        # print(label)
        # print('input dim: ', text.shape)
        # print('input labels: ', batch.label)

        optimizer.zero_grad()
        output = model(text, label)
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        label = label[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, label)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            text = batch.text
            label = batch.label

            output = model(text, label, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            label = label[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, label)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

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
                                                                #  sort_within_batch = True,
                                                                 device = device, 
                                                                )
    # print('train: ', len(train_data))
    # print('val: ', len(val_data))

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    
    model.apply(reset_weights)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, min_lr=1e-6, verbose=True)
    
    loss=[]
    val_loss = []

    CLIP = 1

    for epoch in range(num_epochs):
        wandb.watch(model)

        train_loss = train(model, train_iterator, optimizer, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        lr_scheduler.step(valid_loss)
        
        logger.info(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} |')
        logger.info(f'| Val. Loss: {train_loss:.3f} |')

        print(f'\tTrain Loss: {train_loss:.3f} |')
        print(f'\tValid Loss: {valid_loss:.3f} |')
        
        wandb.log({f'Train Loss fold{fold}': train_loss})
        wandb.log({f'Valid Loss fold{fold}': valid_loss})

        loss.append(train_loss)
        val_loss.append(valid_loss)

        model_name = f"./textGen/runs/classifier_fold{fold}"
        save(model, model_name, val_loss)
        
    print(f'time:{time.time()-t:.3f}')


# %%

# %%

# %%
