#%%
import torch
import torchtext.legacy.data as lData
import os
from tqdm import tqdm

from utils import TestDataset
from utils import tokenizer, TextDataset
from utils import Classifier

TEST_PATH = "./challenge-data/train.txt"
# TEST_PATH = "./challenge-data/test.rand.txt"
TEXT_VOCAB_PATH = "./Classifier_runs/TEXT_vocab.pth"
LABEL_VOCAB_PATH = "./Classifier_runs/LABEL_vocab.pth"
MODEL_PATH = "./Classifier_runs/runs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = []

def load_model(path, device):
    paths = os.listdir(path)
    model = Classifier(INPUT_DIM, 
                        EMBEDDING_DIM, 
                        HIDDEN_DIM, 
                        OUTPUT_DIM, 
                        N_LAYERS, 
                        BIDIRECTIONAL, 
                        DROPOUT, 
                        PAD_IDX)
    for i in paths:
        if "best" in i:
            model.load_state_dict(torch.load(os.path.join(path, i)))
            model.to(device)
            model.eval()
            models.append(model)

# device = torch.device("cpu")
#%%
test_dataset = TestDataset(TEST_PATH).get_data()
print("Total data: ", len(test_dataset))

TEXT = lData.Field(tokenize = tokenizer, include_lengths = True, batch_first=True)
LABEL = lData.LabelField(dtype = torch.float)

TEXT.vocab = torch.load(TEXT_VOCAB_PATH)
LABEL.vocab = torch.load(LABEL_VOCAB_PATH)
# %%
fields = [('text',TEXT), ('label',None)]
# fields = [('text',TEXT)]
test_ds = TextDataset.splits(fields, train_ds=None, test_ds=test_dataset)[0]

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
BATCH_SIZE = 256

load_model(MODEL_PATH, device)
#%%
# torch_test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
#%%
test_iterator = lData.Iterator(
                                test_ds, 
                                train=False, 
                                batch_size = BATCH_SIZE,
                                sort_within_batch = False,
                                sort=False, 
                                device = device, 
                                )
#%%
def predict(models, batch, dst):
    batch_size = batch.batch_size
    preds = torch.zeros(batch_size, device=device)
    for model in models:
        text, length = batch.text
        with torch.no_grad():
            pred = model(text, length, train=False).squeeze(1)
            pred = torch.sigmoid(pred)
            preds += pred
            preds /= len(models)
    labels = torch.round(preds).to(torch.int32).cpu()
    count = write_res(labels, dst, LABEL)
    return count

def write_res(labels, dst, LABEL):
    count = 0
    with open(dst, 'a') as f:
        for label in labels:
            content = 'B' if LABEL.vocab.itos[label] else 'A'
            if content == 'A':
                count += 1
            # print(count)
            f.write(content + '\n')
    return count

#%%
dst = './Classifier_runs/part1train.txt'
count = 0
for batch in tqdm(test_iterator):
    count += predict(models, batch, dst)

print(count / len(test_dataset))
#%%
# def predict(model, sentence):
#     true, false = sentence.split('\t')
#     false = false[:-1]
#     tokenized = tokenizer((true+false).lower())  #tokenize the sentence 
#     indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
#     print(indexed)
#     length = [len(indexed)]                                    #compute no. of words
#     tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
#     tensor = tensor.unsqueeze(0)                             #reshape in form of batch,no. of words
#     length_tensor = torch.LongTensor(length)                   #convert to tensor
#     prediction = model(tensor, length_tensor)                  #prediction 
#     return torch.sigmoid(prediction).item()
# %%
# text = "Energy security, rising energy costs and climate change are perhaps the greatest challenges facing Europe in the 21st century.	Energy security, rising energy costs and climate change are the perhaps greatest challenges facing Europe in the 21st century.\n"
# print(text)
# predict(model, text)
# # %%
# torch.sigmoid(pred)
# %%
