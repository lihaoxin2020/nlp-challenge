#%%
import torch
import torchtext.legacy.data as lData
import os
from tqdm import tqdm
import random

from utils import TestTextDataset
from utils import tokenizer, TextDataset
from utils import Seq2Seq, Encoder, Decoder

TEST_PATH = "./challenge-data/train_text.txt"
# TEST_PATH = "./challenge-data/test.rand.txt"
TEXT_VOCAB_PATH = "./TextGen_runs/nlp_textGenerator/TEXT_vocab.pth"
LABEL_VOCAB_PATH = "./TextGen_runs/nlp_textGenerator/LABEL_vocab.pth"
MODEL_PATH = "./TextGen_runs/nlp_textGenerator/runs/textGen_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

#%%
test_dataset = TestTextDataset(TEST_PATH).get_data()
print("Total data: ", len(test_dataset))

# train_ds, val_ds = train_test_split(train_dataset, test_size=0.2)
# print('train size: ', len(train_ds), '\tval size: ', len(val_ds))
#%%
TEXT = lData.Field(tokenize=tokenizer, init_token='<sos>', eos_token='<eos>')
LABEL = lData.Field(tokenize=tokenizer, init_token='<sos>', eos_token='<eos>')

TEXT.vocab = torch.load(TEXT_VOCAB_PATH)
LABEL.vocab = torch.load(LABEL_VOCAB_PATH)
# %%
fields = [('text',TEXT), ('label',None)]
test_ds = TextDataset.splits(fields, train_ds=None, test_ds=test_dataset)[0]

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

BATCH_SIZE = 128

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
#%%
# test_iterator = lData.Iterator(
#                                 test_ds, 
#                                 train=False, 
#                                 batch_size = BATCH_SIZE,
#                                 sort_within_batch = False,
#                                 sort=False, 
#                                 device = device, 
#                                 )

# def predict(model, sentence):
#     true, _ = sentence.split('\t')
#     tokenized = tokenizer(true)  #tokenize the sentence 
#     indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
#     print(indexed)
#     length = [len(indexed)]                                    #compute no. of words
#     tensor = torch.LongTensor(indexed).T.to(device)              #convert to tensor
#     tensor = tensor.unsqueeze(0).T                               #reshape in form of batch,no. of words
#     target = torch.full(tensor.shape, 2, dtype=torch.long)
#     # length_tensor = torch.LongTensor(length)                   #convert to tensor
#     prediction = model(tensor, target, teacher_forcing_ratio=0)                  #prediction
#     print(prediction)
#     return prediction
# %%
def predict(model, batch, TEXT, LABEL, device, dst, max_length=500):
    text = batch.text
    with torch.no_grad():
        hidden, cell = model.encoder(text)

    # outputs = [LABEL.vocab.stoi['<sos>']]
    batch_size = batch.batch_size
    outputs = [torch.full(torch.Size([batch_size]), LABEL.vocab.stoi['<sos>'], dtype=torch.int64)]
    
    flag = torch.zeros(batch_size, dtype=torch.int64).to(device)
    for _ in range(max_length):
        previous_word = outputs[-1].to(device)
        # print("prev: ", previous_word.shape)
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            top1 = output.argmax(1)
            # best_guess = output.argmax(1).item()
        # print(top1)
        # print(top1.shape)

        top1 = torch.where(flag == 1, LABEL.vocab.stoi['<pad>'], top1)

        outputs.append(top1)
        flag = torch.where(top1 == LABEL.vocab.stoi['<eos>'], 1, flag)

        # outputs.append(top1)

        # Model predicts it's the end of the sentence
        if flag.sum().item() == batch_size:
            break

    # translated_sentence = [LABEL.vocab.itos[idx] for idx in outputs]
    write_res(outputs[1:], text[1:], dst, TEXT, LABEL)

def tensor2lang(tensor, FIELD):
    sentence = ""
    for i in tensor:
        if i != FIELD.vocab.stoi['<eos>'] and i != FIELD.vocab.stoi['<pad>']:
            sentence += FIELD.vocab.itos[i]
        else:
            break
    return sentence

def write_res(preds, text, path, TEXT, LABEL):
    preds = torch.stack(preds).T
    text = text.T
    for label, input in zip(preds, text):
        pred = tensor2lang(label, LABEL)
        src = tensor2lang(input, TEXT)

        if pred != src:
            with open(path, 'a') as f:
                f.write(pred + '\n')
        else:
            print('Collision! ', src)
#%%
def translate_sentence(model, sentence, TEXT, LABEL, device, max_length=500):
    tokens = [token for token in tokenizer(sentence)]
    tokens.insert(0, TEXT.init_token)
    tokens.append(TEXT.eos_token)
    text_to_indices = [TEXT.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [LABEL.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == LABEL.vocab.stoi["<eos>"]:
            break

    translated_sentence = [LABEL.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:-1]
#%%
dst = "./TextGen_runs/part2test.txt"
f = open(TEST_PATH, 'rb')
lines = f.readlines()
f.close()
for idx, line in tqdm(enumerate(lines)):
    text = line.decode("ascii", errors='ignore')[:-1]
    output = translate_sentence(model, text, TEXT, LABEL, device, max_length=500)
    string = "".join(str(e) for e in output)
    if string == text:
        string = string.split()
        random.shuffle(string)
        string = "".join(str(e) for e in string)
        print('Collision! ', text, string)
    string = '\t' + string + '\n'
    res = string.encode('utf8')
    lines[idx] = line[:-1] + res

with open(dst, 'wb') as f:
    f.writelines(lines)
# for batch in tqdm(test_iterator):
#     text = batch.text
    
#     predict(model, text, TEXT, LABEL, device, dst)
    # translated_sentence = [LABEL.vocab.itos[idx] for idx in outputs]

    # text = batch.text
    # preds = predict(model, text, LABEL, device)
    # print(preds)
    # break
# %%
