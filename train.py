#%%
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
tokenizer = get_tokenizer('basic_english')
# train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])
#%%
TRAIN_JSON = "./challenge-data/train_test.json"
with open(TRAIN_JSON) as train_json:
    json_read = json.load(train_json)

#%%
text_pipeline = lambda x: vocab(tokenizer(x))
# label_pipeline = lambda x: int(x)
