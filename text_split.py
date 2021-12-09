PATH = "./challenge-data/train.txt"
TAR_PATH = "./challenge-data/train_text.txt"

tar = open(TAR_PATH, "wb")
with open(PATH, 'rb') as f:
    for line in f:
        tar.write(line)
tar.close()