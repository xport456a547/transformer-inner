from tokenizers import ByteLevelBPETokenizer

DATA = "../data/sample.txt"
VOCAB_SIZE = 256 
SAVING_PATH = "byte_tokenizer"


# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Training
tokenizer.train([DATA], vocab_size=VOCAB_SIZE, special_tokens=["<pad>", "<mask>", "<unk>", "<s>", "</s>"])

tokenizer.save_model(".", SAVING_PATH)