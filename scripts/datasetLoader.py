import nlp
import zipfile
import os

'''
    Script used to flush several datasets into text files.
'''

print([dataset.id for dataset in nlp.list_datasets()])

DATASET = 'text8'  # supported: bookcorpus wikipedia enwik8 text8


output_file = "/data/xp/transformer_inner/text8.10000.modif.txt"

print("Flushing dataset ", DATASET, " into file")

if DATASET == 'bookcorpus':
    data_dir = "/data/huggingface/datasets"
    dataset = nlp.load_dataset('bookcorpus', data_dir=data_dir, cache_dir=data_dir)
    f = open(output_file, "w")
    for entry in dataset['train']:
        text = entry['text']
        if 'isbn' in text.strip():
            f.write("\n")
        f.write(text + "\n")
    f.close()
    print("dataset generated at: ", output_file)

elif DATASET == 'wikipedia':
    data_dir = "/data/huggingface/datasets"
    dataset = nlp.load_dataset('wikipedia', '20200501.en', data_dir=data_dir, cache_dir=data_dir)
    f = open(output_file, "w")
    i = 0
    for entry in dataset['train']:
        i+=1
        if i % 10000 == 0:
            print(str(i),"/",len(dataset['train']))
        title = entry['title'].strip()
        text = entry['text'].strip()
        title = os.linesep.join([s for s in title.splitlines() if s.strip()])
        text = os.linesep.join([s for s in text.splitlines() if s.strip()])
        f.write(title + '\n')
        f.write(text + "\n\n")
        f.flush()
    f.close()
    print("dataset generated at: ", output_file)

elif DATASET == 'enwik8':
    data = zipfile.ZipFile('/data/nlp/enwik8.zip').read('enwik8')
    data_text = data.decode('utf-8')

    f = open(output_file, "w")

    for line in data_text.split('\n'):
        line = line.strip()
        if len(line) == 0:
            continue
        elif line == '<page>':
            f.write("\n")
        f.write(line+'\n')
    f.close()
    print("dataset generated at: ", output_file)

elif DATASET == 'text8':
    with open('/data/nlp/text8', 'r') as fr:

        doc_char_len = 10000

        data = fr.read()
        f = open(output_file, "w")
        s_cur_doc = 0
        for d in data:
            if s_cur_doc >= doc_char_len and d == ' ':
                f.write("\n\n")
                s_cur_doc = 0
            else:
                f.write(d)
                s_cur_doc += 1
        f.close()
        print("dataset generated at: ", output_file)
