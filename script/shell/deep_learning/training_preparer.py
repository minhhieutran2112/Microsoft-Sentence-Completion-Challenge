# Loading module and downloading required resourced
import os,random,math,sys
import pandas as pd
import numpy as np
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import ne_chunk,ne_chunk_sents,pos_tag,tree2conlltags
from nltk.corpus import words
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
nltk.download('words')
vocab=words.words()
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
spacy_nlp = spacy.load("en_core_web_sm")

# Defining helper variables
device='cuda' if torch.cuda.is_available() else 'cpu'
parentdir='data/raw_data'
TRAINING_DIR=parentdir+'/Holmes_Training_Data'
base_path='data/processed_data/'

# Helper function
def pre_processing(tokenized, e_tagged):
    """
    Performing preprocessing on the tokenized text. I.e. removing number, combining NER, lowercase
    
    Arguments:
    - tokenized: List of list
        Sent-tokenized document, each sentence should be tokenized also
    - e-tagged: Boolean
        Is the NER tag of the tokenizer has ending (E-) tag

    Return:
    - result: List
        Preprocessed tokenized text
    """
    result=[]
    in_ner=False
    for sent in tokenized:
        tmp_sent=[]
        tmp_word=''
        for word in sent:
            # Combining NER tag
            if word.startswith('B-') or word.startswith('S-'):
                in_ner=True
                tag=word[2:]
            elif word.startswith('I-') or word.startswith('E-'):
                continue
            else:
                in_ner=False
            if in_ner:
                tmp_word=tag
            else:
                tmp_word=word.lower()
            # Normalise number
            if sum([char.isdigit() for char in word]) > 0:
                tmp_word='NUM'
            # Appending result to list
            tmp_sent.append(tmp_word)
        result.append(tmp_sent)
    return result
    
def filter_sent(preprocessed,threshold):
    """
    Removing sentences with small number of tokens

    Arguments:
    - preprocessed: List of list
        Preprocessed text by passing text through `pre_processing()`
    - threshold: Int
        Threshold for length of tokenized text
    
    Return: List of list
    """
    return [i for i in preprocessed if len(i)>=threshold]

def spacy_tokenize(document,threshold=10):
    """
    Spacy tokenizer. Passing a document of type string to perform sentence tokenizer; word tokenizer; NER, number, case normalisation. Sentence with smaller number of tokens are also removed.

    Arguments:
    - document: String
        Documents to be processed
    - threshold: Int
        Threshold for length of tokenized text

    Return: List of list
    """
    processed=spacy_nlp(document)
    tokenized=[[token.text if token.ent_type_=='' else token.ent_iob_+'-'+token.ent_type_ for token in sent] for sent in processed.sents]
    return filter_sent(pre_processing(tokenized,False),threshold)

def nltk_tokenize(document,threshold=10):
    """
    NLTK tokenizer. Passing a document of type string to perform sentence tokenizer; word tokenizer; NER, number, case normalisation. Sentence with smaller number of tokens are also removed.

    Arguments:
    - document: String
        Documents to be processed
    - threshold: Int
        Threshold for length of tokenized text

    Return: List of list
    """
    sents=sent_tokenize(document)
    tokenized=[tree2conlltags(ne_chunk(pos_tag(word_tokenize(sent)))) for sent in sents]
    tokenized=[[w if ne=='O' else ne for (w,pos,ne) in sent] for sent in tokenized]
    return filter_sent(pre_processing(tokenized,False),threshold)

def process_training(training_dir):
    """
    Loading all the files in the training directory to a list of string

    Arguments:
    - training_dir: String
        Training directory
    
    Return:
    - documents: List of string
    """
    files=os.listdir(training_dir)
    documents=[]
    for i,filepath in enumerate(files):
        tmp=[]
        try:
            with open(os.path.join(training_dir,filepath)) as instream:
                for line in instream:
                    line=line.strip()
                    if len(line)>0:
                        tmp.append(line)
        except UnicodeDecodeError:
            print("UnicodeDecodeError processing {}: ignoring rest of file".format(filepath))
        documents.append(' '.join(tmp))
    return documents

def get_train_val(documents,val_ratio=0.2):
    """
    Splitting the documents to train and validation set

    Arguments:
    - documents: List of string
        List of documents
    - val_ratio: Float
        Ratio of validation set

    Return:
    - (train,val): Tuple
        train vand validation documents    
    """
    ind=list(range(len(documents)))
    np.random.shuffle(ind)
    train_ind_thresh=int(len(documents)*(1-val_ratio))
    train_ind = ind[:train_ind_thresh]
    val_ind = ind[train_ind_thresh:]
    train, val = [documents[i] for i in train_ind], [documents[i] for i in val_ind]
    return train,val

class qa_dataset(Dataset):
    """
    Pytorch Dataset for question answering task
    """

    def __init__(self, documents, transform=nltk_tokenize, transform_params={'threshold':10}):
        """
        Class initialisation. Adding a question, together with a random mask, correct answer and 4 other imposter answers.

        Arguments:
        - documents: List of string
            List of documents
        - transform: Function
            Preprocess function
        - transform params: Dictionary
            Parameters passed to the preprocess function
        """
        self.train_sents=[]
        tqdm_doc=tqdm(documents)
        for document in tqdm_doc:
            sents=transform(document,**transform_params)
            rand_int=[np.random.randint(0,len(sent)) for sent in sents]
            self.train_sents.extend([(sent,mask,sent[mask],np.random.permutation(list(np.random.choice(vocab,4))+[sent[mask]])) for sent,mask in list(zip(sents,rand_int))])

    def __len__(self):
        return len(self.train_sents)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.train_sents[idx]

if __name__ == '__main__':
    # Creating train and validation dataset
    documents=process_training(TRAINING_DIR)
    spacy_nlp.max_length=len(max(documents,key=lambda x: len(x)))
    train, val = get_train_val(documents)
    ## NLTK train & val
    train_dataset_nltk=qa_dataset(train,nltk_tokenize)
    torch.save(train_dataset_nltk,base_path+'train_dataset_nltk.pt')
    val_dataset_nltk=qa_dataset(val,nltk_tokenize)
    torch.save(val_dataset_nltk,base_path+'val_dataset_nltk.pt')
    ## Spacy train & val
    train_dataset_spacy=qa_dataset(train,spacy_tokenize)
    torch.save(train_dataset_spacy,base_path+'train_dataset_spacy.pt')
    val_dataset_spacy=qa_dataset(val,spacy_tokenize)
    torch.save(val_dataset_spacy,base_path+'val_dataset_spacy.pt')