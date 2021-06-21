# Loading modules
import warnings
import argparse
warnings.filterwarnings("ignore", category=FutureWarning)
import sys, os
sys.path.append(os.getcwd()) # may need to change this
from training_preparer import *
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext
from gensim.models import KeyedVectors
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='tensorboard')

# get testing data
test_data=pd.read_csv(os.path.join(parentdir,'testing_data.csv'),index_col=0)
test_answer=pd.read_csv(os.path.join(parentdir,'test_answer.csv'),index_col=0).iloc[:,0]

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--embd_model', type=str, default='w2v', help='embedding pretrained model')
parser.add_argument('--embd_name', type=str, default='840B', help='embedding pretrained model name')
parser.add_argument('--embd_dim', type=int, default=300, help='word embedding size')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden size of LSTM')
parser.add_argument('--lstm_layers', type=int, default=1, help='number of LSTM layers')
parser.add_argument('--training_ratio', type=float, default=1., help='ratio of training samples used')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--margin', type=float, default=0.2, help='margin for loss function')
parser.add_argument('--preprocessing', type=str, default='nltk', help='preprocessing type')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()

print(' | '.join(['Batch size:' + str(args.batch_size), 'Embd model:' + str(args.embd_model), 'Embd name:' + str(args.embd_name), 'Embd dim:' + str(args.embd_dim), 'Hidden dim:' + str(args.hidden_dim), 'LSTM layers:' + str(args.lstm_layers), 'Training ratio:' + str(args.training_ratio), 'LR:' + str(args.lr), 'Margin:' + str(args.margin), 'Preprocessing:' + str(args.preprocessing)]))

# set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# answer reference
answers={
    0:'a',
    1:'b',
    2:'c',
    3:'d',
    4:'e'
}

def collate_fn_padd_train(batch):
    """
    Padds batch of variable length (use on the train set only)
    """
    text=[]
    mask=[]
    pos=[]
    neg=[]
    for i in batch:
        text.append(i[0])
        pos.append(i[1])
        mask.append(i[2])
        neg.append(list(i[3][np.where(i[3]!=i[1])[0]]))
    ## get sequence lengths
    lengths = [len(t) for t in text]
    ## padd
    text=[t+['']*(max(lengths)-len(t)) for t in text]
    return text, mask, lengths, pos, neg

def collate_fn_padd_test(batch):
    """
    Padds batch of variable length (use on the val/test set only)
    """
    text=[]
    mask=[]
    option=[]
    label=[]
    for i in batch:
        text.append(i[0])
        option.append(list(i[3]))
        mask.append(i[2])
        label.append(np.where(i[3]==i[1])[0][0])
    ## get sequence lengths
    lengths = [len(t) for t in text]
    ## padd
    text=[t+['']*(max(lengths)-len(t)) for t in text]
    return text, mask, lengths, option, label

def validate(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs in dataloader:
            inputs,options,labels=inputs[:-2],inputs[-2],inputs[-1]
            a_ind=torch.tensor([model.get_index_sent(i) for i in options],dtype=torch.long).to(device)
            labels=torch.tensor(labels).to(device)

            # forward pass
            q_encoded=model(inputs)
            a_encoded=[model.embeddings(a_ind[:,i]) for i in range(a_ind.shape[1])]
            sim=torch.stack([cos_sim(q_encoded,i) for i in a_encoded],dim=1)
            predicted=torch.max(sim,dim=1)[1]

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def get_accuracy(prediction,labels):
    return sum(prediction==labels)/len(labels)

def validate_test(model):
    # Get prediction on testing data
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        result = []
        for index, row in test_data.iterrows():
            question=row[0]
            if args.preprocessing == 'nltk':
                tokenized=tree2conlltags(ne_chunk(pos_tag(word_tokenize(question))))
                tokenized=[w if ne=='O' or w == '_____' else ne for (w,pos,ne) in tokenized]
                tokenized=pre_processing([tokenized],False)[0]
            elif args.preprocessing == 'spacy':
                processed=spacy_nlp(question.replace('_____','MASK'))
                tokenized=['_____' if token.text == 'MASK' else token.text if token.ent_type_=='' else token.ent_iob_+'-'+token.ent_type_ for sent in processed.sents for token in sent]
                tokenized=pre_processing([tokenized],False)[0]
            masked_index=tokenized.index('_____')
            inputs=[[tokenized],[masked_index],[len(tokenized)]]
            a_ind=torch.tensor([model.get_index_sent([i])[0] for i in row[1:].tolist()],dtype=torch.long).to(device)
            
            # forward pass
            q_encoded=model(inputs)
            a_encoded=model.embeddings(a_ind)
            sim=torch.stack([cos_sim(q_encoded,i.view(1,-1)) for i in a_encoded],dim=1)
            predicted=torch.max(sim,dim=1)[1][0]
    
            result.append(answers[predicted.item()])
        result = pd.Series(result,index=test_data.index)
    return get_accuracy(result,test_answer)

# Language model main class
cos_sim = torch.nn.CosineSimilarity()
class my_languagemodel(nn.Module):
    def __init__(self,embedding_weight,embedding_dim,word2index_dict,hidden_state_dim,lstm_layers):
        super().__init__()
        num_directions = 2
        self.hidden_state_dim=hidden_state_dim
        # RNN
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_state_dim,
                            num_layers=lstm_layers,
                            bidirectional=True,
                            batch_first=True)
        # FC
        self.fc1 = nn.Linear(hidden_state_dim * num_directions, embedding_dim)
        # helper
        self.embeddings=nn.Embedding.from_pretrained(embedding_weight)
        self.word2index_dict=word2index_dict
        self.relu = nn.ReLU()          
        self.vocab = self.word2index_dict.keys()
        self.unk_id = max(self.word2index_dict.values())+1
    
    def get_index_word(self,word):
        if word not in self.vocab:
            if word.lower() not in self.vocab:
                return self.unk_id
            else:
                return self.word2index_dict[word.lower()]
        else:
            return self.word2index_dict[word]

    def get_index_sent(self,sent):
        return [self.get_index_word(w) for w in sent]

    def forward(self, inputs):
        # encode texts and options
        texts,lengths,masks=[],[],[]
        for text, mask, length in zip(*inputs):
            text = self.get_index_sent(text)
            texts.append(text)
            lengths.append(length)
            masks.append(mask)
        texts=torch.tensor(texts,dtype=torch.long).to(device)
        lengths=torch.tensor(lengths,dtype=torch.long)
        masks=torch.tensor(masks,dtype=torch.long).to(device)

        # Phase 1 - Apply RNN
        ## embedding & lstm
        embedded_texts = self.embeddings(texts)
        packed_embedded = pack_padded_sequence(embedded_texts, lengths, batch_first=True, enforce_sorted=False) 
        output, _ = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(output, batch_first=True)
        ## Get hidden state of forward and backward lstm
        forward_cond=torch.where(masks-1<0,0,masks-1)
        backward_cond=torch.where(masks+1>=lengths.cuda(),masks,masks+1)
        out_forward = output[range(len(output)), forward_cond, :self.hidden_state_dim]
        out_reverse = output[range(len(output)), backward_cond, self.hidden_state_dim:]
        cat = torch.cat((out_forward, out_reverse), 1)

        ## Phase 2 - Apply FC
        rel = self.relu(cat)
        preds = self.fc1(rel)

        return preds

# Initiating train, val set
if args.preprocessing == 'nltk':
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize
    from nltk import ne_chunk,ne_chunk_sents,pos_tag,tree2conlltags
    train_path=base_path+'train_dataset_nltk.pt'
    val_path=base_path+'val_dataset_nltk.pt'
elif args.preprocessing == 'spacy':
    import spacy
    from spacy.lang.en import English
    from spacy.tokenizer import Tokenizer
    spacy_nlp = spacy.load("en_core_web_sm")
    train_path=base_path+'train_dataset_spacy.pt'
    val_path=base_path+'val_dataset_spacy.pt'

train=torch.load(train_path)
ind=list(range(len(train)))
np.random.shuffle(ind)
train.train_sents = [train.train_sents[i] for i in ind[:int(len(train)*args.training_ratio)]]
val=torch.load(val_path)
batch_size=args.batch_size
train_loader=DataLoader(train,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate_fn_padd_train)
train_loader_val=DataLoader(train,batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn_padd_test)
val_loader=DataLoader(val,batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn_padd_test)

# Initiating model
if args.embd_model=='w2v':
    embd_path=base_path+'GoogleNews-vectors-negative300.bin.gz'
    embd_model=KeyedVectors.load_word2vec_format(embd_path,binary=True)
    embedding_weight=embd_model.vectors
    embedding_weight=np.vstack([embedding_weight,np.zeros((1,300))])
    embedding_weight=torch.tensor(embedding_weight,dtype=torch.float)
    word2index_dict = {token: token_index for token_index, token in enumerate(embd_model.index2word)}
else:
    cache=base_path+'.vector_cache'
    if args.embd_model=='glove':
        embd_model=torchtext.vocab.GloVe(name=args.embd_name,dim=args.embd_dim,cache=cache)
    elif args.embd_model=='fasttext':
        embd_model=torchtext.vocab.FastText(language=args.embd_name,cache=cache)
    embedding_weight=embd_model.vectors
    embedding_weight=torch.cat([embedding_weight,torch.zeros((1,embedding_weight.shape[1]))])
    word2index_dict=embd_model.stoi

hidden_state_dim=args.hidden_dim
lstm_layers=args.lstm_layers
embedding_dim=args.embd_dim
lm=my_languagemodel(embedding_weight,embedding_dim,word2index_dict,hidden_state_dim,lstm_layers).to(device)
writer_main =' | '.join(['Batch size:' + str(args.batch_size), 'Embd model:' + str(args.embd_model), 'Embd name:' + str(args.embd_name), 'Embd dim:' + str(args.embd_dim), 'Hidden dim:' + str(args.hidden_dim), 'LSTM layers:' + str(args.lstm_layers), 'Training ratio:' + str(args.training_ratio), 'LR:' + str(args.lr), 'Margin:' + str(args.margin), 'Preprocessing:' + str(args.preprocessing)])
acc_train=validate(lm,train_loader_val)
acc_val=validate(lm,val_loader)
acc_test=validate_test(lm)
writer.add_scalar(writer_main+'/train',acc_train,0)
writer.add_scalar(writer_main+'/val',acc_val,0)
writer.add_scalar(writer_main+'/test',acc_test,0)

# Define optimizer and loss
lr=args.lr
loss_margin=args.margin
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lm.parameters()), lr=lr)

# Train model
num_epochs=args.n_epochs
epochs=tqdm(range(num_epochs))
lm.train()      
for e in epochs:
    lm.train()      
    for i,inputs in enumerate(train_loader):
        inputs,pos,neg=inputs[:-2],inputs[-2],inputs[-1]
        neg=np.random.randint(0,lm.unk_id+1,len(neg))
        a_pos_ind=torch.tensor(lm.get_index_sent(pos),dtype=torch.long).to(device)
        a_neg_ind=torch.tensor(neg,dtype=torch.long).to(device)

        # forward pass
        q_encoded=lm(inputs)
        a_pos=lm.embeddings(a_pos_ind)
        a_neg=lm.embeddings(a_neg_ind)

        pos_sim=cos_sim(q_encoded,a_pos)
        neg_sim=cos_sim(q_encoded,a_neg)

        loss=torch.mean(torch.clamp(loss_margin - pos_sim + neg_sim, min=0))

        # backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            epochs.set_description(
                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    e + 1, num_epochs, i + 1, len(train_loader), loss.item()))
    
    acc_train=validate(lm,train_loader_val)
    acc_val=validate(lm,val_loader)
    acc_test=validate_test(lm)

    writer.add_scalar(writer_main+'/train',acc_train,e+1)
    writer.add_scalar(writer_main+'/val',acc_val,e+1)
    writer.add_scalar(writer_main+'/test',acc_test,e+1)
    
writer.close()