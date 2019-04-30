"""This class implements the state-of-the-art Dual Co-Matching Network on 
the RACE dataset. See https://arxiv.org/pdf/1901.09381.pdf for 
details.

Implementation by Samuel Oshay for University of Pennsylvania's 
CIS-700 Deep Learning, Spring 2019.

Teammates: Leonardo Murri, Dewang Sultania, Peyman Norouzi."""

# ==========================================================
# IMPORTS
# ==========================================================

# !pip install pytorch-pretrained-bert

from pytorch_pretrained_bert import BertModel, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert.optimization import BertAdam
import torch
import torch.nn as nn
import json, pickle
from random import shuffle

# ==========================================================
# DUAL CO-MATCHING NETWORK MODEL
# ==========================================================

class DCMN(nn.Module):
    '''The main Dual Co-Matching Network model.'''
    
    def __init__(self, batch_size):
        super(DCMN, self).__init__()
        self.bs = batch_size
        # CLS and SEP tokens used for BERT input segmentation
        self.cls = self.gen_token_vector(101)
        self.sep = self.gen_token_vector(102)
        # Load pre-trained BERT. If resources exist, can be made 'bert-large-uncased'.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Matching units for passage-option and passage-question, respectively.
        self.match_opt = Matching_Attention()
        self.match_que = Matching_Attention()
        # Single final pooling unit
        self.pool = Final_Pooling()
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, art, q, a, b, c, d, y=None):
        '''In our forward pass, we obtain the full hidden
        state for our article and our question from BERT. The
        shape of this state is:
        
        [batch size, num of words, size of hidden state].
        
        Inputs:
            art: The passage as BERT tokens. Shape is 
                 [batch size, article length, size of hidden state].
            q: The question as BERT tokens. Shape is 
                 [batch size, question length, size of hidden state].
            a, b, c, d: Options a, b, c, d, respectively. Shape is:
                 [batch size, option length, size of hidden state].
            y: Optional label, one of [0, 1, 2, 3].    
            
        Outputs:
            If y is not provided, output is the probabilities of 
            options a, b, c, and d, respectively. Shape is:
                [batch size, 4].
                
            If y is provided, output is cross-entropy loss and 
            batch accuracy.'''
       
        # Generate article hidden state.
        
        # Append CLS and SEP tokens
        art = torch.cat((self.cls, art, self.sep), dim=1) 
        # Generate ids and masks
        ids, mask = self.get_comps(art.shape[1])
        # Forward pass through BERT
        art, _ = self.bert(art, ids, mask, output_all_encoded_layers=False)
        del ids, mask
        
        # Generate question hidden state.
        
        q = torch.cat((self.cls, q, self.sep), dim=1)
        ids, mask = self.get_comps(q.shape[1])
        q, _ = self.bert(q, ids, mask, output_all_encoded_layers=False)
        del ids, mask
        
        # Match question to passage.

        SQ, SPP = self.match_que.forward(q, art)
        del q
        
        # Use matching unit and pooling unit to generate option probabilites.
        
        a = self.eval_option(a, art, SQ, SPP)
        b = self.eval_option(b, art, SQ, SPP)
        c = self.eval_option(c, art, SQ, SPP)
        d = self.eval_option(d, art, SQ, SPP)
        del art, SQ, SPP
        
        final = torch.cat((a,b,c,d), dim=1)
        del a, b, c, d
        
        # For convenience, loss and accuracy can be calculated below.
        
        if y is None:
            return self.logsoftmax(final, dim=1)
        else:
            y_pred_val, y_pred = torch.max(final, 1)
            accuracy = (y == y_pred.squeeze()).float().mean()
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(final, y)
            del y_pred_val, y_pred, criterion, final, y
            return loss, accuracy
    
    def eval_option(self, opt, art, SQ, SPP):
        '''This helper method matches each option to its
        corresponding passage, and then uses the previous
        question-passage matching to compute option
        probability through the final pooling unit.'''
        
        # Generate option hidden state
        
        opt = torch.cat((self.cls, opt, self.sep), dim=1)
        ids, mask = self.get_comps(opt.shape[1])
        opt, _ = self.bert(opt, ids, mask, output_all_encoded_layers=False)
        del ids, mask
        
        # Match option hidden state to passage hidden state
        
        SA, SP = self.match_opt.forward(opt, art)
        del opt, art
        
        # Pool matchings to output probabilities
        
        prob = self.pool(SP, SA, SPP, SQ)
        del SP, SA, SPP, SQ
        return prob

    def get_comps(self, size):
        '''This helper method generates BERT segments and masks. 
        Because only one sequence is used, segments are all zero.
        For masking, all ones were used. These hypterparameters 
        were not specified in the SOTA paper, and represent an 
        opportuntiy for improvement with the model.'''
        
        ids = torch.zeros(self.bs, size, dtype=torch.long).to(device)
        mask = torch.ones(self.bs, size, dtype=torch.long).to(device)
        return ids, mask
    
    def gen_token_vector(self, tkn):
        '''This helper method generates CLS and SEP
        vectors of the batch size for concatenation with 
        passage, question, and options.'''
        
        v = torch.tensor([tkn]*self.bs)
        v = v.unsqueeze(1).to(device)
        return v
    
# ==========================================================
     
class Matching_Attention(nn.Module):
    '''This unit implements the matching attention mechanism used
    by the DCMN to match an option or the question to the passage.
    
    The notation used below is the exact notation used by the paper
    at https://arxiv.org/pdf/1901.09381.pdf.'''
    
    def __init__(self, size=768):
        super(Matching_Attention, self).__init__()
        self.G = nn.Linear(size, size, bias=True)
        # Paper calls for two sets of weights:
        # One for matching candidate answer (self.fca)
        # One for matching passage (self.fcp)
        self.fca = nn.Sequential(
            nn.Linear(2*size, size, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fcp = nn.Sequential(
            nn.Linear(2*size, size, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, A, P):
        bs, A_len, l = A.size()
        P_len = P.shape[1]

        # Self-attention and weight calculation
        A = A.view(bs * A_len, l)
        A = self.G(A)
        A = A.view(bs, A_len, l)
        AT = A.transpose(1, 2).contiguous()
        W = torch.bmm(P, AT)
        assert(W.shape==(bs, P_len, A_len))
        W = W.view(bs * P_len, A_len)
        W = self.softmax(W)
        W = W.view(bs, P_len, A_len)
        assert(W.shape==(bs, P_len, A_len))
                
        # Calculate matchings using attention weights W.
        MP = torch.bmm(W, A)
        W = W.transpose(1, 2).contiguous()
        MA = torch.bmm(W, P)
        
        assert(MP.shape == (bs, P_len, l))
        assert(MA.shape == (bs, A_len, l))
        
        # Element wise subtraction and dot, constructing matching.
        SA = torch.cat((MA - A, MA * A), dim=2)
        SP = torch.cat((MP - P, MP * P), dim=2)
        SA = self.fca(SA)
        SP = self.fcp(SP)
        assert(SA.shape==(bs, A_len, l))
        assert(SP.shape==(bs, P_len, l))
        del W, MA, MP, A, P
        return SA, SP
    
# ==========================================================
    
class Final_Pooling(nn.Module):
    '''As specified in the paper, the final pooling layer
    takes in the final hidden states from the two matching
    units, does sequence-wise max pool on them, and passes
    the concantenated vectors into a linear layer to construct
    the relative probability that the corresponding candidate 
    answer is correct.'''
    
    def __init__(self, size=768):
        super(Final_Pooling, self).__init__()
        self.size = size
        self.out = nn.Linear(4*size, 1, bias=False)

    def forward(self, SP, SA, SPP, SQ):
        bs = SP.shape[0]
        
        # Sequence-wise max pool.
        CP, _ = SP.max(1, keepdim=False)
        CA, _ = SA.max(1, keepdim=False)
        CPP, _ = SPP.max(1, keepdim=False)
        CQ, _ = SQ.max(1, keepdim=False)
        
        # Concatenate and linear layer.
        final = torch.cat((CP, CA, CPP, CQ), dim=1)
        assert(final.shape==(bs, 4*self.size))
        final = self.out(final)
        del SP, SA, SPP, SQ, CP, CA, CPP, CQ, bs
        
        return final

# ==========================================================
# RACE DATA LOADER
# ==========================================================

class RACE_Loader():
    '''This class implements a data iterator for RACE.
    
    Inputs:
        f: filename of data
        bs: batch size
        art_len: number of tokens in article 
        q_len: number of tokens in q_len
        opt_len: number of tokens in options/candidate answers
     
    Data format:    
        Data must be pre-processed as BERT token embeddings in a JSON file. 
        See the attached data files for format.
        
        Fields should be:
            'article_tokens': token embeddings of article
            'q_tokens': token embeddings of question
            'a_tokens': token embeddings of option a
            'b_tokens': token embeddings of option b
            'c_tokens': token embeddings of option c
            'd_tokens': token embeddings of option d
            'y': integer representation of label, in [0, 1, 2, 3].'''
    
    def __init__(self, f, bs, art_len, q_len, opt_len):
        
        # Save parameters
        self.bs = bs
        self.file = f 
        self.art_len = art_len
        self.q_len = q_len
        self.opt_len = opt_len
        
        # Single-sequence base BERT is limited to 510 + CLS + SEP tokens.
        assert(self.art_len <= 510)
        assert(self.q_len <= 510)
        assert(self.opt_len <= 510)
        
        # Load data
        self.loader = self.load_data(f)
        
    def has_next(self):
        return (len(self.loader) >= self.bs)
    
    def next(self):
        '''Constructs next batch, Java iterator style!'''
        
        assert(self.has_next())
        count = self.bs
        article = []
        q = []
        a = []
        b = []
        c = []
        d = []
        y = []
        while (count > 0):
            # Get observation
            obs = self.loader.pop()
            
            # Build and pad batch
            article.append(self.pad(obs['article_tokens'], self.art_len))
            q.append(self.pad(obs['q_tokens'], self.q_len))
            a.append(self.pad(obs['a_tokens'], self.opt_len))
            b.append(self.pad(obs['b_tokens'], self.opt_len))
            c.append(self.pad(obs['c_tokens'], self.opt_len))
            d.append(self.pad(obs['d_tokens'], self.opt_len))
            y.append(int(obs['y']))
            count -= 1
         
        # Tensorify and put to GPU
        article = torch.tensor(article).to(device)
        q = torch.tensor(q).to(device)
        a = torch.tensor(a).to(device)
        b = torch.tensor(b).to(device)
        c = torch.tensor(c).to(device)
        d = torch.tensor(d).to(device)
        y = torch.tensor(y).to(device)  
        
        return article, q, a, b, c, d, y
    
    def pad(self, tokens, length):
        '''This helper method pads BERT tokens with zeros to
        make a tensor batch.'''
        
        if len(tokens) > length:
            tokens = tokens[0:length]
            
        elif len(tokens) < length:
            tokens = tokens + [0] * (length - len(tokens))
            
        return tokens
        
    def load_data(self, file):
        '''Loads data into list, from JSON.'''
        
        with open(file) as f:  
            data = json.load(f)
        
        # Remove header
        data_loader = data[1:]
        
        # Randomize order of data
        shuffle(data_loader)
        
        return data_loader
    
    def num_batches(self):
        return (len(self.loader) // self.bs)
    
    def reset_loader(self):
        '''Reloads and shuffles data, for next epoch.'''
        
        self.loader = self.load_data(self.file)
        
# ==========================================================
# TRAINING LOOP
# ==========================================================

def train_bert(mdl, data_loader, optimizer, num_epochs):
    '''Trains specified mdl, using data in data_loader and the
    specified optimizer for num_epochs.
    
    Saves model state, as well as loss and acurracy history, 
    at end of each epoch.'''
    
    losses = []
    accs = []
    criterion = nn.CrossEntropyLoss()
    mdl.train()
    
    for epoch in range(num_epochs):
        curr = 0
        total = data_loader.num_batches()
        while data_loader.has_next():
            art, q, a, b, c, d, y = data_loader.next()

            # Forward pass
            loss, acc = mdl.forward(art, q, a, b, c, d, y)
            del art, q, a, b, c, d, y

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss stats and update user
            losses.append(loss.item())
            accs.append(acc.item())

            # Print an update
            if (curr+1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}' 
                       .format(epoch+1, num_epochs, curr+1, total, loss.item(), acc.item()))
            curr += 1
         
        # At end of epoch, save model and results
        torch.save(mdl, "dcmn_" + str(epoch+1) + ".pt")
        with open("dcmn_results_" + str(epoch+1) + ".txt", "wb") as outf: 
            pickle.dump((losses, accs), outf)
            
        # Reset data loader and shuffle data.
        data_loader.reset_loader()
        
    return mdl, losses, accs

# ==========================================================
# TESTING LOOP
# ==========================================================

def test_bert(mdl, dev_loader, test_loader):
    mdl.eval()
    dev_acc = sub_test_bert(mdl, dev_loader)
    test_acc = sub_test_bert(mdl, test_loader)
    
    return dev_acc, test_acc
    
def sub_test_bert(mdl, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        while loader.has_next():
            art, q, a, b, c, d, y = loader.next() 

            yhat = mdl.forward(art, q, a, b, c, d)
            del art, q, a, b, c, d
            _, y_pred = torch.max(yhat, 1)
            
            correct += (y == y_pred.squeeze()).float().sum()
            total += y.shape[0]
            del y, yhat, y_pred
    return correct / total

# ==========================================================
# MAIN
# ==========================================================

def main():
    
    print("Initializing DCMN...")
    
    # Push to GPU. 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Parameters. Lack of memory, compute power, and researcher documentation 
    # led to suboptimal performance. See technical report for details.
    
    batch_size = 12
    art_len = 400
    q_len = 30
    opt_len = 15
    num_epochs = 3

    train_file = 'train_data_json.txt'
    dev_file = 'dev_data_json.txt'
    test_file = 'test_data_json.txt'

    # Instantiate model, optimizer, and data loader

    mdl = DCMN(batch_size).to(device)

    train_loader = RACE_Loader(train_file, 
        batch_size, art_len, q_len, opt_len)

    optimizer = BertAdam(mdl.parameters(), lr=2e-5)

    # Train

    print("Beginning training...")
    print(str('-'*60))

    mdl, losses, accs = train_bert(mdl, train_loader, optimizer, num_epochs)
    del optimizer, train_loader

    print("Training complete!")
    print(str('-'*60))

    # Test

    print("Beginning testing...")
    print(str('-'*60))

    dev_loader = RACE_Loader(dev_file, 
        batch_size, art_len, q_len, opt_len)

    test_loader = RACE_Loader(test_file, 
        batch_size, art_len, q_len, opt_len)

    dev_acc, test_acc = test_bert(mdl, dev_loader, test_loader)
    
    print('Accuracy on the dev set: {} %'.format(100 * dev_acc))
    print('Accuracy on the test set: {} %'.format(100 * test_acc))

    print("DCMN complete!")

if __name__=='__main__':
    main()
