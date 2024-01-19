import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import AdamW
from tqdm import tqdm
import argparse
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768,hidden_size=768,bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.linear  = nn.Linear(768*2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,data):
            if type(data['input_ids'])!=type([]):
                #slide_num=data['input_ids'].shape[0]
                data['input_ids']=[data['input_ids']]
                data['attention_mask']=[data['attention_mask']]
                data['token_type_ids']=[data['token_type_ids']]
            batch_list=[self.bert(input_ids= data['input_ids'][i],
                                        attention_mask=data['attention_mask'][i],
                                        token_type_ids=data['token_type_ids'][i],
                                        return_dict=False)[1] for i in range(len(data['input_ids']))]
            #print(batch_list,len(batch_list),len(batch_list[0]),len(batch_list[0][0]))
            lstmout=[self.lstm(i)[1][1] for i in batch_list]
            c = torch.stack([torch.concat([j[0],j[1]]) for j in lstmout])
            #print(c,len(c),len(c[0]))
            reluout=self.relu(c)
            drop = self.dropout(reluout)
            linearout = self.linear(drop)
            final = self.sigmoid(linearout)
            return final
class NNmanager():
    model = None
    use_cuda = None
    device = None
    def __init__(self):
        self.model=BertClassifier()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.batch_size=2
        self.best_acc=0
        self.best_F1=0
    def load(self):
        self.model.load_state_dict(torch.load('BertClassifier2.pth'))
    def dataloader(self,data):
        out=[]
        for i in range(0,len(data),self.batch_size):
            temp={}
            temp['input_ids']=[]
            temp['attention_mask']=[]
            temp['token_type_ids']=[]
            label=[]
            for j in range(0,self.batch_size):
                if i+j>=len(data):
                    continue
                temp['input_ids'].append(data[i+j][0]['input_ids'])
                temp['attention_mask'].append(data[i+j][0]['attention_mask'])
                temp['token_type_ids'].append(data[i+j][0]['token_type_ids'])
                label.append(data[i+j][1])
            label=torch.stack(label)
            out.append([temp,label])
        return out
    def train(self,epochs,data,testdata=[]):
        criterion = nn.BCELoss()
        optimizer = AdamW(self.model.parameters(), lr=1e-6, weight_decay=1e-3)
        train_dataloader=self.dataloader(data)
        for epoch_num in range(epochs):
            total_acc_train = 0
            total_loss_train = 0
            self.model=self.model.train()
            import time
            t1=time.time()
            print('epoch len: ' ,len(train_dataloader),flush=True)
            for train_input, train_label in train_dataloader:
                #print(train_input, train_label)
                train_label = train_label.float()
                train_input = train_input
                #mask = train_input['attention_mask'].to(self.device)
                #type_id = train_input['token_type_ids'].squeeze(1).to(self.device)
                #input_id = train_input['input_ids'].squeeze(1).to(self.device)
                output = self.model(train_input) 
                #print(output,train_label)
                #formatout = torch.tensor([x for x in output]).to(self.device)
                #print(output)
                #print(train_label)
                batch_loss = criterion(output, train_label)
                #print(batch_loss)
                total_loss_train += batch_loss.item()
                for i in range(len(train_label)):
                    try:
                        if output[i][0]<0.5 and train_label[i][0]<0.5:
                            total_acc_train +=1
                        if output[i][0]>=0.5 and train_label[i][0]>=0.5:
                            total_acc_train +=1
                    except Exception as err:
                        print(err, output,train_label)
                optimizer.zero_grad()
                #batch_loss.requires_grad = True
                batch_loss.backward()
                optimizer.step()
                #scheduler.step()
                #print([x.grad for x in optimizer.param_groups[0]['params']])
            print('Epochs: {} ; Train Loss: {} ; Train Accuracy: {}'.
                  format(epoch_num + 1, round(total_loss_train / len(data) ,3),
                   round(total_acc_train / len(data) ,3)),flush=True)
            acc, F1=self.predlist(testdata)
            if (F1>=self.best_F1):
                torch.save(self.model.state_dict(), 'BertClassifier2.pth')
                self.best_acc=acc
                self.best_F1=F1
                print('model refreshed',flush=True)
            print('best acc: ',self.best_acc, 'best F1:' , self.best_F1, flush=True)
            #if (total_acc_train / len(data))>0.950:
            #    break
            print(time.time()-t1)
    def predict(self,x):
        self.model=self.model.to(self.device)
        self.model=self.eval()
        with torch.no_grad():
            #print(x)
            #mask = x['attention_mask'].to(self.device)
            #input_id = x['input_ids'].squeeze(1).to(self.device)
            #type_id = x['token_type_ids'].squeeze(1).to(self.device)
            out = self.model(x.to(self.device))
            return out
    def predlist(self,xlist):
        self.model=self.model.eval()
        test=[[0,0],[0,0]]
        wrong=[[],[]]
        with torch.no_grad():
            for x in xlist:
                out = self.model(x[0])
                #print(out,x[1])
                if out[0][0]<0.5 and x[1][0]<0.5:
                    test[1][1]+=1
                if out[0][0]>=0.5 and x[1][0]>=0.5:
                    test[0][0]+=1
                if out[0][0]<0.5 and x[1][0]>=0.5:
                    test[0][1]+=1
                    wrong[0].append([x,out])
                if out[0][0]>=0.5 and x[1][0]<0.5:
                    test[1][0]+=1
                    wrong[1].append([x,out])
        print(test,flush=True)
        try:
            P1=test[0][0]/(test[0][0]+test[0][1])
            R1=test[0][0]/(test[0][0]+test[1][0])
            F11=2*P1*R1/(P1+R1)
            P2=test[1][1]/(test[1][1]+test[0][1])
            R2=test[1][1]/(test[1][1]+test[1][0])
            F12=2*P2*R2/(P2+R2)
            acc=(test[0][0]+test[1][1])/len(xlist)
            F1=(F11+F12)/2
            print('acc: ', acc, flush=True)
            print('macro ave F1: ',F1, flush=True)
            torch.save(wrong,"wrongout")
            return acc,F1
        except Exception as err:
            print(err)
        return 0,0
