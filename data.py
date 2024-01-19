import pandas as pd
import random
import numpy as np
import torch
import transformers
from transformers import BertTokenizer
import json
class data():
    clslabel=[]
    feature=[]
    model_input=[]
    max_limit=28996
    dfload=[]
    jsonload={}
    def loadcsv(self):
        for i in ["Gen_review_factuality_1.csv","Gen_review_factuality_2.csv",
                "Gen_review_factuality_3.csv","Gen_review_factuality_4.csv",
                "Gen_review_factuality_5.csv"]:
            if len(self.dfload)==0:
                self.dfload = pd.read_csv(i)
            else:
                self.dfload = pd.concat([self.dfload,pd.read_csv(i)],axis=0)
    def find(self,l,s):
        if s=='':
            return True
        for i in range(len(s)):
            if ((s[0:i+1] in l) or ('##'+s[0:i+1] in l))\
            and self.find(l,s[i+1:]):
                return True
        return False
    def process(self,l,s):
        templ=s.lower().split()
        outl=[]
        for i in templ:
            #print(i)
            if i=='@response.questionnaire_by_answer(answer)‚äù,':
                continue#此处有bug，但未排查出
            if (i in l) or (self.find(l,i)):
                outl.append(i)
        return outl
    def truncate(self,l1,l2):
        if (len(l1)+len(l2))<=509:
            return l1
        else:
            return l1[0:509-len(l2)]
    def gen_json(self):
        index=0
        temp=''
        for i in range(len(self.dfload)):
            if self.dfload.iloc[i]['feedback'] == np.nan:
                continue
            if self.dfload.iloc[i]['label'] == np.nan:
                continue
            if self.dfload.iloc[i]['pid'] == temp:
                self.jsonload[index][1].append(self.dfload.iloc[i]['feedback'])
                self.jsonload[index][2].append(self.dfload.iloc[i]['label'])
            else:
                index+=1
                temp=self.dfload.iloc[i]['pid']
                self.jsonload[index]=[self.dfload.iloc[i]['doc'],[self.dfload.iloc[i]['feedback']],[self.dfload.iloc[i]['label']]]
    def save_json(self,name='data.json'):
        with open(name,mode='w+',encoding='utf-8') as f:
            json.dump(self.jsonload,f,ensure_ascii=False)
    def load_json(self,name='data_cf.json'):
        with open(name,encoding='utf-8') as f:
            self.jsonload=json.load(f)
    def generate(self):
        #with open(r'bert-base-uncased/vocab.txt',encoding='gb18030',errors='ignore') as f:
        #    l=f.read().splitlines()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.clslabel=torch.tensor([[x] for x in list(self.dfload['label'])])
        transformers.logging.set_verbosity_error()
        memory={}
        for i in range(len(self.dfload)):
            #if self.dfload.iloc[i]['doc'] in memory.keys():
            #    Q=memory[self.dfload.iloc[i]['doc']]
            #else:
            #    Q=self.process(l,self.dfload.iloc[i]['doc'])#不在词表内的删除
            #    memory[self.dfload.iloc[i]['doc']]=Q
            #A=self.process(l,self.dfload.iloc[i]['feedback'])
            #Q=self.truncate(Q,A)#BERT输入长度为512，长度大了要删除
            #fl=[]
            if True:
                print(i,end=' ')
            try:
                temp=(tokenizer(self.dfload.iloc[i]['doc'],
                                self.dfload.iloc[i]['feedback'],
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors="pt") )
                self.model_input.append((temp,torch.tensor([int(self.dfload.iloc[i]['label'])])))
            except:
                print('check data')
        if False:#如果显示有数据超出embedding最大值，这里要处理
            for i in self.feature:
                for j in range(len(i['input_ids'][0])):
                    if i['input_ids'][0][j]>=self.max_limit:
                        i['input_ids'][0][j]=self.max_limit-1;
        #self.model_input=list(zip(self.feature,self.clslabel))
        random.shuffle(self.model_input)
    def json_replace_this_example(self):
        for i in self.jsonload:
            for j in range(len(self.jsonload[i][1])):
                if "For example" in self.jsonload[i][1][j] and (j>=1):
                    if (self.jsonload[i][2][j]>0.5 and self.jsonload[i][2][j-1]>0.5) or\
                            (self.jsonload[i][2][j]<0.5 and self.jsonload[i][2][j-1]>0.5) or\
                            (self.jsonload[i][2][j]<0.5 and self.jsonload[i][2][j-1]<0.5):
                        print(i,"About that "+self.jsonload[i][1][j-1][:-1]+" : "+self.jsonload[i][1][j])
                        temp=input()
                        if temp=='1':
                            self.jsonload[i][1][j]="About that "+self.jsonload[i][1][j-1][:-1]+" : "+self.jsonload[i][1][j]
                        if len(temp)>3:
                            self.jsonload[i][1][j]=temp
        for i in self.jsonload:
            for j in range(len(self.jsonload[i][1])):
                if "This is" in self.jsonload[i][1][j] and (j>=2):
                    if (self.jsonload[i][2][j]>0.5 and self.jsonload[i][2][j-1]>0.5) or\
                            (self.jsonload[i][2][j]<0.5 and self.jsonload[i][2][j-1]>0.5) or\
                            (self.jsonload[i][2][j]<0.5 and self.jsonload[i][2][j-1]<0.5):
                                out=""
                                if 'but' in self.jsonload[i][1][j]:
                                    print(i,self.jsonload[i][1][j-1][:-1] + " : "+ "The writeup" +self.jsonload[i][1][j][4:])
                                    out=self.jsonload[i][1][j-1][:-1] + " : "+ "The writeup" +self.jsonload[i][1][j][4:]
                                else:
                                    print(i,"that "+ self.jsonload[i][1][j-1][:-1] + self.jsonload[i][1][j][4:])
                                    out="that "+ self.jsonload[i][1][j-1][:-1] + self.jsonload[i][1][j][4:]
                                temp=input()
                                if temp=='1':
                                    self.jsonload[i][1][j]=out
                                if len(temp)>3:
                                    self.jsonload[i][1][j]=temp
        with open('data_cf.json',mode='w+',encoding='utf-8') as f:
            json.dump(self.jsonload,f,ensure_ascii=False)
    def sli_cf_generate(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        transformers.logging.set_verbosity_error()
        for i in self.jsonload:
          try:
            temp=tokenizer(self.jsonload[i][0],
                    padding='max_length',
                    max_length = 3600,
                    truncation=True,
                    return_tensors="pt")
            for k in range(len(self.jsonload[i][1])):
                    out=[]
                    slidenum=int((sum(temp['attention_mask'][0])-100)/330)+1
                    for j in range(0,slidenum):
                        s=tokenizer.decode(temp['input_ids'][0][j*330:j*330+430])
                        piece=tokenizer(s,
                            self.jsonload[i][1][k],
                            padding='max_length',
                            max_length = 512,
                            truncation=True,
                            return_tensors="pt")
                        if len(out)==0:
                            out=piece
                        else:
                            out['input_ids'] = torch.concat([out['input_ids'],piece['input_ids']],dim=0)
                            out['attention_mask'] = torch.concat([out['attention_mask'],piece['attention_mask']],dim=0)
                            out['token_type_ids'] = torch.concat([out['token_type_ids'],piece['token_type_ids']],dim=0)
                    self.model_input.append((out,torch.tensor([int(self.jsonload[i][2][k])])))
          except:
            print(i)
        random.shuffle(self.model_input)

    def sliwin_generate(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        transformers.logging.set_verbosity_error()
        for i in range(len(self.dfload)):
            try:
                temp=tokenizer(self.dfload.iloc[i]['doc'],
                        padding='max_length',
                        max_length = 1300,
                        truncation=True,
                        return_tensors="pt")
                out=[]
                for j in range(0,4):
                    s=tokenizer.decode(temp['input_ids'][0][j*300:j*300+400])
                    #print(s)
                    piece=tokenizer(s,self.dfload.iloc[i]['feedback'],
                            padding='max_length',
                            max_length = 512,
                            truncation=True,
                            return_tensors="pt")
                    if len(out)==0:
                        out=piece
                    else:
                        out['input_ids'] = torch.concat([out['input_ids'],piece['input_ids']],dim=0)
                        out['attention_mask'] = torch.concat([out['attention_mask'],piece['attention_mask']],dim=0)
                        out['token_type_ids'] = torch.concat([out['token_type_ids'],piece['token_type_ids']],dim=0)
                self.model_input.append((out,torch.tensor([int(self.dfload.iloc[i]['label'])])))
            except:
                print(i)
        random.shuffle(self.model_input)
    def save(self):
        torch.save(self.model_input,'processed_data')
    def load(self):
        self.model_input=torch.load('processed_data')
    def save_win(self):
        torch.save(self.model_input,'processed_data_win')
    def load_win(self):
        self.model_input=torch.load('processed_data_win')

        
        
