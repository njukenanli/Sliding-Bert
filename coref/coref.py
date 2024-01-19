import json
import numpy as np
import spacy
nlp = spacy.load('en_core_web_lg')
import neuralcoref
neuralcoref.add_to_pipe(nlp)
keys2pop=[]
with open('data.json',encoding='utf-8') as f:
    l=json.load(f)
for i in l:
    print(i)
    origin_s_list = l[i][1]
    if type(origin_s_list[0]) == type(0.1):
        keys2pop.append(i)
        continue
    try:
        doc = nlp("   ".join(origin_s_list))
    except:
        print(type(origin_s_list[0]))
    gen_s_list = doc._.coref_resolved.split('   ')
    for j in range(len(gen_s_list)):
        if gen_s_list[j][0]==" ":
            gen_s_list[j]=gen_s_list[j][1:]
    if len(origin_s_list) == len(gen_s_list):
        l[i][1] = gen_s_list
    else:
        print(origin_s_list)
        print(gen_s_list)
    #print(origin_s_list,gen_s_list)
    #break
for i in keys2pop:
    l.pop(i)
with open('data_cf.json',mode="w+",encoding='utf-8') as f:
    json.dump(l,f)
