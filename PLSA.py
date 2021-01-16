import numba
from numba.typed import List,Dict
from numba import types
import numpy as np
import os
from scipy.sparse import csr_matrix
import time

def GetValcabularyAndCalculateTF():
    Corpus_dict = {}
    Corpus_index = 0
    DocumentTF_list = []
    BackGround = {}
    
    T = 0
    
    for index,DocumentPath in enumerate(DocumentPaths):
        TF = {}
        with open(DocumentPath,'r') as file:
            Terms = file.read().split()
        
        for Term in Terms:
            TF[Term] = 1 if Term not in TF else TF[Term] + 1
            if Term not in Corpus_dict:
                Corpus_dict[Term] = Corpus_index
                Corpus_index += 1
            if Term not in BackGround:
                BackGround[Term] = 1
            else:
                BackGround[Term] += 1
        DocumentTF_list.append(TF)
    
        
        print(f"finish document:{index}",end = '\r')
        T += 1
        if T == 1000:
            break
    return Corpus_dict,DocumentTF_list,BackGround

# Conver TF Dictionary to List
# @numba.jit()
def ConvertTFDict_to_List(Corpus_dict,DocumentTF_list):
    length = len(Corpus_dict)
    newTF = np.zeros([len(DocumentTF_list),length])
    for index,DocumentTF in enumerate(DocumentTF_list):
        tempTF = np.zeros(length)
        for Term in DocumentTF:
            tempTF[Corpus_dict[Term]] = DocumentTF[Term]
        newTF[index] = tempTF
            
    return newTF

def ConverBGTerm_to_ID(Corpus,BackGround):
    tempBG = {}
    for Term in BackGround:
        tempBG[Corpus[Term]] = BackGround[Term]
#     print(f"sum:{sum(BackGround.values())}")
    for Term in tempBG:
        tempBG[Term] /= sum(tempBG.values())
    return tempBG

# @numba.jit()
def initial_P_WiTk(TopicNum,WordNum):
    P_WiTk = np.random.rand(TopicNum,WordNum)
    for k in range(TopicNum):
        P_WiTk[k] /= P_WiTk[k].sum()
    return P_WiTk
            
# @numba.jit()
def initial_P_TkDj(DocumentNum,TopicNum):
    P_TkDj = np.random.rand(DocumentNum,TopicNum)
    for j in range(DocumentNum):
            P_TkDj[j] /=sum( P_TkDj[j])
    return P_TkDj

@numba.jit()
def E_step(P_WiTk,P_TkDj,TopicNum,i,j,k):
    Denominator = 0
    for  k_prime in range(TopicNum):
        Denominator += P_WiTk[k_prime][i] * P_TkDj[j][k_prime]
    if(Denominator ==0):
        print(i,j,k)
    
    P_TkWiDj = (P_WiTk[k][i] * P_TkDj[j][k]) / Denominator
    return P_TkWiDj
    
@numba.jit()    
def M_step(P_WiTk,P_TkDj,TopicNum,WordNum,DocumentNum,DocumentTF_list):
    for k in range(TopicNum):
        for i in range(WordNum):
            value = 0
            for j in range(DocumentNum):
                if DocumentTF_list[j][i] != 0:
                    value += DocumentTF_list[j][i] * E_step(P_WiTk,P_TkDj,TopicNum,i,j,k)
            P_WiTk[k][i] = value
        P_WiTk[k] /= summation(P_WiTk[k])
            
    for j in range(DocumentNum):
        for k in range(TopicNum):
            value = 0
            for i in range(WordNum):
                if DocumentTF_list[j][i] != 0:
                    value += DocumentTF_list[j][i] * E_step(P_WiTk,P_TkDj,TopicNum,i,j,k)
            value /= summation(DocumentTF_list[j])
            P_TkDj[j][k] = value

            
@numba.jit()
def summation(_list):
    all_sum = 0
    for i in range(len(_list)):
        all_sum += _list[i] 
    return all_sum

@numba.jit()
def CalculateLoss(P_WiDj,P_TkDj,DocumentNum,WordNum,TopicNum,DocumentTF_list):
    loss = 0
    for j in range(DocumentNum):
        for i in range(WordNum):
            temp_loss = 0
            if DocumentTF_list[j][i] > 0:
                for k in range(TopicNum):
                    temp_loss += P_WiTk[k][i] * P_TkDj[j][k]
                if temp_loss > 0:
                    temp_loss /= DocumentNum
                    loss += np.log(temp_loss) * DocumentTF_list[j][i]
    return loss

def EM_Algorithm(epochs, P_WiTk, P_TkDj, DocumentNum, WordNum, TopicNum, DocumentTF_list):
    print(f"start at {time.strftime('%X')}")
    for epoch in range(epochs):
        M_step(P_WiTk,P_TkDj,TopicNum,WordNum,DocumentNum,DocumentTF_list)
        print(f"M_step end at{time.strftime('%X')}")
        loss = CalculateLoss(P_WiTk,P_TkDj,DocumentNum,WordNum,TopicNum,DocumentTF_list)
        print(f"loss:{loss}")



QueryFolderPath = 'ntust-ir-2020_hw5_new/queries'
QueryNames = [pathimfor[2] for pathimfor in os.walk(QueryFolderPath)][0]
QueryPaths = [os.path.join(QueryFolderPath,QueryName) for QueryName in QueryNames]
DocumentFolderPath = 'ntust-ir-2020_hw5_new/docs'
DocumentNames = [pathimfor[2] for pathimfor in os.walk(DocumentFolderPath)][0]
DocumentPaths = [os.path.join(DocumentFolderPath, DocumentName) for DocumentName in DocumentNames]

#GetValcabulary
Corpus,DocumentTF_list,BackGround = GetValcabularyAndCalculateTF()
# set parameter
TopicNum = 256
DocumentNum = len(DocumentTF_list)
WordNum = len(Corpus)
epochs = 50

DocumentTF_list = ConvertTFDict_to_List(Corpus, DocumentTF_list)
P_WiTk = initial_P_WiTk(TopicNum,WordNum)
print('Done')
P_TkDj = initial_P_TkDj(DocumentNum, TopicNum)

EM_Algorithm(epochs, P_WiTk, P_TkDj, DocumentNum, WordNum, TopicNum, DocumentTF_list)