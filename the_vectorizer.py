
# virtualenv --python=python3.6 transformers
# source /home/dpappas/venvs/transformers/bin/activate
# pip install transformers[sentencepiece]
# pip install torch==1.10.2

# source /media/dpappas/dpappas_data/sdg_classifier_api/sdg_api_venv/bin/activate
# /media/dpappas/dpappas_data/sdg_classifier_api/sdg_api_venv/bin/python

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pprint import pprint

class Ontop_Modeler(nn.Module):
    def __init__(self, hidden, catname_to_catid):
        super(Ontop_Modeler, self).__init__()
        self.linear1    = nn.Linear(768, hidden, bias=True)
        self.linear2    = nn.Linear(hidden, len(catname_to_catid), bias=True)
        self.loss       = nn.BCELoss()
        self.sigmoid    = nn.Sigmoid()
    def emit(self, input_xs):
        y     = self.linear1(input_xs)
        y     = self.sigmoid(y)
        y     = self.linear2(y)
        y     = self.sigmoid(y)
        return y
    def forward(self, input_xs):
        return self.emit(input_xs)

class Vectorizer:
    def __init__(self, model_name = "distilbert-base-uncased"):
        self.use_cuda   = torch.cuda.is_available()
        self.device     = torch.device("cuda") if (self.use_cuda) else torch.device("cpu")
        print('device: {}'.format(self.device))
        self.model_name         = model_name
        print("=> loading vectorizer with '{}'".format(model_name))
        self.bert_model         = AutoModel.from_pretrained(model_name).to(self.device)
        self.bert_tokenizer     = AutoTokenizer.from_pretrained(model_name)
        _ = self.bert_model.eval()
    def embed_abstracts(self, abstracts):
        with torch.no_grad():
            #######################################################################################################
            inputs  = self.bert_tokenizer(abstracts, padding=True, truncation=True, return_tensors="pt")
            bpe_mask = inputs['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=inputs['input_ids'].to(self.device), attention_mask=inputs['attention_mask'].to(self.device),
                    output_attentions=False, output_hidden_states=False, return_dict=True
                )
                vecs    = outputs[0]
                pooled  = outputs['last_hidden_state'][:, 0, :]
            return pooled, vecs, bpe_mask
            #######################################################################################################
    def emit_for_abstracts(self, abstracts):
        with torch.no_grad():
            abs_vecs    = self.embed_abstracts(abstracts)
        return abs_vecs

if __name__ == '__main__':
    abstracts = [
        '''
        HIV disproportionately impacts youth, particularly young men who have sex with men (YMSM), a population that includes subgroups of young men who have sex with men only (YMSMO) and young men who have sex with men and women (YMSMW). In 2015, among male youth, 92% of new HIV diagnoses were among YMSM. The reasons why YMSM are disproportionately at risk for HIV acquisition, however, remain incompletely explored. We performed event-level analyses to compare how the frequency of condom use, drug and/or alcohol use at last sex differed among YMSMO and YMSWO (young men who have sex with women only) over a ten-year period from 2005–2015 within the Youth Risk Behavior Survey (YRBS). YMSMO were less likely to use condoms at last sex compared to YMSWO. However, no substance use differences at last sexual encounter were detected. From 2005–2015, reported condom use at last sex significantly declined for both YMSMO and YMSWO, though the decline for YMSMO was more notable. While there were no significant differences in alcohol and substance use at last sex over the same ten-year period for YMSMO, YMSWO experienced a slight but significant decrease in reported alcohol and substance use. These event-level analyses provide evidence that YMSMO, similar to adult MSMO, may engage in riskier sexual behaviors compared to YMSWO, findings which may partially explain the increased burden of HIV in this population. Future work should investigate how different patterns of event-level HIV risk behaviors vary over time among YMSMO, YMSWO, and YMSMW, and are tied to HIV incidence among these groups.
        '''.strip(),
        "french populism and discourses on secularism nilsson pereriknew york bloomsbury  pp  isbn  since the first controversies over hijabs back in  laicitethe particular french version of secularismhas increasingly been used as a legitimizing frame for an exclusionary agenda aimed at re".strip(),
        '''
        comprendre des histoires en cours prparatoire lexemple du rappel de rcit accompagn cette tude propose une analyse qualitative de trois sances de rappel de rcit choisies afin de mieux cerner les gestes professionnels denseignants de cours prparatoire dans le domaine de la comprhension de textes les interactions orales lors de ces rappels de rcit prsentent des caractristiques communes le questionnement de lenseignant facilite la caractrisation des personnages vise  expliciter leurs penses et leurs actions les reformulations aident  coconstruire le rcit lclaircissement du lexique guide les lves grce  un retour systmatique  lnoncsource ces modalits reprsenteraient des gestes didactiques fondamentaux tayant la comprhension notamment pour les lves les plus en difficult this study offers a qualitative analysis of three selected teaching practices in an attempt to identify the professional actions of teachers of reading comprehension the collective moments devoted to retelling seem to present a certain number of common characteristics the questioning techniques of the teacher help with the description of the characters aims to explain their thoughts and their actions the reformulation of the original text and the pupils suggestions help to coconstruct the story the clarification of single words in the text guide the pupils through a systematic return to the source text all these methods seem to constitute fundamental educational actions which underpin comprehension especially for those pupils who find it most difficult
        '''.strip(),
    ]
    vectorizer  = Vectorizer(model_name="distilbert-base-uncased")
    (pooled_vecs, token_vecs, masks) = vectorizer.emit_for_abstracts(abstracts)
    print(40*'=')
    for i in range(len(abstracts)):
        print(abstracts[i])
        print(pooled_vecs[0][i].size())
        print(token_vecs[1][i].size())
        print(masks[2][i].size())
        print(40*'-')

'''

CUDA_VISIBLE_DEVICES=1 /media/dpappas/dpappas_data/sdg_classifier_api_dec_22/sdg_api_venv/bin/python the_vectorizer.py

'''


