
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
from the_vectorizer import Vectorizer
from tqdm import tqdm

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

class K1_model:
    def __init__(self, hidden=100, resume_from=None):
        self.use_cuda   = torch.cuda.is_available()
        self.device     = torch.device("cuda") if (self.use_cuda) else torch.device("cpu")
        print('Using device: {}'.format(self.device))
        self.catid_to_catname   = {
          0: '1. No poverty',
          1: '2. Zero hunger',
          2: '3. Good health',
          3: '4. Education',
          4: '5. Gender equality',
          5: '6. Clean water',
          6: '7. Clean energy',
          7: '8. Economic growth',
          8: '9. Industry and infrastructure',
          9: '10. No inequality',
          10: '11. Sustainability',
          11: '12. Responsible consumption',
          12: '13. Climate action',
          13: '14. Life underwater',
          14: '15. Life on land',
          15: '16. Peace & justice'
        }
        self.catname_to_catid   = dict((v, k) for k, v in self.catid_to_catname.items())
        self.my_model           = Ontop_Modeler(hidden=hidden, catname_to_catid=self.catname_to_catid).to(self.device)
        checkpoint              = torch.load(resume_from, map_location=lambda storage, loc: storage)
        self.my_model.load_state_dict(checkpoint['state_dict'])
        print("=> loading checkpoint '{}'".format(resume_from))
        _ = self.my_model.eval()
    def emit_for_abstracts(self, abstracts, abstract_vectors):
        with torch.no_grad():
            y           = self.my_model.emit(abstract_vectors)
            ret         = []
            for i in range(len(abstracts)):
                results_of_abs = dict(
                    (
                        k,
                        round(y.tolist()[i][v], 2)
                    )
                    for k, v in self.catname_to_catid.items()
                )
                ret.append((abstracts[i], results_of_abs))
        return ret

if __name__ == '__main__':
    abstracts = 10*[
        '''
        HIV disproportionately impacts youth, particularly young men who have sex with men (YMSM), a population that includes subgroups of young men who have sex with men only (YMSMO) and young men who have sex with men and women (YMSMW). In 2015, among male youth, 92% of new HIV diagnoses were among YMSM. The reasons why YMSM are disproportionately at risk for HIV acquisition, however, remain incompletely explored. We performed event-level analyses to compare how the frequency of condom use, drug and/or alcohol use at last sex differed among YMSMO and YMSWO (young men who have sex with women only) over a ten-year period from 2005–2015 within the Youth Risk Behavior Survey (YRBS). YMSMO were less likely to use condoms at last sex compared to YMSWO. However, no substance use differences at last sexual encounter were detected. From 2005–2015, reported condom use at last sex significantly declined for both YMSMO and YMSWO, though the decline for YMSMO was more notable. While there were no significant differences in alcohol and substance use at last sex over the same ten-year period for YMSMO, YMSWO experienced a slight but significant decrease in reported alcohol and substance use. These event-level analyses provide evidence that YMSMO, similar to adult MSMO, may engage in riskier sexual behaviors compared to YMSWO, findings which may partially explain the increased burden of HIV in this population. Future work should investigate how different patterns of event-level HIV risk behaviors vary over time among YMSMO, YMSWO, and YMSMW, and are tied to HIV incidence among these groups.
        '''.strip(),
        "french populism and discourses on secularism nilsson pereriknew york bloomsbury  pp  isbn  since the first controversies over hijabs back in  laicitethe particular french version of secularismhas increasingly been used as a legitimizing frame for an exclusionary agenda aimed at re".strip(),
        '''
        comprendre des histoires en cours prparatoire lexemple du rappel de rcit accompagn cette tude propose une analyse qualitative de trois sances de rappel de rcit choisies afin de mieux cerner les gestes professionnels denseignants de cours prparatoire dans le domaine de la comprhension de textes les interactions orales lors de ces rappels de rcit prsentent des caractristiques communes le questionnement de lenseignant facilite la caractrisation des personnages vise  expliciter leurs penses et leurs actions les reformulations aident  coconstruire le rcit lclaircissement du lexique guide les lves grce  un retour systmatique  lnoncsource ces modalits reprsenteraient des gestes didactiques fondamentaux tayant la comprhension notamment pour les lves les plus en difficult this study offers a qualitative analysis of three selected teaching practices in an attempt to identify the professional actions of teachers of reading comprehension the collective moments devoted to retelling seem to present a certain number of common characteristics the questioning techniques of the teacher help with the description of the characters aims to explain their thoughts and their actions the reformulation of the original text and the pupils suggestions help to coconstruct the story the clarification of single words in the text guide the pupils through a systematic return to the source text all these methods seem to constitute fundamental educational actions which underpin comprehension especially for those pupils who find it most difficult
        '''.strip(),
    ]
    ######################################################################################################
    distilbert_vectorizer   = Vectorizer(model_name="distilbert-base-uncased")
    bert_vectorizer         = Vectorizer(model_name='bert-base-uncased')
    ######################################################################################################
    k1_1 = K1_model(hidden=100, resume_from='./models/distilbert-base-uncased_100_5e-05_29_84_85.pth.tar')
    k1_2 = K1_model(hidden=50,  resume_from='./models/distilbert-base-uncased_50_5e-05_23_83_84.pth.tar')
    k1_3 = K1_model(hidden=100, resume_from='./models/bert-base-uncased_100_5e-05_16_84_84.pth.tar')
    ######################################################################################################
    for i in tqdm(range(10)):
        abstract_vectors_distilbert = distilbert_vectorizer.emit_for_abstracts(abstracts)
        abstract_vectors_bert       = bert_vectorizer.emit_for_abstracts(abstracts)
        ######################################################################################################
        res_1 = k1_1.emit_for_abstracts(abstracts, abstract_vectors_distilbert)
        res_2 = k1_2.emit_for_abstracts(abstracts, abstract_vectors_distilbert)
        res_3 = k1_3.emit_for_abstracts(abstracts, abstract_vectors_bert)
        ######################################################################################################
        print(40*'=')
        for abs, sdg_cats in res_1:
            print(abs)
            pprint(sdg_cats)
            print(40*'-')
        ######################################################################################################
        print(40*'=')
        for abs, sdg_cats in res_2:
            print(abs)
            pprint(sdg_cats)
            print(40*'-')
        ######################################################################################################
        print(40*'=')
        for abs, sdg_cats in res_3:
            print(abs)
            pprint(sdg_cats)
            print(40*'-')
        ######################################################################################################

