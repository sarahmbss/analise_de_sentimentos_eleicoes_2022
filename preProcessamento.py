# Bibliotecas utilizadas
import numpy as np
import pandas as pd
import spacy
import nltk
import re
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Instanciação do NLP
nlp = spacy.load('pt_core_news_sm')

# Planilha com dados categorizados
Corpus = pd.read_excel('df_treino.xlsx')

# Remove linhas vazias.
Corpus['content'].dropna(inplace=True)

# Remove duplicados
Corpus.drop_duplicates(inplace=True)

# Remove marcações de usuários
for i in range(len(Corpus['content'])):
    palavras = Corpus['content'][i].split()
    palavras = [x for x in palavras if "@" not in x]
    # Remove palavras que representam risadas
    palavras = [x for x in palavras if "kkk" not in x]
    entry = ' '.join(palavras)
    Corpus['content'][i] = entry

# Padroniza todo o texto em caixa baixa (letras minúsculas)
Corpus['content'] = [entry.lower() for entry in Corpus['content']]

# Remove números e caracteres especiais
Corpus['content'] = Corpus['content'].apply(lambda x: re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x))

# Remove @RT de retweets
Corpus['content'] = [re.sub(r'^rt[\s]+', '', entry) for entry in Corpus['content']]
    
# Remove hiperlinks
Corpus['content'] = [re.sub(r'https?:\/\/.*[\r\n]*http', '', entry) for entry in Corpus['content']]
Corpus['content'] = [re.sub(r'https', '', entry) for entry in Corpus['content']]
Corpus['content'] = [re.sub(r'http', '', entry) for entry in Corpus['content']]

# Remove enters
Corpus['content'] = [re.sub(r'\n', '', entry) for entry in Corpus['content']]

# Tokenização: Cada tweet é dividido em um array de palavras
Corpus['content']= [word_tokenize(entry) for entry in Corpus['content']]

# Remover stop-words e aplicar lematização
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['content']):
    Final_words = []
    for word, tag in pos_tag(entry):
        # Verifica se a palavra não é stopword
        if word not in stopwords and word.isalpha():
            # Se a palavra estiver no dicionário de correção, ela é corrigida
            if word in correcao:
                word = correcao[word]
            # Remove os acentos
            word = unidecode(word)
            # Faz o NLP
            word_Final = [token.lemma_ for token in nlp(word)]
            Final_words.extend(word_Final)
    Corpus.loc[index,'text_final'] = str(Final_words)
