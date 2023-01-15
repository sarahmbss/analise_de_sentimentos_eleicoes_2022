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

# Treinamento do modelo
nlp = spacy.load('pt_core_news_sm')

# Planilha com dados categorizados
Corpus = pd.read_excel('df_treino.xlsx')

# Remover linhas vazias.
Corpus['content'].dropna(inplace=True)

# Remover duplicados
Corpus.drop_duplicates(inplace=True)

# Remover marcações e risadas
for i in range(len(Corpus['content'])):
    palavras = Corpus['content'][i].split()
    palavras = [x for x in palavras if "@" not in x]
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

# Remover stop-words e aplicar stemming
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['content']):
    Final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords and word.isalpha():
            if word in correcao:
                word = correcao[word]
            word = unidecode(word)
            word_Final = [token.lemma_ for token in nlp(word)]
            Final_words.extend(word_Final)
    Corpus.loc[index,'text_final'] = str(Final_words)
    
# Algumas linhas ficaram com valor Nan, por isso devem ser excluídas
Corpus.dropna(inplace=True)

# Transformar a polaridade em números
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Corpus['sentiment'])

# TF-IDF
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Corpus['text_final'])

# ------------------------------------------------#
# Lula
# ------------------------------------------------#
df_lula = pd.read_csv("df_lula.csv")
df_lula.drop(columns=['Unnamed: 0'], inplace=True)

# Remover linhas vazias.
df_lula['content'].dropna(inplace=True)

# Remover duplicados
df_lula.drop_duplicates(inplace=True)
df_lula.reset_index(inplace=True, drop=True)

# Remover marcações e risadas
for i in range(len(df_lula['content'])):
    palavras = df_lula['content'][i].split()
    palavras = [x for x in palavras if "@" not in x]
    palavras = [x for x in palavras if "kkk" not in x]
    entry = ' '.join(palavras)
    df_lula['content'][i] = entry
    
# Padroniza todo o texto em caixa baixa (letras minúsculas)
df_lula['content'] = [entry.lower() for entry in df_lula['content']]

# Remove números e caracteres especiais
df_lula['content'] = df_lula['content'].apply(lambda x: re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x))

# Remove @RT de retweets
df_lula['content'] = [re.sub(r'^rt[\s]+', '', entry) for entry in df_lula['content']]
    
# Remove hiperlinks
df_lula['content'] = [re.sub(r'https?:\/\/.*[\r\n]*http', '', entry) for entry in df_lula['content']]

# Remove enters
df_lula['content'] = [re.sub(r'\n', '', entry) for entry in df_lula['content']]

# Remove enters
df_lula['content'] = [re.sub(r'https', '', entry) for entry in df_lula['content']]

# Remove enters
df_lula['content'] = [re.sub(r'http', '', entry) for entry in df_lula['content']]

# Remove as palavras chaves buscadas na API
df_lula['content'] = [re.sub(r'lula', '', entry) for entry in df_lula['content']]
df_lula['content'] = [re.sub(r'Lula', '', entry) for entry in df_lula['content']]

# Tokenização: Cada tweet é dividido em um array de palavras
df_lula['content']= [word_tokenize(entry) for entry in df_lula['content']]

# Remover stop-words e aplicar lematização
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(df_lula['content']):
    Final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords and word.isalpha():
            if word in correcao:
                word = correcao[word]
            word = unidecode(word)
            word_Final = [token.lemma_ for token in nlp(word)]
            Final_words.extend(word_Final)
    df_lula.loc[index,'text_final'] = str(Final_words)
    
Test_X_TfidfLula = Tfidf_vect.transform(df_lula['text_final'])

# Classificador Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NBTest = Naive.predict(Test_X_TfidfLula)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
df_lula['previsao_NB'] = Test_YRecover

# Classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions = SVM.predict(Test_X_TfidfLula)
predictions_recovered = Encoder.inverse_transform(predictions)
df_lula['previsao_SVM'] = predictions_recovered

# ------------------------------------------------#
# Bolsonaro
# ------------------------------------------------#
df_bolsonaro = pd.read_csv("df_bolsonaro.csv")
df_bolsonaro.drop(columns=['Unnamed: 0'], inplace=True)

# Remover linhas vazias.
df_bolsonaro['content'].dropna(inplace=True)

# Remover duplicados
df_bolsonaro.drop_duplicates(inplace=True)
df_bolsonaro.reset_index(inplace=True, drop=True)

# Remover marcações e risadas
for i in range(len(df_bolsonaro['content'])):
    palavras = df_bolsonaro['content'][i].split()
    palavras = [x for x in palavras if "@" not in x]
    palavras = [x for x in palavras if "kkk" not in x]
    entry = ' '.join(palavras)
    df_bolsonaro['content'][i] = entry
    
# Padroniza todo o texto em caixa baixa (letras minúsculas)
df_bolsonaro['content'] = [entry.lower() for entry in df_bolsonaro['content']]

# Remove números e caracteres especiais
df_bolsonaro['content'] = df_bolsonaro['content'].apply(lambda x: re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x))

# Remove @RT de retweets
df_bolsonaro['content'] = [re.sub(r'^rt[\s]+', '', entry) for entry in df_bolsonaro['content']]
    
# Remove hiperlinks
df_bolsonaro['content'] = [re.sub(r'https?:\/\/.*[\r\n]*http', '', entry) for entry in df_bolsonaro['content']]

# Remove enters
df_bolsonaro['content'] = [re.sub(r'\n', '', entry) for entry in df_bolsonaro['content']]

# Remove enters
df_bolsonaro['content'] = [re.sub(r'https', '', entry) for entry in df_bolsonaro['content']]

# Remove enters
df_bolsonaro['content'] = [re.sub(r'http', '', entry) for entry in df_bolsonaro['content']]

# Remove as palavras chaves buscadas na API
df_bolsonaro['content'] = [re.sub(r'bolsonaro', '', entry) for entry in df_bolsonaro['content']]
df_bolsonaro['content'] = [re.sub(r'jair bolsonaro', '', entry) for entry in df_bolsonaro['content']]

# Tokenização: Cada tweet é dividido em um array de palavras
df_bolsonaro['content']= [word_tokenize(entry) for entry in df_bolsonaro['content']]

# Remover stop-words e aplicar lematização
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(df_bolsonaro['content']):
    Final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords and word.isalpha():
            if word in correcao:
                word = correcao[word]
            word = unidecode(word)
            word_Final = [token.lemma_ for token in nlp(word)]
            Final_words.extend(word_Final)
    df_bolsonaro.loc[index,'text_final'] = str(Final_words)
    
Test_X_TfidfBolsonaro = Tfidf_vect.transform(df_bolsonaro['text_final'])

# Classificador Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NBTest = Naive.predict(Test_X_TfidfBolsonaro)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
df_bolsonaro['previsao_NB'] = Test_YRecover

# Classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions = SVM.predict(Test_X_TfidfBolsonaro)
predictions_recovered = Encoder.inverse_transform(predictions)
df_bolsonaro['previsao_SVM'] = predictions_recovered

# ------------------------------------------------#
# Ciro
# ------------------------------------------------#
df_ciro = pd.read_csv("df_ciro.csv")
df_ciro.drop(columns=['Unnamed: 0'], inplace=True)

# Remover linhas vazias.
df_ciro['content'].dropna(inplace=True)

# Remover duplicados
df_ciro.drop_duplicates(inplace=True)
df_ciro.reset_index(inplace=True, drop=True)

# Remover marcações e risadas
for i in range(len(df_ciro['content'])):
    palavras = df_ciro['content'][i].split()
    palavras = [x for x in palavras if "@" not in x]
    palavras = [x for x in palavras if "kkk" not in x]
    entry = ' '.join(palavras)
    df_ciro['content'][i] = entry
    
# Padroniza todo o texto em caixa baixa (letras minúsculas)
df_ciro['content'] = [entry.lower() for entry in df_ciro['content']]

# Remove números e caracteres especiais
df_ciro['content'] = df_ciro['content'].apply(lambda x: re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x))

# Remove @RT de retweets
df_ciro['content'] = [re.sub(r'^rt[\s]+', '', entry) for entry in df_ciro['content']]
    
# Remove hiperlinks
df_ciro['content'] = [re.sub(r'https?:\/\/.*[\r\n]*http', '', entry) for entry in df_ciro['content']]

# Remove enters
df_ciro['content'] = [re.sub(r'\n', '', entry) for entry in df_ciro['content']]

# Remove enters
df_ciro['content'] = [re.sub(r'https', '', entry) for entry in df_ciro['content']]

# Remove enters
df_ciro['content'] = [re.sub(r'http', '', entry) for entry in df_ciro['content']]

# Remove as palavras chaves buscadas na API
df_ciro['content'] = [re.sub(r'ciro gomes', '', entry) for entry in df_ciro['content']]
df_ciro['content'] = [re.sub(r'ciro', '', entry) for entry in df_ciro['content']]

# Tokenização: Cada tweet é dividido em um array de palavras
df_ciro['content']= [word_tokenize(entry) for entry in df_ciro['content']]

# Remover stop-words e aplicar lematização
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(df_ciro['content']):
    Final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords and word.isalpha():
            if word in correcao:
                word = correcao[word]
            word = unidecode(word)
            word_Final = [token.lemma_ for token in nlp(word)]
            Final_words.extend(word_Final)
    df_ciro.loc[index,'text_final'] = str(Final_words)
    
Test_X_TfidfCiro = Tfidf_vect.transform(df_ciro['text_final'])

# Classificador Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NBTest = Naive.predict(Test_X_TfidfCiro)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
df_ciro['previsao_NB'] = Test_YRecover

# Classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions = SVM.predict(Test_X_TfidfCiro)
predictions_recovered = Encoder.inverse_transform(predictions)
df_ciro['previsao_SVM'] = predictions_recovered

# ------------------------------------------------#
# Simone Tebet
# ------------------------------------------------#
df_simone = pd.read_csv("df_simone.csv")
df_simone.drop(columns=['Unnamed: 0'], inplace=True)

# Remover linhas vazias.
df_simone['content'].dropna(inplace=True)

# Remover duplicados
df_simone.drop_duplicates(inplace=True)
df_simone.reset_index(inplace=True, drop=True)

# Remover marcações e risadas
for i in range(len(df_simone['content'])):
    palavras = df_simone['content'][i].split()
    palavras = [x for x in palavras if "@" not in x]
    palavras = [x for x in palavras if "kkk" not in x]
    entry = ' '.join(palavras)
    df_simone['content'][i] = entry
    
# Padroniza todo o texto em caixa baixa (letras minúsculas)
df_simone['content'] = [entry.lower() for entry in df_simone['content']]

# Remove números e caracteres especiais
df_simone['content'] = df_simone['content'].apply(lambda x: re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x))

# Remove @RT de retweets
df_simone['content'] = [re.sub(r'^rt[\s]+', '', entry) for entry in df_simone['content']]
    
# Remove hiperlinks
df_simone['content'] = [re.sub(r'https?:\/\/.*[\r\n]*http', '', entry) for entry in df_simone['content']]

# Remove enters
df_simone['content'] = [re.sub(r'\n', '', entry) for entry in df_simone['content']]

# Remove enters
df_simone['content'] = [re.sub(r'https', '', entry) for entry in df_simone['content']]

# Remove enters
df_simone['content'] = [re.sub(r'http', '', entry) for entry in df_simone['content']]

# Remove as palavras chaves buscadas na API
df_simone['content'] = [re.sub(r'simone tebet', '', entry) for entry in df_simone['content']]
df_simone['content'] = [re.sub(r'tebet', '', entry) for entry in df_simone['content']]

# Tokenização: Cada tweet é dividido em um array de palavras
df_simone['content']= [word_tokenize(entry) for entry in df_simone['content']]

# Remover stop-words e aplicar lematização
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(df_simone['content']):
    Final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords and word.isalpha():
            if word in correcao:
                word = correcao[word]
            word = unidecode(word)
            word_Final = [token.lemma_ for token in nlp(word)]
            Final_words.extend(word_Final)
    df_simone.loc[index,'text_final'] = str(Final_words)
    
Test_X_TfidfSimone = Tfidf_vect.transform(df_simone['text_final'])

# Classificador Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NBTest = Naive.predict(Test_X_TfidfSimone)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
df_simone['previsao_NB'] = Test_YRecover

# Classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions = SVM.predict(Test_X_TfidfSimone)
predictions_recovered = Encoder.inverse_transform(predictions)
df_simone['previsao_SVM'] = predictions_recovered

# ------------------------------------------------#
# Felipe d'Avila
# ------------------------------------------------#
df_felipe = pd.read_csv("df_felipe.csv")
df_felipe.drop(columns=['Unnamed: 0'], inplace=True)

# Remover linhas vazias.
df_felipe['content'].dropna(inplace=True)

# Remover duplicados
df_felipe.drop_duplicates(inplace=True)
df_felipe.reset_index(inplace=True, drop=True)

# Remover marcações e risadas
for i in range(len(df_felipe['content'])):
    palavras = df_felipe['content'][i].split()
    palavras = [x for x in palavras if "@" not in x]
    palavras = [x for x in palavras if "kkk" not in x]
    entry = ' '.join(palavras)
    df_felipe['content'][i] = entry
    
# Padroniza todo o texto em caixa baixa (letras minúsculas)
df_felipe['content'] = [entry.lower() for entry in df_felipe['content']]

# Remove números e caracteres especiais
df_felipe['content'] = df_felipe['content'].apply(lambda x: re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x))

# Remove @RT de retweets
df_felipe['content'] = [re.sub(r'^rt[\s]+', '', entry) for entry in df_felipe['content']]
    
# Remove hiperlinks
df_felipe['content'] = [re.sub(r'https?:\/\/.*[\r\n]*http', '', entry) for entry in df_felipe['content']]

# Remove enters
df_felipe['content'] = [re.sub(r'\n', '', entry) for entry in df_felipe['content']]

# Remove enters
df_felipe['content'] = [re.sub(r'https', '', entry) for entry in df_felipe['content']]

# Remove enters
df_felipe['content'] = [re.sub(r'http', '', entry) for entry in df_felipe['content']]

# Remove as palavras chaves buscadas na API
df_felipe['content'] = [re.sub(r"felipe d'avila", '', entry) for entry in df_felipe['content']]
df_felipe['content'] = [re.sub(r'felipe davila', '', entry) for entry in df_felipe['content']]

# Tokenização: Cada tweet é dividido em um array de palavras
df_felipe['content']= [word_tokenize(entry) for entry in df_felipe['content']]

# Remover stop-words e aplicar lematização
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(df_felipe['content']):
    Final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords and word.isalpha():
            if word in correcao:
                word = correcao[word]
            word = unidecode(word)
            word_Final = [token.lemma_ for token in nlp(word)]
            Final_words.extend(word_Final)
    df_felipe.loc[index,'text_final'] = str(Final_words)
    
Test_X_TfidfFelipe = Tfidf_vect.transform(df_felipe['text_final'])

# Classificador Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NBTest = Naive.predict(Test_X_TfidfFelipe)
Test_YRecover = Encoder.inverse_transform(predictions_NBTest)
df_felipe['previsao_NB'] = Test_YRecover

# Classificador SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
predictions = SVM.predict(Test_X_TfidfFelipe)
predictions_recovered = Encoder.inverse_transform(predictions)
df_felipe['previsao_SVM'] = predictions_recovered
