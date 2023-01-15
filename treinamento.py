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
from sklearn.model_selection import cross_validate

#-------------------------------------------#
# Experimento 1
#-------------------------------------------#
nlp = spacy.load('pt_core_news_sm')

# Planilha de treino
Corpus = pd.read_csv(r"DadosAnotados.csv",encoding='latin-1')

# Planilha de teste
df = pd.read_excel('df_treino.xlsx')

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

# Remover stop-words e aplicar lematização
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

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Corpus['Polaridade'])

# TF-IDF
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Corpus['text_final'])

# Pré-processamento da base rotulada
# Remover linhas vazias.
df['content'].dropna(inplace=True)

# Remover duplicados
df.drop_duplicates(inplace=True)

# Remover marcações e risadas
for i in range(len(df['content'])):
    palavras = df['content'][i].split()
    palavras = [x for x in palavras if "@" not in x]
    palavras = [x for x in palavras if "kkk" not in x]
    entry = ' '.join(palavras)
    df['content'][i] = entry

# Padroniza todo o texto em caixa baixa (letras minúsculas)
df['content'] = [entry.lower() for entry in df['content']]

# Remove números e caracteres especiais
df['content'] = df['content'].apply(lambda x: re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x))

# Remove @RT de retweets
df['content'] = [re.sub(r'^rt[\s]+', '', entry) for entry in df['content']]
    
# Remove hiperlinks
df['content'] = [re.sub(r'https?:\/\/.*[\r\n]*http', '', entry) for entry in df['content']]
df['content'] = [re.sub(r'https', '', entry) for entry in df['content']]
df['content'] = [re.sub(r'http', '', entry) for entry in df['content']]

# Remove enters
df['content'] = [re.sub(r'\n', '', entry) for entry in df['content']]

# Tokenização: Cada tweet é dividido em um array de palavras
df['content']= [word_tokenize(entry) for entry in df['content']]

# Remover stop-words e aplicar lematização
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(df['content']):
    Final_words = []
    for word, tag in pos_tag(entry):
        if word not in stopwords and word.isalpha():
            if word in correcao:
                word = correcao[word]
            word = unidecode(word)
            word_Final = [token.lemma_ for token in nlp(word)]
            Final_words.extend(word_Final)
    df.loc[index,'text_final'] = str(Final_words)
    
Test_X_Tfidf = Tfidf_vect.transform(df['text_final'])
Test_Y = Encoder.fit_transform(df['sentiment'])

# ajustar o conjunto de dados de treinamento no classificador NB
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)

#imprimir na tela a acurácia, f-measure, precisão e recall
print("----------- Naive Bayes -----------")
print("Naive Bayes Accuracy: ",accuracy_score(predictions_NB,Test_Y)*100)
print("Naive Bayes F-Measure: ",f1_score(predictions_NB, Test_Y, average="macro")*100)
print("Naive Bayes Precision: ",precision_score(predictions_NB, Test_Y, average="macro")*100)
print("Naive Bayes Recall: ",recall_score(predictions_NB, Test_Y, average="macro")*100)

# Classificador SVM
# ajustar o conjunto de dados de treinamento no classificador SVM
SVM = svm.SVC()
SVM.fit(Train_X_Tfidf,Train_Y)
# prever os rótulos no conjunto de dados de validação
predictions_SVM = SVM.predict(Test_X_Tfidf)
#imprimir na tela a acurácia, f-measure, precisão e recall
print("-------------- SVM ----------------")
print("SVM Accuracy: ",accuracy_score(predictions_SVM, Test_Y)*100)
print("SVM F-Measure: ",f1_score(predictions_SVM, Test_Y, average="macro")*100)
print("SVM Precision: ",precision_score(predictions_SVM, Test_Y, average="macro")*100)
print("SVM Recall: ",recall_score(predictions_SVM, Test_Y, average="macro")*100)

#-------------------------------------------#
# Experimento 2
#-------------------------------------------#
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
Y = Encoder.fit_transform(Corpus['sentiment'])

# TF-IDF
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
X = Tfidf_vect.transform(Corpus['text_final'])

Naive = naive_bayes.MultinomialNB()
SVM = svm.SVC()

nome_metricas = ['accuracy', 'precision_macro', 'recall_macro']
metricas = cross_validate(SVM, X, y, cv=10, scoring=nome_metricas)
print('--------- SVM ---------')
for met in metricas:
    print(f"- {met}:")
    print(f"-- {metricas[met].mean()}")
print('\n')

metricas = cross_validate(Naive, X, y, cv=10, scoring=nome_metricas)
print('--------- Naive Bayes ---------')
for met in metricas:
    print(f"- {met}:")
    print(f"-- {metricas[met].mean()}")
print('\n')

metricas = cross_validate(RF, X, y, cv=10, scoring=nome_metricas)
print('--------- Random Forest ---------')
for met in metricas:
    print(f"- {met}:")
    print(f"-- {metricas[met].mean()}")
print('\n')
