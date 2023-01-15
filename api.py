import snscrape.modules.twitter as sntwitter
import pandas as pd
import itertools

# Exemplo de coleta 

dat_ini = '2022-10-01'
dat_fim = '2022-10-02'

def search_bolsonaro_sudeste(text = 'bolsonaro', start = dat_ini, end = dat_fim):
    tweets =[]
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Belo Horizonte" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_bh = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"São Paulo" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_sp = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Rio de Janeiro" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_rj = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Vitória" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_vt = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    
    # Junção dos tweets das cidades em 1 arquivo só
    df_final_bolsonaro_sudeste = pd.concat([tweets_bolsonaro_bh, tweets_bolsonaro_sp, tweets_bolsonaro_rj, tweets_bolsonaro_vt], ignore_index=True)
    df_final_bolsonaro_sudeste.to_csv('df_bolsonaro_sudeste_outubro.xlsx')

search_bolsonaro_sudeste()

def search_bolsonaro_sul(text = 'bolsonaro', start = dat_ini, end = dat_fim, numberOfTweets = 200000):
    tweets =[]
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Porto Alegre" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_pa = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou pa ", len(tweets_bolsonaro_maio_pa))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Florianópolis" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_fl = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou fl ", len(tweets_bolsonaro_maio_fl))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Curitiba" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_ct = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou ct ", len(tweets_bolsonaro_maio_ct))
    
    df_final_bolsonaro_sul = pd.concat([tweets_bolsonaro_maio_pa, tweets_bolsonaro_maio_fl, tweets_bolsonaro_maio_ct], ignore_index=True)

    df_final_bolsonaro_sul.to_csv('df_bolsonaro_sul_outubro.xlsx')

search_bolsonaro_sul()

def search_bolsonaro_centro_oeste(text = 'bolsonaro', start = dat_ini, end = dat_fim, numberOfTweets = 200000):
    tweets =[]
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Campo Grande" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_cg = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou cg ", len(tweets_bolsonaro_maio_cg))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Cuiabá" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_cb = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou cb ", len(tweets_bolsonaro_maio_cb))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Goiânia" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_go = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou go ", len(tweets_bolsonaro_maio_go))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Brasília" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_br = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou br ", len(tweets_bolsonaro_maio_br))
    
    df_final_bolsonaro_centro_oeste = pd.concat([tweets_bolsonaro_maio_cg, tweets_bolsonaro_maio_cb, tweets_bolsonaro_maio_go, tweets_bolsonaro_maio_br], ignore_index=True)

    df_final_bolsonaro_centro_oeste.to_csv('df_bolsonaro_centro_oeste_outubro.xlsx')

search_bolsonaro_centro_oeste()

def search_bolsonaro_nordeste(text = 'bolsonaro', start = dat_ini, end = dat_fim, numberOfTweets = 200000):
    tweets =[]
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Salvador" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_ss = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou ss ", len(tweets_bolsonaro_maio_ss))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Aracaju" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_aj = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou aj ", len(tweets_bolsonaro_maio_aj))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Maceió" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_mc = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou mc ", len(tweets_bolsonaro_maio_mc))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Recife" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_rc = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou rc ", len(tweets_bolsonaro_maio_rc))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"João Pessoa" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_jp = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou jp ", len(tweets_bolsonaro_maio_jp))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Natal" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_na = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou na ", len(tweets_bolsonaro_maio_na))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Fortaleza" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_fo = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou fo ", len(tweets_bolsonaro_maio_fo))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"São Luís" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_sl = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou sl ", len(tweets_bolsonaro_maio_sl))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Teresina" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_te = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou te ", len(tweets_bolsonaro_maio_te))
    
    df_final_bolsonaro_nordeste = pd.concat([tweets_bolsonaro_maio_ss, tweets_bolsonaro_maio_aj, tweets_bolsonaro_maio_mc, tweets_bolsonaro_maio_rc, tweets_bolsonaro_maio_jp, tweets_bolsonaro_maio_na, tweets_bolsonaro_maio_fo, tweets_bolsonaro_maio_sl, tweets_bolsonaro_maio_te], ignore_index=True)

    df_final_bolsonaro_nordeste.to_csv('df_bolsonaro_nordeste_outubro.xlsx')

search_bolsonaro_nordeste()

def search_bolsonaro_norte(text = 'bolsonaro', start = dat_ini, end = dat_fim, numberOfTweets = 200000):
    tweets =[]
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Palmas" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_pl = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou pl ", len(tweets_bolsonaro_maio_pl))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Belém" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_bl = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou bl ", len(tweets_bolsonaro_maio_bl))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Macapá" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_mc = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou mc ", len(tweets_bolsonaro_maio_mc))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Manaus" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_mn = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou mn ", len(tweets_bolsonaro_maio_mn))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Boa Vista" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_bv = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou bv ", len(tweets_bolsonaro_maio_bv))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Porto Velho" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_pv = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou pv ", len(tweets_bolsonaro_maio_pv))
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{text} since:{start} until:{end} near:"Rio Branco" lang:pt').get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username])
    tweets_bolsonaro_maio_rc = pd.DataFrame(tweets, columns = ['date', 'tweet id', 'content', 'username'])
    print("Terminou rc ", len(tweets_bolsonaro_maio_rc))
    
    df_final_bolsonaro_norte = pd.concat([tweets_bolsonaro_maio_pl, tweets_bolsonaro_maio_bl, tweets_bolsonaro_maio_mc, tweets_bolsonaro_maio_mn, tweets_bolsonaro_maio_bv, tweets_bolsonaro_maio_pv, tweets_bolsonaro_maio_rc], ignore_index=True)

    df_final_bolsonaro_norte.to_csv('df_bolsonaro_norte_outubro.xlsx')

search_bolsonaro_norte()
