import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import datetime 
import plotly.express as px

# Quantidade de tweets coletados por mês
trace1 = go.Bar(x = ['Junho', 'Julho', 'Agosto', 'Setembro'], #meses
                y = [677116, 393644, 1618228, 326847], #valor
                name = 'Jair Bolsonaro',
                marker = {'color': '#38e62c'},
                text_auto=True)
trace2 = go.Bar(x = ['Junho', 'Julho', 'Agosto', 'Setembro'], #meses
                y = [415246, 460611, 1512184, 479789], #valor
                name = 'Lula',
                marker = {'color': '#f51616'},
                text_auto=True)
trace3 = go.Bar(x = ['Junho', 'Julho', 'Agosto', 'Setembro'], #meses
                y = [106655, 55995, 390979, 179816], #valor
                name = 'Ciro Gomes',
                marker = {'color': '#1d4fdb'},
                text_auto=True)
trace4 = go.Bar(x = ['Junho', 'Julho', 'Agosto', 'Setembro'], #meses
                y = [22559, 12591, 37946, 18327], #valor
                name = 'Simone Tebet',
                marker = {'color': '#f0f029'},
                text_auto=True)
trace5 = go.Bar(x = ['Junho', 'Julho', 'Agosto', 'Setembro'], #meses
                y = [131, 542, 8229, 865], #valor
                name = "Felipe D'avila",
                marker = {'color': '#eb7317'},
                text_auto=True)

data = [trace1, trace2, trace3, trace4, trace5]
fig = go.Figure(data=data)
py.iplot(fig)

# Comparação do desempenho geral dos candidatos
trace1 = go.Bar(x = ['Ciro Gomes', "Felipe d'Avila", 'Jair Bolsonaro', 'Lula', 'Simone Tebet'], #meses
                y = [65, 73, 44, 36, 39], #valor
                name = 'Taxa de aprovação no Twitter',
                text = ['65%', '73%', '44%', '37%', '39%'],
                textposition='outside')
trace2 = go.Bar(x = ['Ciro Gomes', "Felipe d'Avila", 'Jair Bolsonaro', 'Lula', 'Simone Tebet'], #meses
                y = [3, 0.4, 43, 48, 4], #valor
                name = 'Percentual de votos recebidos',
                text = ['3%', '0,4%', '43%', '48%', '4%'],
                textposition='outside')

data = [trace1, trace2]
fig = go.Figure(data=data)
py.iplot(fig)
