import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from datetime import datetime
import calendar
from datetime import date
from matplotlib import cm
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import locale
#locale.setlocale(locale.LC_TIME, "it_IT.utf8")
import altair as alt
# https://docs.streamlit.io/develop/api-reference/charts/st.altair_chart

with st.sidebar:
    st.image("Gpi_CMYK_payoff.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="PNG")

hypermeteo = pd.read_csv('hypermeteo_new.csv', sep = ';', encoding= 'unicode_escape')

hypermeteo["DATA"] = pd.to_datetime(hypermeteo["DATA"], format='%Y-%m-%d')
hypermeteo_23 = hypermeteo
hypermeteo_23 = hypermeteo_23[hypermeteo_23["DATA"].dt.year==2023]
hypermeteo['CAP'] = hypermeteo['CAP'].astype(str)
hypermeteo_23['CAP'] = hypermeteo_23['CAP'].astype(str)

source = hypermeteo
source_23 = hypermeteo_23

scale = alt.Scale(
domain=["25121", "25122", "25123", "25124", "25125", "25126", "25127", "25128", "25129", "25131", "25132", "25133", "25134", "25135", "25136"],
range=["#ffef00", "#fd7c00", "#e7ba52", "#783f04", "#f4bbbb", "#fc3c80", "#ceaadf", "#9467bd", "#a8e10c", "#10af4d", "#1b7e4e", "#4ed4c5", "#aad6ec", "#256d91", "#a08d83"],
#scheme = "tableau20"
)
color = alt.Color("CAP:N", scale=scale)


option = st.selectbox(
   "Di quale variabile ambientale vorresti vedere il trend per i vari CAP di Brescia?",
   ("PM10", "PM2.5", "NOx",	"SO2",	"03", "NMVOC",	"Temperatura minima", "Temperatura massima", "Umidità minima", "Umidità massima", "Precipitazioni", "Irradiazione solare"),
    placeholder="Seleziona variabile"
)
st.write('Hai selezionato la seguente variabile ambientale:', option)

if option=="PM10":
    option = "PM10_DAILY_ugm-3"
elif option=="PM2.5":
    option = "PM2p5_DAILY_ugm-3"
elif option=="NOx":
    option = "NOx_DAILY_ugm-3"
elif option=="SO2":
    option = "SO2_DAILY_ugm-3" 
elif option=="03":
    option = "O3_DAILY_ugm-3" 
elif option=="NMVOC":
    option = "NMVOC_DAILY_ugm-3"
elif option=="Temperatura minima":
    option = "TMIN_DAILY_C" 
elif option=='Temperatura massima':
    option='TMAX_DAILY_C'	
elif option=='Umidità minima':
    option='RHMIN_DAILY_%'
elif option=='Umidità massima': 
    option='RHMAX_DAILY_%'
elif option=='Precipitazioni':
    option='PREC_DAILY_mm'	
elif option=='Irradiazione solare':
    option='SSWTOT_DAILY_Whm-2'

######## PRIMO GRAFICO
points = (
    alt.Chart(source_23, title="Variabili ambientali Brescia 2023").mark_point(size=3).encode(
    x='DATA',
    y=option,
    color=alt.Color("CAP:N", scale=scale),
    tooltip=['CAP', 'DATA', option]
).interactive().properties(
    # Adjust chart width and height to match size of legend
    width=600,
    height=450
)
)

st.altair_chart(points, theme="streamlit", use_container_width=True)
######## FINE PRIMO GRAFICO

######## SECONDO GRAFICO, FATTO DA NOI

points = (
    alt.Chart(source, title="Variabili ambientali Brescia 2020-2024").mark_point(size=3).encode(
    x='DATA',
    y=option,
    color=alt.Color("CAP:N", scale=scale),
    tooltip=['CAP', 'DATA', option]
).interactive().properties(
    # Adjust chart width and height to match size of legend
    width=600,
    height=450
)
)

st.altair_chart(points, theme="streamlit", use_container_width=True)
######## FINE SECONDO GRAFICO, FATTO DA NOI


hypermeteo_range = hypermeteo

hypermeteo_range['NOx_DAILY_ugm-3'] = (hypermeteo_range['NOx_DAILY_ugm-3']>=25).astype(int)
hypermeteo_range['PM2p5_DAILY_ugm-3'] = (hypermeteo_range['PM2p5_DAILY_ugm-3']>= 15).astype(int)
hypermeteo_range['PM10_DAILY_ugm-3'] = (hypermeteo_range['PM10_DAILY_ugm-3']>= 45).astype(int)
hypermeteo_range['O3_DAILY_ugm-3'] = (hypermeteo_range['O3_DAILY_ugm-3']>= 100).astype(int)
hypermeteo_range['SO2_DAILY_ugm-3'] = (hypermeteo_range['SO2_DAILY_ugm-3']>= 40).astype(int)
hypermeteo_range['TMIN_DAILY_C'] = (hypermeteo_range['TMIN_DAILY_C']<= -10).astype(int)
hypermeteo_range['TMAX_DAILY_C'] = (hypermeteo_range['TMAX_DAILY_C']>= 35).astype(int)
hypermeteo_range['RHMIN_DAILY_%'] = (hypermeteo_range['RHMIN_DAILY_%']>= 15).astype(int)
hypermeteo_range['RHMAX_DAILY_%'] = (hypermeteo_range['RHMAX_DAILY_%']>= 95).astype(int)
hypermeteo_range['PREC_DAILY_mm'] = (hypermeteo_range['PREC_DAILY_mm']>= 10).astype(int)
hypermeteo_range['SSWTOT_DAILY_Whm-2'] = (hypermeteo_range['SSWTOT_DAILY_Whm-2']> 8500).astype(int)
hypermeteo_range['TOT'] = hypermeteo_range['PM10_DAILY_ugm-3'] + hypermeteo_range['PM2p5_DAILY_ugm-3'] + hypermeteo_range['NOx_DAILY_ugm-3'] + hypermeteo_range['SO2_DAILY_ugm-3'] + hypermeteo_range['O3_DAILY_ugm-3'] + hypermeteo_range['TMIN_DAILY_C'] + hypermeteo_range['TMAX_DAILY_C'] + hypermeteo_range['RHMIN_DAILY_%'] + hypermeteo_range['RHMAX_DAILY_%'] + hypermeteo_range['PREC_DAILY_mm'] + hypermeteo_range['SSWTOT_DAILY_Whm-2']

anno = st.selectbox(
   "Di quale anno vorresti vedere il numero di valori (su 12) che sono stati fuori range per i vari CAP?",
   (2020, 2021, 2022, 2023, 2024),
   placeholder="Seleziona anno"
)

hypermeteo_range= hypermeteo_range[hypermeteo_range["DATA"].dt.year==anno]
source_range = hypermeteo_range

# Bottom panel is a bar chart of weather type
bars = (
    alt.Chart()
    .mark_bar()
    .encode(
        x="count()",
        y="TOT:Q",
        color=alt.Color("CAP:N", scale=scale),
    )
    .properties(
        width=550,
    )
)

chart = alt.vconcat(bars, data=source_range, title="Valori fuori soglia nel %s" %anno)

st.altair_chart(chart, theme="streamlit", use_container_width=True)
