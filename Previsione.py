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
#locale.setlocale(locale.LC_ALL, 'it_IT.utf8')

st.set_page_config(
    page_title="Previsione PS",
    page_icon="⚕️",
)

with st.sidebar:
    st.image("Gpi_CMYK_payoff.png", caption=None, width=250, use_column_width=None, clamp=False, channels="RGB", output_format="PNG")

hypermeteo = pd.read_csv('hypermeteo.csv', sep = ';', encoding= 'unicode_escape')
prediction = pd.read_csv('pred.csv', sep = ';', encoding= 'unicode_escape')
prediction['prediction'] = round(prediction['prediction'])
cardio = pd.read_csv('cardio.csv', sep = ';', encoding= 'unicode_escape')
cardio['prediction'] = round(cardio['prediction'])
resp = pd.read_csv('resp.csv', sep = ';', encoding= 'unicode_escape')
resp['prediction'] = round(resp['prediction'])


hypermeteo['NOx_DAILY_ugm-3'] = (hypermeteo['NOx_DAILY_ugm-3']>=25).astype(int)
hypermeteo['PM2p5_DAILY_ugm-3'] = (hypermeteo['PM2p5_DAILY_ugm-3']>= 15).astype(int)
hypermeteo['PM10_DAILY_ugm-3'] = (hypermeteo['PM10_DAILY_ugm-3']>= 45).astype(int)
hypermeteo['O3_DAILY_ugm-3'] = (hypermeteo['O3_DAILY_ugm-3']>= 100).astype(int)
hypermeteo['SO2_DAILY_ugm-3'] = (hypermeteo['SO2_DAILY_ugm-3']>= 40).astype(int)
hypermeteo['TMIN_DAILY_C'] = (hypermeteo['TMIN_DAILY_C']<= -10).astype(int)
hypermeteo['TMAX_DAILY_C'] = (hypermeteo['TMAX_DAILY_C']>= 35).astype(int)
hypermeteo['RHMIN_DAILY_%'] = (hypermeteo['RHMIN_DAILY_%']>= 15).astype(int)
hypermeteo['RHMAX_DAILY_%'] = (hypermeteo['RHMAX_DAILY_%']>= 95).astype(int)
hypermeteo['PREC_DAILY_mm'] = (hypermeteo['PREC_DAILY_mm']>= 10).astype(int)
hypermeteo['SSWTOT_DAILY_Whm-2'] = (hypermeteo['SSWTOT_DAILY_Whm-2']> 8500).astype(int)

hypermeteo['TOT'] = hypermeteo['PM10_DAILY_ugm-3'] + hypermeteo['PM2p5_DAILY_ugm-3'] + hypermeteo['NOx_DAILY_ugm-3'] + hypermeteo['SO2_DAILY_ugm-3'] + hypermeteo['O3_DAILY_ugm-3'] + hypermeteo['TMIN_DAILY_C'] + hypermeteo['TMAX_DAILY_C'] + hypermeteo['RHMIN_DAILY_%'] + hypermeteo['RHMAX_DAILY_%'] + hypermeteo['PREC_DAILY_mm'] + hypermeteo['SSWTOT_DAILY_Whm-2']

hypermeteo["DATA"] = pd.to_datetime(hypermeteo["DATA"], format='%d/%m/%y')
later = hypermeteo.copy()

min_date = pd.to_datetime("2022-01-01", format="%Y-%m-%d")
max_date = pd.to_datetime("2022-12-31", format="%Y-%m-%d")

with st.expander("Seleziona intervallo temporale"):
    start_date = st.date_input("Scegli data di inizio previsione:", value=pd.to_datetime("2022-01-01", format="%Y-%m-%d"), min_value = min_date, max_value = max_date)
    end_date = st.date_input("Scegli data di fine previsione:", value=pd.to_datetime("2022-12-31", format="%Y-%m-%d"), min_value = min_date, max_value = max_date)

start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
end_date = pd.to_datetime(end_date, format='%Y-%m-%d')

hypermeteo=hypermeteo[(hypermeteo['DATA'].between(start_date, end_date))]

hypermeteo_2 = hypermeteo.groupby('DATA')['TOT'].median()
hypermeteo_2 = hypermeteo_2.to_frame().sort_values(by='DATA')
hypermeteo_2 = hypermeteo_2.reset_index()

hypermeteo_2["DATA"] = hypermeteo_2["DATA"].dt.strftime('%d %B')

df_tot = pd.DataFrame()
df_pred = pd.DataFrame()
df_cardio = pd.DataFrame()
df_resp = pd.DataFrame()
df_tot['DATA'] = hypermeteo_2['DATA']
df_pred['DATA'] = hypermeteo_2['DATA']
df_cardio['DATA'] = hypermeteo_2['DATA']
df_resp['DATA'] = hypermeteo_2['DATA']
df_tot['TOT'] = hypermeteo_2['TOT']
df_pred['prediction'] = prediction['prediction']
df_cardio['prediction'] = cardio['prediction']
df_resp['prediction'] = resp['prediction']


## inizio grafico

fig = go.Figure(
    data=go.Bar(
        x=df_tot['DATA'],
        y=df_tot['TOT'],
        name="Valori ambientali",
        marker=dict(color="#f48918")
    )
)

fig.add_trace(
    go.Scatter(
        x=df_pred['DATA'],
        y=df_pred['prediction'],
        yaxis="y2",
        name="Accessi PS",
        marker=dict(color="#962086")
    )
)

fig.update_layout(
    legend=dict(
    orientation="h",
    yanchor="top",
    y=-0.2,
    xanchor="left",
    x=0.01),
    yaxis=dict(
        title=dict(text="Totale valori fuori soglia"),
        side="left",
        range=[0, 6],
    ),
    yaxis2=dict(
        title=dict(text="Totale accessi previsti"),
        side="right",
        range=[0, 48],
        overlaying="y",
        tickmode="sync"
    ),
    xaxis_title='Data'
)

fig.update_xaxes(nticks=6)

fig.update_layout(
title = {'text':'Numero di accessi previsti e valori ambientali fuori soglia',
        'x' : 0.5,
        'xanchor': 'center',
        'y' : 0.8,
        'yanchor': 'top'
}
)

st.plotly_chart(fig)

## fine grafico

# Grafico cardio

fig_cardio = go.Figure(
    data=go.Bar(
        x=df_tot['DATA'],
        y=df_tot['TOT'],
        name="Valori ambientali",
        marker=dict(color="#f48918")
    )
)

fig_cardio.add_trace(
    go.Scatter(
        x=df_cardio['DATA'],
        y=df_cardio['prediction'],
        yaxis="y2",
        name="Ricoveri da PS",
        marker=dict(color="#962086")
    )
)

fig_cardio.update_layout(
    legend=dict(
    orientation="h",
    yanchor="top",
    y=-0.2,
    xanchor="left",
    x=0.01),
    yaxis=dict(
        title=dict(text="Totale valori fuori soglia"),
        side="left",
        range=[0, 6],
    ),
    yaxis2=dict(
        title=dict(text="Totale ricoveri previsti"),
        side="right",
        range=[0, 2],
        overlaying="y",
        tickmode="sync"
    ),
    xaxis_title='Data'
)

fig_cardio.update_xaxes(nticks=6)

fig_cardio.update_layout(
title = {'text':'Numero di ricoveri previsti per malattie cardiovascolari e valori ambientali fuori soglia',
        'x' : 0.5,
        'xanchor': 'center',
        'y' : 0.8,
        'yanchor': 'top'
}
)

# Grafico resp

fig_resp = go.Figure(
    data=go.Bar(
        x=df_tot['DATA'],
        y=df_tot['TOT'],
        name="Valori ambientali",
        marker=dict(color="#f48918")
    )
)

fig_resp.add_trace(
    go.Scatter(
        x=df_resp['DATA'],
        y=df_resp['prediction'],
        yaxis="y2",
        name="Ricoveri da PS",
        marker=dict(color="#962086")
    )
)

fig_resp.update_layout(
    legend=dict(
    orientation="h",
    yanchor="top",
    y=-0.2,
    xanchor="left",
    x=0.01),
    yaxis=dict(
        title=dict(text="Totale valori fuori soglia"),
        side="left",
        range=[0, 6],
    ),
    yaxis2=dict(
        title=dict(text="Totale ricoveri previsti"),
        side="right",
        range=[0, 6],
        overlaying="y",
        tickmode="sync"
    ),
    xaxis_title='Data'
)

fig_resp.update_xaxes(nticks=6)

fig_resp.update_layout(
title = {'text':'Numero di ricoveri previsti per malattie respiratorie e valori ambientali fuori soglia',
        'x' : 0.5,
        'xanchor': 'center',
        'y' : 0.8,
        'yanchor': 'top'
}
)

accuratezza = st.checkbox(":mag: Confronto tra previsione e valori reali di accessi al PS")

if accuratezza:
    st.image("Valori effettivi e predetti.jpg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="PNG")

st.markdown("""---""")

ricoveri = st.checkbox(":hospital: Grafici ricoveri")

if ricoveri:
    st.plotly_chart(fig_cardio)
    st.plotly_chart(fig_resp)
    accuratezza_cardio_resp = st.checkbox(":mag: Confronto tra previsione e valori reali di ricoveri da triage")
    if accuratezza_cardio_resp:
        st.image("Valori effettivi e predetti cardiovascolare range lag 1 giorno.jpg", caption='Confronto per ricoveri per malattie cardiovascolari', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="PNG")
        st.image("Valori effettivi e predetti respiratorio ranges lag 1 giorno.jpg", caption='Confronto per ricoveri per malattie respiratorie', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="PNG")

st.markdown("""---""")

# Mappa Brescia

jan_1 = date(2022, 1, 1)
dec_31 = date(2022, 12, 31)

with st.expander("Seleziona l'intervallo di tempo per cui visualizzare la media dei valori ambientali fuori soglia"):
    interval_date = st.date_input("Scegli l'intervallo di date:", (jan_1, dec_31), jan_1, dec_31, format="YYYY/MM/DD")
    agree = st.checkbox('Stesso intervallo selezionato per il precedente grafico')
    if agree:
        interval_date = (start_date, end_date)

interval_date = pd.to_datetime(interval_date, format='%Y-%m-%d')

#later['DATA'] = pd.to_datetime(later["DATA"], format='Y-%m-%d')
later=later[(later['DATA'].between(interval_date[0], interval_date[1]))]

df_lat_lon  = pd.DataFrame()
df_lat_lon['CAP'] = [25121, 25122, 25123, 25124, 25125, 25126, 25127, 25128, 25129, 25131, 25132, 25133, 25134, 25135, 25136]
df_lat_lon['lat'] = [45.53569852790294, 45.53937111311257, 45.54511384110169, 45.51185339367707, 45.52034383913882, 45.54215077369812, 45.55991665286687, 45.558834619732835, 45.49822512580028, 45.5010238104388, 45.545654153228774, 45.5683370129516, 45.50303595174109, 45.51862245633552, 45.57987442372339]
df_lat_lon['lon'] = [10.226469983115456, 10.216011789254754, 10.256035581372357, 10.219970002783604, 10.17898527039134, 10.192798463757107, 10.190401896642486, 10.218750030946387, 10.272947248954008, 10.171937024931385, 10.159549777784841, 10.251040801876048, 10.254802314682074, 10.282920791967863, 10.22884619352044]
merge_per_mappa = later.merge(df_lat_lon, on=['CAP'], how = 'outer')
merge_per_mappa = round(merge_per_mappa.groupby(['CAP', 'lat', 'lon'])['TOT'].mean(), 3)
merge_per_mappa = merge_per_mappa.reset_index()
mappa = merge_per_mappa[['CAP', 'lat', 'lon', 'TOT']]


column_layer = pdk.Layer(
    "ColumnLayer",
    data=mappa,
    get_position=["lon", "lat"],
    get_fill_color=[180, 0, 200, 140],
    get_elevation="TOT",
    elevation_scale=500,
    radius=200,
    pickable=True,
    auto_highlight=True
)

tooltip = {
    "html": "{CAP}: <b>{TOT}</b>",
    "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
}

r = pdk.Deck(
    column_layer,
    map_provider="mapbox",
    map_style=None,
    tooltip = tooltip,
    initial_view_state=pdk.ViewState(
        latitude=45.541553,
        longitude=10.211802,
        zoom=11,
        pitch=50,
    ),
)

st.pydeck_chart(r)
