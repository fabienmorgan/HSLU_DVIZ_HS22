import numpy as np
import pandas as pd
from dash import Dash, html, dcc
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

### Viz Prep ###

ratings_single_account = pd.read_csv('Data_Processing/data/ratings_single_account.csv')
movies_single_account = pd.read_csv('Data_Processing/data/movies_single_account.csv')
all_ratings = pd.read_csv('Data_Processing/data/all_ratings.csv')

pio.templates.default = "plotly"

def ratings_for_uid(user_id):
    all_user_ratings = all_ratings.query(f'userId == {user_id}').copy()
    all_user_ratings.sort_values(by=['rating_date'], inplace=True)
    return all_user_ratings

def apply_grid_bg_design(go):
    go.update_layout(
        plot_bgcolor= 'rgb(244,247,251)',
    )
    go.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='lightgrey'
    )
    go.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='lightgrey'
    )

uid = 134596
all_user_ratings = ratings_for_uid(uid)

fgo = make_subplots(rows=1, cols=2, column_widths=[0.75,0.25], subplot_titles=('Rating Distribution','Histogram'), shared_yaxes=True)

fig1_1 = px.scatter(all_user_ratings,x='rating_date',y='rating', hover_data=['movieId'], color="rating")
fig1_1.update_traces(
    marker=dict(size=16, symbol="line-ns", line=dict(width=0, color="DarkSlateGrey")),
    #marker=dict(size=5, symbol="circle", line=dict(width=0, color="DarkSlateGrey")),
    selector=dict(mode="markers"),
)

fig1_2 = px.histogram(all_user_ratings,y='rating')
fig1_2.update_traces(
    marker_color='darkgrey'
)


fgo.update_layout(
    plot_bgcolor= 'rgb(244,247,251)',
    yaxis = dict(
        tickmode = 'array',
        tickvals = np.arange(0.5,5.5,0.5)
    ),
    title_text=f'Movie Ratings of User <b>{uid}</b> ({len(all_user_ratings)} Ratings)'
)

fgo.update_xaxes(title_text="Time", titlefont_size=12, row = 1, col = 1)
fgo.update_yaxes(title_text="Rating", titlefont_size=12, row = 1, col = 1)
fgo.update_xaxes(title_text="Count per Rating Level", titlefont_size=12, row = 1, col = 2)

fgo.update_xaxes(
    showgrid=True, gridwidth=1, gridcolor='lightgrey'
)
fgo.update_yaxes(
    showgrid=True, gridwidth=1, gridcolor='lightgrey'
)


fgo.add_trace(fig1_1['data'][0], row=1, col=1)
fgo.add_trace(fig1_2['data'][0], row=1, col=2)

### Dash Part ###

app = Dash(__name__)

html_structure = [
    html.H1(children='Finding Bot Activity in MovieLens Community Ratings'),

    html.Div(children='''
        It is with great pleasure to inform you that Fabien has a small pp.
    '''),

    html.Div(id='wrapper', style={'textAlign': 'center'}, children=[
        dcc.Graph(figure=fgo,
        id='chart1',
        style={"display": "inline-block", "margin": "0 auto", "width": "80%"})
    ]),

    html.Div(children="Below Chart")
]

app.layout = html.Div(children=html_structure)

if __name__ == '__main__':
    app.run_server(debug=True)