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

#Data transformation
def return_list_of_binned_ratings(list_of_ratings, ratings):
    ratings_for_movie_binned_lst = len(list_of_ratings) * [0]
    
    for rating in ratings:
        index = list_of_ratings.index(rating)
        ratings_for_movie_binned_lst[index] += 1

    return dict(zip(list_of_ratings,ratings_for_movie_binned_lst))

common = all_ratings.merge(ratings_single_account,on=['rating_id'])
ratings_without_single_account = all_ratings[~all_ratings.rating_id.isin(common.rating_id)]

list_of_ratings = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]  
list_of_ratings = list_of_ratings[::-1]

binned_ratings_single_account = return_list_of_binned_ratings(list_of_ratings, ratings_single_account['rating'])
binned_ratings_without_single_accounts = return_list_of_binned_ratings(list_of_ratings, ratings_without_single_account['rating'])

binned_ratings_single_account_percentage = {k: v / len(ratings_single_account) for k, v in binned_ratings_single_account.items()}
binned_ratings_without_single_accounts_percentage = {k: v / len(ratings_without_single_account) for k, v in binned_ratings_without_single_accounts.items()}

#Horizontal Barchart
def calcuate_position_outside_annotations(dict_of_stacked_elements, list_of_ratings, threshold):
    sum = 0
    dict_position_to_small_ratings = {}
    for rating in list_of_ratings:
        if dict_of_stacked_elements[rating] >= threshold:
            sum += dict_of_stacked_elements[rating]
        else:
            dict_position_to_small_ratings[rating] = sum + (dict_of_stacked_elements[rating] / 2)
            sum += dict_of_stacked_elements[rating]

    return dict_position_to_small_ratings

def create_stacked_barchart_subplot(figure, threshold, description, list_with_content, position_x, position_y, colors, list_of_ratings, text_over):
    if text_over:
        y_offset = 0.5
    else:
        y_offset = -0.5

    traces = []
    for i, rating in enumerate(list_of_ratings):
        text = 'inside' if list_with_content[rating] > threshold else 'none'

        trace = go.Bar(y=description,
                        x=[list_with_content[rating]],
                        name=f'{rating}',
                        showlegend=False,
                        orientation='h',
                        textangle=0,
                        marker_color=colors[i],
                        textposition=text,
                        text=rating,
                        insidetextanchor='middle')
        traces.append(trace)

    position_to_small_content = calcuate_position_outside_annotations(list_with_content, list_of_ratings, threshold)

    for rating in position_to_small_content:
        figure.add_annotation(x=position_to_small_content[rating], y=y_offset,
                            text=rating,
                            showarrow=False,
                            xref=f'x{position_x}',
                            yref=f'y{position_y}')
    
    figure.add_traces(traces, position_y, position_x)

#Max threshold that makes sense
#threshold = 0.0312
threshold = 0.02
descripion_only_one_ratings = ['Score distribution from <br> Users with only one rating']
descriont_multiple_ratings = ['Score distribution from <br> Users with multiple ratings']
colors = px.colors.sequential.Viridis
fig_horizontal_barchart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing = 0.00)
create_stacked_barchart_subplot(fig_horizontal_barchart, threshold, descripion_only_one_ratings, binned_ratings_single_account_percentage, 1, 1, colors, list_of_ratings, True)
create_stacked_barchart_subplot(fig_horizontal_barchart, threshold, descriont_multiple_ratings, binned_ratings_without_single_accounts_percentage, 1, 2, colors, list_of_ratings, False)

fig_horizontal_barchart.update_layout(barmode='stack')
fig_horizontal_barchart.update_layout(title={'text': "Comparison of rating score from users with only one ratign compared to score from users with multiple ratings"})
fig_horizontal_barchart.update_xaxes(visible=False)
fig_horizontal_barchart.update_layout(plot_bgcolor="#FFFFFF")

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

    html.Div(children="Below Chart"),

    html.Div(id='wrapper', style={'textAlign': 'center'}, children=[
        dcc.Graph(figure=fig_horizontal_barchart,
        id='horizontal_barchart',
        style={"display": "inline-block", "margin": "0 auto", "width": "80%"})
    ])
]

app.layout = html.Div(children=html_structure)

if __name__ == '__main__':
    app.run_server(debug=True)