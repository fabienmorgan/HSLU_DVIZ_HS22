import numpy as np
import pandas as pd
from dash import Dash, html, dcc
import plotly.graph_objects as go
import plotly.express as px
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


### Many Ratings Part ###

def ratings_for_uid(user_id):
    all_user_ratings = all_ratings.query(f'userId == {user_id}').copy()
    all_user_ratings.sort_values(by=['rating_date'], inplace=True)
    all_user_ratings['rating_date'] = pd.to_datetime(all_user_ratings['rating_date'])
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
    

def plot_indicators():
    all_user_ratings = ratings_for_uid(172357)

    t_range = ['2016-06-25 08:00:00','2016-06-25 12:00:00']
    all_user_ratings = all_user_ratings.query('rating_date >= @t_range[0] and rating_date <= @t_range[1]')

    fgo2 = make_subplots(rows=1, cols=2, column_widths=[0.5,0.5], subplot_titles=(f'(1) Rating Bursts','(2) Unnatural Distribution'))

    fig2_1 = px.scatter(all_user_ratings,x='rating_date',y='rating', range_x=t_range)
    fig2_2 = px.histogram(all_user_ratings,x='rating')

    fgo2.add_trace(fig2_1['data'][0], row=1, col=1)
    fgo2.add_trace(fig2_2['data'][0], row=1, col=2)



    fgo2.update_layout(
        title_text=f'Potential Indicators for Bot Activity (Visualized for User 172357)',
    )

    fgo2.update_layout(
        yaxis = dict(
            tickmode = 'array',
            tickvals = np.arange(0.5,5.5,0.5)
        ),
        bargap = 0.05,
    )

    fgo2.update_xaxes(
        range=t_range,
        row=1,
        col=1,
    )

    fgo2.update_xaxes(
        tickmode = 'array',
        tickvals = np.arange(0.5,5.5,0.5),
        row=1,
        col=2,
    )

    fgo2.update_xaxes(title_text="Time", titlefont_size=12, row = 1, col = 1)
    fgo2.update_yaxes(title_text="Rating", titlefont_size=12, row = 1, col = 1)
    fgo2.update_xaxes(title_text="Rating", titlefont_size=12, row = 1, col = 2)
    fgo2.update_yaxes(title_text="Count", titlefont_size=12, row = 1, col = 2)


    fgo2.update_traces(
        marker_color='darkgrey',
    )

    apply_grid_bg_design(fgo2)

    return fgo2


def plot_strip_scatter(uid):
    all_user_ratings = ratings_for_uid(uid)

    fgo = make_subplots(rows=1, cols=2, column_widths=[0.75,0.25], subplot_titles=('Rating Distribution','Histogram'), shared_yaxes=True)

    fig1_1 = px.scatter(all_user_ratings,x='rating_date',y='rating', hover_data=['movieId'], color="rating")
    fig1_1.update_traces(
        marker=dict(size=16, symbol="line-ns", line=dict(width=0, color="DarkSlateGrey")),
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

    return fgo

def plot_freq_polygon(uid,startyear,endyear):
    all_user_ratings = ratings_for_uid(uid)
    all_user_ratings['hour'] = all_user_ratings['rating_date'].dt.hour.astype(int)

    fgo = go.Figure()

    color_seq = px.colors.qualitative.D3
    color_seq_count = 0

    empty_df = pd.DataFrame(index=np.arange(0,24,1))
    empty_df['count'] = 0

    for i in np.arange(startyear,endyear + 1,1):
        all_user_ratings_for_year_i = all_user_ratings[all_user_ratings['rating_date'].dt.year == i]

        #This only includes hours present in data
        hour_freq = all_user_ratings_for_year_i.hour.value_counts().to_frame()
        hour_freq.rename(columns={'hour':'count'},inplace=True)
        hour_freq.sort_index(inplace=True)

        #This fills in missing hours with 0
        hour_freq = hour_freq.combine_first(empty_df)
        
        fgo.add_trace(go.Scatter(x=hour_freq.index, y=hour_freq['count'], name=f'Year {i}', line=dict(color=color_seq[color_seq_count]), mode='lines+markers'))
        color_seq_count += 1


    fgo.update_xaxes(
        title_text="Timeline through the day",
        titlefont_size=12,
    )
    fgo.update_yaxes(
        title_text="Rating Count for Hour",
        titlefont_size=12,
        range=[0,120]
    )
    fgo.add_vrect(
        x0=17,
        x1=22,
        line_width=1,
        fillcolor='black',
        opacity=0.15,
        annotation_text='Hours with<br>no activity',
        annotation_position='top left',
        annotation=dict(font_size=14, font_color='black'),
    )

    fgo.update_layout(
        title_text=f'Favorite Rating Hours for User <b>{uid}</b>',
        xaxis = dict(
            tickmode = 'array',
            tickvals = np.arange(0,24,1)
        ),
    )
    apply_grid_bg_design(fgo)
    return fgo


### Dash Part ###

app = Dash(__name__)

html_structure = [
    html.H1(children='Finding Bot Activity in MovieLens Community Ratings'),

    html.Div(id='project_specification', children=[
        html.P([
        'Course: I.BA_DVIZ.H2201 @ University of Applied Sciences Lucerne',html.Br(),
        'Lecturer: Dr. Teresa Kubacka',html.Br(),
        'Authors: Fabien Morgan, Flavio Kluser',html.Br(),
        'Date: 2023-01-08',html.Br(),
    ]),
    ]),

    html.H2(children='''
        Part 1: Empty Profiles
    '''),


    html.Div(id='div_horizontal_barchart', className='viz', children=[
        dcc.Graph(figure=fig_horizontal_barchart,
        id='horizontal_barchart',
        style={"display": "inline-block", "margin": "0 auto", "width": "80%"})
    ]),

    html.H2(children='''
        Part 2: Busy Users
    '''),

    html.P([

'It is not uncommon to have very enthusiastic users. There will always be a small fraction of people with significantly more interactions than the average person. However, we do have some users with interesting rating activity.',html.Br(),html.Br(),

'The most active user has rated over 23\'000 movies in less than 3 years. We don\'t want to be judgmental, but this is a bit too much, even for a very enthusiastic movie connoisseur. The average movie length is around 90 minutes. [1] Under the assumption that this user was legit and watched all movies in his 3-year rating period, he would have spent about 32 hours a day watching movies.',html.Br(),html.Br(),

'Unlike detecting suspect activity in empty profiles, we have a lot more data to work with here. This allows us to get an insight into the rating patterns of individual users.',html.Br(),html.Br(),

'For most of the Top 20 most active users, there are obvious signs indicating bot activity. One phenomenon a lot of them share is \'Rating Bursts\', short timeframes with hundreds of ratings in minutes. Another common pattern is a very even, unnatural distribution of ratings.',html.Br(),html.Br(),

'We are more interested in the ones who deviate from obvious bot patterns - heavy users with seemingly legitimate rating activity.',html.Br(),html.Br(),


    ]),

    html.Div(className='citation', children=[
        html.P([
           '[1] Average Movie Length - https://towardsdatascience.com/are-new-movies-longer-than-they-were-10hh20-50-year-ago-a35356b2ca5b',html.Br(),html.Br(), 
    ]),
    ]),

    html.Div(id='div_plot_indicators', className='viz', children=[
        dcc.Graph(figure=plot_indicators(),
        id='plot_indicators',
        style={"display": "inline-block", "margin": "0 auto", "width": "80%"})
    ]),

    html.Div(id='div_plot_strip_scatter', className='viz', children=[
        dcc.Graph(figure=plot_strip_scatter(134596),
        id='plot_strip_scatter',
        style={"display": "inline-block", "margin": "0 auto", "width": "80%"})
    ]),

    html.Div(id='div_plot_freq_polygon', className='viz', children=[
        dcc.Graph(figure=plot_freq_polygon(134596,2011,2013),
        id='plot_freq_polygon',
        style={"display": "inline-block", "margin": "0 auto", "width": "80%"})
    ]),
]

app.layout = html.Div(children=html_structure)

if __name__ == '__main__':
    app.run_server(debug=True)