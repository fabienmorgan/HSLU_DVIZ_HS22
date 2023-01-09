import numpy as np
import pandas as pd
from dash import Dash, html, dcc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

### Viz Prep ###
reduced_ratings = pd.read_csv('Data_Processing/data/reduced_ratings.csv')

#Chart preperation
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
                        insidetextanchor='middle',
                        hovertemplate='Percentage of score '+ str(rating) +': '+ str(round(list_with_content[rating] * 100, 2)) +'%' +
                        "<extra></extra>")
        traces.append(trace)

    position_to_small_content = calcuate_position_outside_annotations(list_with_content, list_of_ratings, threshold)

    for rating in position_to_small_content:
        figure.add_annotation(x=position_to_small_content[rating], y=y_offset,
                            text=rating,
                            showarrow=False,
                            xref=f'x{position_x}',
                            yref=f'y{position_y}')
    
    figure.add_traces(traces, position_y, position_x)

#Create Horizontally stacked barchart
def build_horizontal_stacked_barchart():
    list_of_ratings = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]  
    list_of_ratings = list_of_ratings[::-1]

    binned_ratings_single_account_percentage = pd.read_csv('Data_Processing/data/data_visualizations/horizontal_stacked_barchart_single_rating.csv').to_dict()
    binned_ratings_without_single_accounts_percentage = pd.read_csv('Data_Processing/data/data_visualizations/horizontal_stacked_barchart_multi_rating.csv').to_dict()

    binned_ratings_single_account_percentage = dict(zip(list(binned_ratings_single_account_percentage['score'].values()), list(binned_ratings_single_account_percentage['value'].values())))
    binned_ratings_without_single_accounts_percentage = dict(zip(list(binned_ratings_without_single_accounts_percentage['score'].values()), list(binned_ratings_without_single_accounts_percentage['value'].values())))

    #Max threshold that makes sense
    #threshold = 0.0312
    threshold = 0.02
    descripion_only_one_ratings = ['Score distribution from <br> Users with only one rating']
    descriont_multiple_ratings = ['Score distribution from <br> Users with multiple ratings']
    colors = px.colors.sequential.Viridis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing = 0.00)
    create_stacked_barchart_subplot(fig, threshold, descripion_only_one_ratings, binned_ratings_single_account_percentage, 1, 1, colors, list_of_ratings, True)
    create_stacked_barchart_subplot(fig, threshold, descriont_multiple_ratings, binned_ratings_without_single_accounts_percentage, 1, 2, colors, list_of_ratings, False)

    fig.update_layout(barmode='stack')
    fig.update_layout(title={'text': "Comparison of rating score from users with only one ratign compared to score from users with multiple ratings"})
    fig.update_xaxes(visible=False)
    fig.update_layout(plot_bgcolor="#FFFFFF")

    return fig

def build_columnLineTimlineChart_and_BarChart():
    single_user_ratings_binned_in_years_reduced_to_hundred = pd.read_csv('Data_Processing/data/data_visualizations/columnLineTimeline_single_user_ratings.csv')
    google_trends_fake_rating_years_reduced_to_hundered = pd.read_csv('Data_Processing/data/data_visualizations/columnLineTimeline_google_trends.csv')
    growth_user_single_ratings_compared_all_ratings = pd.read_csv('Data_Processing/data/data_visualizations/bar_chart_growth.csv')

    figure = make_subplots(rows=1, cols=2,
                        subplot_titles=("Comparison of rating amount from users <br> with only one rating to search traffic of the term<br> \"fake rating\" between the years 20011 and 2017",
                                        "Yearly growth of ratings from users with only one rating<br> in relation to the growth of all ratings"), horizontal_spacing=0.2)
    colors = px.colors.sequential.Viridis

    trace_bar_trends = go.Bar(
            x=single_user_ratings_binned_in_years_reduced_to_hundred['year'],
            y=single_user_ratings_binned_in_years_reduced_to_hundred['rating_amount'],
            marker_color=colors[3], name='Rating amount',
            legendgroup='1',
            legendgrouptitle_text="Comparison chart:",
            hovertemplate="Year: %{x}<br>" +
                            "Users with one rating amount: %{y:.0f}/100" +
                            "<extra></extra>"
        )

    trace_scatter_google_trends = go.Scatter(
            x=google_trends_fake_rating_years_reduced_to_hundered['year'],
            y=google_trends_fake_rating_years_reduced_to_hundered['rating_amount'],
            line=dict(color=colors[8], width=2), name='Google trends<br>traffic', legendgroup='1',
            hovertemplate="Year: %{x}<br>" +
                            "Search traffic: %{y:.0f}/100"+
                            "<extra></extra>"
        )

    figure.add_traces([trace_bar_trends, trace_scatter_google_trends], 1, 1)

    figure.add_traces(go.Bar(x=growth_user_single_ratings_compared_all_ratings['year'],
                            y=growth_user_single_ratings_compared_all_ratings['growth'],
                            marker_color=colors[5], name='Growth in<br> percentage',
                            legendgroup='2', legendgrouptitle_text="Growth chart:",
                            hovertemplate="Year: %{x}<br>" +
                            "Growth in relation: %{y:.0f}%"+
                            "<extra></extra>"), 1, 2)

    figure.add_hline(y=20, line_dash="dot", row=1, col=1, line=dict(color='black'))
    figure.add_hline(y=40, line_dash="dot", row=1, col=1, line=dict(color='black'))
    figure.add_hline(y=60, line_dash="dot", row=1, col=1, line=dict(color='black'))
    figure.add_hline(y=80, line_dash="dot", row=1, col=1, line=dict(color='black'))

    figure.add_hline(y=-150, line_dash="dot", row=1, col=2, line=dict(color='black', width=1))
    figure.add_hline(y=-10, line_dash="dot", row=1, col=2, line=dict(color='black', width=1))
    figure.add_hline(y=0, row=1, col=2, line=dict(color='black', width=1))
    figure.add_hline(y=10, line_dash="dot", row=1, col=2, line=dict(color='black', width=1))
    figure.add_hline(y=20, line_dash="dot", row=1, col=2, line=dict(color='black', width=1))
    figure.add_hline(y=30, line_dash="dot", row=1, col=2, line=dict(color='black', width=1))

    figure.update_layout(legend=dict(x=0.45, y=1))
    figure.update_layout(paper_bgcolor='#FFFFFF')

    figure.update_layout(
        legend_tracegroupgap = 20
    )

    figure.update_layout(plot_bgcolor="#FFFFFF")

    figure.update_layout(yaxis=dict(tickformat=''))
    figure.update_yaxes(ticksuffix='%')

    figure.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 2011,
            dtick = 1
        ),
        xaxis2 = dict(
            tickmode = 'linear',
            tick0 = 2011,
            dtick = 1
        ),
        yaxis2 = dict(
            tickmode = 'array',
            tickvals = [-150, -10, 0, 10, 20, 30]
    ),
    height=700)

    figure.update_layout(
        yaxis = dict(
            tickmode = 'array',
            tickvals = [20, 40, 60, 80, 100],
            ticktext = [20, 40, 60, 80, 100]
        )
    )
    return figure

### Many Ratings Part ###

def ratings_for_uid(user_id):
    all_user_ratings = reduced_ratings.query(f'userId == {user_id}').copy()
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


def plot_strip_scatter(uid,time_range):
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
    
    fgo.add_trace(go.Heatmap(
    z=[np.arange(0.0,5.5,0.5)],
    colorscale=[
        [0, "#0f0787"],
        [0.1, "#0f0787"],

        [0.1, "#5011a4"],
        [0.2, "#5011a4"],

        [0.2, "#790eac"],
        [0.3, "#790eac"],

        [0.3, "#a833aa"],
        [0.4, "#a833aa"],

        [0.4, "#be3e8a"],
        [0.5, "#be3e8a"],

        [0.5, "#d8576b"],
        [0.6, "#d8576b"],

        [0.6, "#ed7953"],
        [0.7, "#ed7953"],

        [0.7, "#fba342"],
        [0.8, "#fba342"],

        [0.8, "#fdcb2d"],
        [0.9, "#fdcb2d"],

        [0.9, "#f1f421"],
        [1.0, "#f1f421"]

    ],
    
    colorbar=dict(
        ticks="outside",
        ticktext= [str(x) for x in np.arange(0.5,5.5,0.5)],
        tickvals=np.arange(0.25,5.25,0.5),
    )  
    ))
    

    fgo.update_layout(
        plot_bgcolor= 'rgb(244,247,251)',
        yaxis = dict(
            tickmode = 'array',
            tickvals = np.arange(0.5,5.5,0.5)
        ),
        title_text=f'Movie Ratings of User <b>{uid}</b> ({len(all_user_ratings)} Ratings)',
        coloraxis_showscale=False,
        xaxis_range=time_range,
    )

    fgo.update_xaxes(title_text="Time", titlefont_size=12, row = 1, col = 1)
    fgo.update_yaxes(title_text="Rating", titlefont_size=12, row = 1, col = 1)
    fgo.update_xaxes(title_text="Count per Rating Level", titlefont_size=12, row = 1, col = 2)

    fgo.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='lightgrey'
    )
    fgo.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='lightgrey', range=[0,5.5]
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
    html.Div(id='center_content', children=[
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
            Part 1: Users with one rating theory
        '''),

        html.P([
            'At first, we thought about our personal experience on social media. A lot of suspicious account on sites like Instagram have low activities. ' + 
            'We thought it has to do with the detection of suspicious actives. If you have a lot of activity, it is easier to find patterns in the usage. ' +
            'Our guess was that the hosts of the websites already monitor the activity of each user and ban those suspicious users themselves.', html.Br(), html.Br(),
            'With that logic, we created our first theory. Our theory was: "Users which rated only one movie are manly bots".', html.Br(), html.Br(),
            'To prove this theory, our goal was to analyzed all the data of users with one rating.', html.Br(),
            'At first, we filtered out all the ratings, which came from users with only one rating. We were surprised by the result. Of the over 27 Million ratings in total, ' +
            'only 5\'620 of ratings came from users which only left one rating. This means in the worst case scenario, only 0.02% of the ratings on movieLens could potentially be ' +
            'from bot activity after our theory.', html.Br(),
            'This result already made us skeptic and there were two options on why there were so few ratings from users with only one rating. Firstly, our theory turns out to be true, ' +
            'and the site could have a low amount of bots. This would mean that the bot activity on the site movieLens doesn\'t have a profound impact on the ratings, ' +
            'which would be a great conclusion. Secondly, it could mean that our theory was completely wrong and there is no correlation between accounts with only one rating and bot activity.', html.Br(),
            'So the next step to prove or disprove our theory, we looked deeper into the data and compared the rating scores of the ratings from the users with one rating ' +
            'to the users with multiple ratings. We expected to find more ratings at both ends of the spectrum, and fewer ratings in the middle of the spectrum for ratings ' +
            'from users with only one rating. The reason behind that logic would be, that the incentive to create botted ratings would be to push the movie you have some ' +
            'kind of relation to up for your personal gain or to push other competing movies down to look better in comparison.', html.Br(),
            'To compare those two metrics, we have decided to make a horizontal stacked bar chart for each class. On the x-axis there are the percentage of each rating group ' +
            'and on the y axes are the two categories "user with only one rating" and "user with multiple ratings".  This gives us an interesting insight into the data.', html.Br()
        ]),


        html.Div(id='div_horizontal_barchart', className='viz', children=[
            dcc.Graph(figure=build_horizontal_stacked_barchart(),
            id='horizontal_barchart')
        ]),

        html.P([
           'We can clearly see that there is a strong deviation for the score 5.0 and 0.5. The probability of a 5.0 rating is more than double for the users with only one rating, ' +
           'then there is the probability of 5.0 rating for the users with multiple ratings. On the other end of the spectrum, the 0.5 ratings are less conclusive. ' +
           'The probability is nearly double for the ratings from users with one rating compared to the ratings from users with multiple ratings, ' +
           'but the probability for 0.5 ratings are in both cases low. But in both cases it is important to see that the dataset of the ratings from users with one rating ' +
           'was only 0.2% compared to users with multiple ratings, which are the other 99.8%. Because of that difference, the deviation in the ratings from users with only one account ' +
           'is not significant.', html.Br(), html.Br(),
           'The previous graph definitively gave us interesting insights, but it wasn\'t conclusive enough to neither prove nor disprove our theory. Because of that, ' +
           'we took further investigation into the timeline of when those ratings were submitted. As we thought, botted ratings and manipulation is a more recent activity that maybe started ' +
           '5 to 10 years ago. To get prove about our thought, we searched for  proof.  We looked into google trends, which collects data about how many searches in Google have the given search term. ' +
           'We have tried different terms and got the best data with the search term "fake rating". Obviously, this term is wide and is not limited to the website movieLens ' +
           'and is neither limited to movie ratings, but it was the narrowest term that had enough data to show a trend. The data from Google trends started at 2004, ' +
           'but the values before August 2010 are not conclusive enough because there are a lot of short term spikes in single months where the month before and after goes to 0. ' +
           'We decided to show the trend in a bar chart grouped by years, so we took the year from 2011 to 2017 which is the last year the movieLens data ranges from the first ' +
           'day to the last of the year.', html.Br(),
           'Grouping it by years, mitigates the variations which it has from the small size of the dataset.', html.Br(), html.Br(),
           'In addition to the Google trends data, we looked into the growth of the ratings from users with only one rating. To reduce the complexity of the graph, we decided to plot ' +
           'the difference between the yearly growth in percentage of ratings from users with only one rating, subtracted by the yearly growth of all ratings.', html.Br(), html.Br()
        ]),

        html.Div(id='div_columnLineTimlineChart_and_BarChart', className='viz', children=[
            dcc.Graph(figure=build_columnLineTimlineChart_and_BarChart(),
            id='columnLineTimlineChart_and_BarChart')
        ]),

        html.P([
            'Looking at the first graph that compares the amount of ratings from users with only one rating, there is a clear correlation between the rise in 2015 of search traffic for ' +
            'the term "fake rating" and the growth of those ratings. These gave us a lot of hope, that we are on something. But the second graph that compares the growth of those ratings ' +
            'to all ratings tells a complete other story. 2015 was by far the worst year in growth of those ratings compared to all ratings. The ratings of users with only one rating, ' +
            'like shown on the left graph, grew that year, but all ratings outgrew them by over 150%. This means that the ration of all ratings to ratings from users with only rating ' +
            'declined in 2015 and 2016.', html.Br(), html.Br(),
            'To conclude our research, the theory: "Users which rated only one movie are manly bots" was disproven. We have looked into all metrics that were given to us and there ' +
            'were no significant deviation on patterns in the data, that proved our theory. The score was on average better but not in a significant way and all ratings from users ' +
            'with only one rating was only made 0.02% of all ratings. Furthermore, there was no significant growth over the year and the only outlier in our analysis was a decline of growth ' +
            'of those ratings compared to the growth of all ratings in 2015.', html.Br(),
            'This means that there is no indication, that users with only one rating on the site movieLens are non humanly generated activity.'
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

        html.P([

    'Our hypothesis for \'busy users\' is, that (1) Rating Bursts and (2) Unnatural Distributions are the main indicators for bot activity. We searched for a user to showcase this pattern. This example shows a data burst of user 172357 with a duration of about 2 hours with the corresponding histogram. The burst consists of 1005 ratings.',html.Br(),html.Br(),

        ]),

        html.Div(id='div_plot_indicators', className='viz_part2', children=[
            dcc.Graph(figure=plot_indicators(),
            id='plot_indicators')
        ]),

        html.P([

    'In the case of (1) Rating Bursts, they are easy to identify. We define Rating Bursts as unusually high amounts of activity in contained timeframes, ranging from seconds to hours with distinct intervals with no activity. The distribution of rating scores is irrelevant to this criteria.',html.Br(),html.Br(),

    'Despite our initial assumption that (2) Unnatural Distribution would be fairly easy to identify, we found that this may not be the case. There is no consensus about what "unnatural" means. We presumed that natural distributions would come in form of a normal curve. However, this is only an assumption, and we are biased by our own rating behavior. In the case of user 172357, the distribution could be legitimate and indicate that the user tends to rate movies in a polarized way.',html.Br(),html.Br(),

        ]),

        html.H3(children='''
            User 134596
        '''),

        html.P([

    'While we were plotting Rating Bursts for the most active users, we stumbled upon a pattern that differed from the rest. User 134596 was intriguing at first sight. Despite his high number of Ratings, his activity seems legitimate at first glance.',html.Br(),html.Br(),

        ]),


        html.Div(id='div_plot_strip_scatter_134596', className='viz_part2', children=[
            dcc.Graph(figure=plot_strip_scatter(134596,['2009-01-01','2019-01-01']),
            id='plot_strip_scatter_134596')
        ]),

        html.P([
    'There are no rating bursts and continuous activity over almost 10 years. The rating distribution does not show any signs of polarization.',html.Br(),html.Br(),

    'For comparison we plotted the same chart for the most active user (yes, the one with 23k Ratings) to demonstrate the differences. Note, that we used a different time period here, as the user only started his rating activity late 2015.',html.Br(),html.Br(),
    
        ]),

        html.Div(id='div_plot_strip_scatter_123100', className='viz_part2', children=[
            dcc.Graph(figure=plot_strip_scatter(123100,['2015-07-01','2019-01-01']),
            id='plot_strip_scatter_123100')
        ]),

        html.P([
    'The Rating Distribution of User 123100 gives the impression of a very even, unnatural distribution. However, the histogram shows clearly, that this is not the case. This confirms our findings in the previous chart about bot activity indicators.',html.Br(),html.Br(),

    'But now back to the infamous User 134596. The previous visualization left us with more questions than answers. There are no indications to justify doubt in the legitimacy of this user. It feels like finding the needle in the haystack. Maybe there is no needle and User 134596 is in fact human. But we are not done yet.',html.Br(),html.Br(),

    'Every Rating has a timestamp. With thousands of ratings, this metadata can reveal a lot of insight into the user\'s life. In the original dataset, the timestamps are in Unix time. Even if the server moved to a different time zone, the data would still be consistent.',html.Br(),html.Br(),
    
    'We created a histogram of the timestamps. As expected, there is no anomaly in minutes and seconds. Ratings are evenly distributed over minutes and seconds. However, the hours tell a different story. There is a distinct pattern of activity over the day. There are even some resting hours with no activity at all. We decided to compare this histogram over years. There was no need for normalization since all selected years had similar value ranges. There is still a very visible correlation between favorite rating hours over years.', html.Br(),html.Br(),
        ]),

        html.Div(id='div_plot_freq_polygon', className='viz_part2', children=[
            dcc.Graph(figure=plot_freq_polygon(134596,2011,2013),
            id='plot_freq_polygon')
        ]),

        html.P([
    'The consistency of hours without any ratings is remarkable. If User 134596 is a human, we can only applaud this disciplined sleep schedule. The decrease in activity from 3UTC to 11UTC could indicate a workday, however, this is pure speculation.',html.Br(),html.Br(),

    'In Conclusion, we found no evidence to justify doubt in the legitimacy of User 134596. If this User actually turns out to be a bot, we can only admire the creators dedication and effort to run this account for over ten years. Of course, there are still more sophisticated ways to detect irregularities we haven\'t covered yet. In summary, our approach worked well to identify anomalies for most busy user accounts.',html.Br(),html.Br(),
    
    'Most of the users with the most ratings showed clear signs of bot activity in the form of Rating Bursts very similar to the one shown for User 123100. With hundreds of ratings per minute, it is safe to assume that they weren\'t manually entered by a human. Despite our initial assumption that the histogram would be a good indicator of bot activity, we found that this is not the case.',html.Br(),html.Br(),

        ]),

        html.H2(children='''
            Main Conclusion
        '''),

        html.Div(className='citation', children=[
            html.P([
            '[1] Average Movie Length - https://towardsdatascience.com/are-new-movies-longer-than-they-were-10hh20-50-year-ago-a35356b2ca5b',html.Br(),html.Br(), 
        ]),
        ]),


    ])
]


app.layout = html.Div(children=html_structure)

if __name__ == '__main__':
    app.run_server(debug=True)