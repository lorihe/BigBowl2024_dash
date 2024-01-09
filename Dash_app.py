import numpy as np
import pandas as pd
import pickle
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output
import json
import plotly.graph_objects as go

import voronoi_mod
import get_data
import tackle_plot

app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA],
          meta_tags=[{"name": "viewport", "content": "width=device-width,"
                      "initial-scale=1, maximum-scale=1"}])

app.title = "Big Data Bowl 2024"

server = app.server
app.config.suppress_callback_exceptions = True

with open('newdata/dash_list.json', 'r') as file:
    dash_list = json.load(file)
with open('gb_tree_1.pkl', 'rb') as file:
    gb_model = pickle.load(file)
ytg_probs_df = pd.read_csv('newdata/ytg_probs_df.csv')
X = ytg_probs_df[['near_defender_MPD', 'near_dist', 'SoF', 'has_blocker']]
ytg_probs_df['sf'] = gb_model.predict_survival_function(X)


def standardize_tracking(tracking):
    """
    Standardize data so all plays are seen as played from left to right

    Parameters:
    df1 (DataFrame): tracking data of one selected play.
    Returns:
    df1 (DataFrame): standardized tracking data of one selected play.
    """
    tracking['x'] = 120-tracking['x']
    tracking['y'] = 53.3-tracking['y']
    tracking['dir'] = (tracking['dir'] + 180) % 360
    tracking['o'] = (tracking['o'] + 180) % 360
    return tracking

def description_card():
    '''
    :return: An HTML Div element introducing the app.
    :rtype: dash_html_components.Div
    '''
    return html.Div(
        id="description",
        children=[
            html.H5("Big Data Bowl 2024",
                    className="text-dark",
                    style={"font-size": "24px", "font-weight": "bold", "margin-left": "40px"}),

            html.P("This Dashboard is part of the submission to NFL Big Data Bowl 2024 competition "
                   "which focuses on 'tackling'. A gradient boosted survival analysis model was trained on the tracking "
                   "data of short pass plays in NFL 2022 season to predict probability of down after caught. "
                   "Detailed explanation can be found in the Kaggle Notebook:",
                   style={"margin-left": "40px", "margin-top": "10px"}),

            dbc.NavLink("Survival Analysis on Progress after Caught",
                        href="https://www.kaggle.com/code/lorihe/survival-analysis-on-progress-after-caught",
                        style={'margin-left': '40px', 'margin-top': '5px', 'color': 'steelblue', "font-weight": "bold",})
        ],
        style={"margin-bottom": "20px", }
    )

def play_select_card():
    '''
    :return: An HTML Div element providing select menu for play selection.
    :rtype: dash_html_components.Div
    '''
    return html.Div([

        dbc.Select(
            id='date-select', value = '10/09/2022',
            options=[{'label': date, 'value': date} for date in dash_list],
        ),
        dbc.Select(
            id='game-select', value = '2022100908'
        ),
        dbc.Select(
            id='play-select',
        ),
    ])


app.layout = dbc.Container(
    fluid=True,
    children=[
        # First row is the banner
        dbc.Row(
            html.Div([
                html.Div(id="banner1", className="banner",
                         children=[
                             html.Img(src=app.get_asset_url("github.jpg"),
                                      style={"height": "24px", "margin-top": "3px", 'margin-left': '73px'})
                         ]),
                html.Div(id="banner2", className="banner",
                         children=[
                             dbc.NavLink("by Lori He",
                                         href="https://github.com/lorihe/BigBowl2024",
                                     style={'margin-left': '105px', 'margin-top': '-25px', 'color': 'honeydew'})
                         ]),
            ]),
            style={"height": "30px", "background-color": "black", 'margin-bottom': '10px',
                   "z-index": "2",},
        ),

        dbc.Row([
            # First column is app description and match selection menu
            dbc.Col(
                html.Div(
                    children=[
                        html.Div(description_card(), style={'margin-right': '0px',}),
                        html.Div(play_select_card(), style={"width": "60%", 'margin-left': '80px', 'margin-top': '40px',}),
                    ], style={'margin-left': '0px', 'margin-right': '-10px',
                              'margin-top': '40px', 'margin-bottom': '0px'}
                ), xs={'size': 12}, sm={'size': 12}, md={'size': 12},
                lg={'size': 2}, xl={'size': 2},
            ),

            dbc.Col(
                html.Div(children = [
                    html.P(id = 'Id-text',
                           style={'margin-top': '10px','margin-left': '50px', 'text-align': 'center',
                                  'font-family': 'Roboto, sans-serif', 'color': 'grey',}),
                    dcc.Graph(id = 'plot_frames_legend',
                              config={'displayModeBar': False}),
                    dcc.Graph(id='plot_frames',)
                    ], style = {'margin-left': '0px', 'margin-right': '0px'}
                ),  xs={'size': 12}, sm={'size': 12}, md={'size': 12},
                lg={'size': 4}, xl={'size': 4},
            ),

            dbc.Col(
                children=[
                    html.Div(
                        children=[
                            html.P(id='SoF-lr-text',
                                   style={'margin-top': '30px', 'margin-left': '40px', 'text-align': 'center',
                                          'font-family': 'Roboto, sans-serif', 'color': 'dimgrey', }),
                            html.P(id='SoF-text',
                                   style={'margin-top': '0px', 'margin-left': '50px', 'text-align': 'center',
                                          'font-family': 'Roboto, sans-serif', 'font-size': 18, }),
                            dcc.Graph(id='plot-voronoi', config={'displayModeBar': False, 'doubleClick': False},
                                      style={'margin-top': '-10px', 'margin-left': '0px'}, )
                        ], style={'display': 'inline-block', 'vertical-align': 'top', 'width': '500px'},

                    ),
                    html.Div(
                        children=[
                            html.P(
                                "SoF (sum of freedom): A measure of area to quantify how free the ball carrier "
                                "can move under the defense team's pressure at caught moment, calculated with Voronoi statistics.",
                                style={'margin-top': '10px', 'margin-left': '60px', 'margin-right': '0px',}, ),
                            html.P("MPD (minimum possible distance): The minimum achievable distance between a defender and "
                                "a carrier, calculated based on their current positions and velocities at caught moment.",
                                   style = {'margin-top': '10px', 'margin-left': '60px', 'margin-right': '0px',},),
                            html.P('Near Defenders MPD (yards) at caught',
                                   style={'margin-top': '20px', 'margin-left': '20px', 'text-align': 'center',
                                          'font-family': 'Roboto, sans-serif', 'font-size': 18}),
                            dcc.Graph(id='plot-MPD', config={'displayModeBar': False, 'doubleClick': False},
                                      style={'margin-top': '20px', 'margin-left': '0px', 'margin-right': '15px'})
                        ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-top': '30px',
                                  'margin-left': '0px', 'margin-right': '0px', 'width': '520px'},

                    ),
                    html.Hr(style={'margin-top': '2px', 'margin-bottom': '16px', 'border-width': '2px',
                                   'margin-left': '60px', 'margin-right': '10px', 'color': 'grey', "display": "flex"}),
                    html.Div(
                        children=[
                            html.P('Survival Curve after Caught',
                                   style={'margin-top': '0px', 'margin-left': '0px', 'text-align': 'center',
                                          'font-family': 'Roboto, sans-serif', 'color': 'black', 'font-size': 18}),
                            html.P(id='probs-text',
                                   style={'margin-top': '0px', 'margin-left': '0px', 'text-align': 'center',
                                          'font-family': 'Roboto, sans-serif', 'color': 'dimgrey'}),
                            dcc.Graph(id='plot-probs', config={'displayModeBar': False},
                                      style={'margin-top': '0px', 'margin-left': '130px'})
                        ], style={'margin-top': '0px', 'margin-left': '0px', 'width': '90%'}
                    )
                ],
                xs={'size': 12}, sm={'size': 12}, md={'size': 12},
                lg={'size': 6}, xl={'size': 6}
            )

        ])

    ]
)

@app.callback(
    Output("game-select", "options"),
    [Input("date-select", "value")]
)
def get_games(date):

    return [{'label': i, 'value': i} for i in dash_list[date]]

@app.callback(
    Output("game-select", "value"),
    [Input("date-select", "value")]
)
def reset_game_select(date):

    return list(dash_list[date].keys())[0]

@app.callback(
    Output("play-select", "options"),
    [Input("date-select", "value"),
     Input("game-select", "value")]
)
def get_plays(date, game):

    return [{'label': i, 'value': i} for i in dash_list[date][game]]

@app.callback(
    Output("play-select", "value"),
    [Input("game-select", "value")]
)
def reset_play_select(game):
    return None

@app.callback(
    Output("plot_frames_legend", "figure"),
    Output("plot_frames", "figure"),
    Output("Id-text", "children"),
    [Input("date-select", "value"),
    Input("game-select", "value"),
    Input("play-select", "value"), ]
)

def plot_frames(date, gameId, playId):

    if not playId:
        playId = dash_list[date][gameId][0]
    gameId = int(gameId)
    playId = int(playId)

    game = get_data.game(gameId)
    play = get_data.play(gameId, playId)
    tracking = get_data.tracking(gameId, playId).reset_index()

    date = game['gameDate'].item()
    tacklers = get_data.tacklers(gameId, playId)
    carrier = play['ballCarrierId'].item()
    offense = play['possessionTeam'].item()
    defense = play['defensiveTeam'].item()
    down = play['down'].item()
    yardstogo = play['yardsToGo'].item()
    ablos = play['absoluteYardlineNumber'].item()

    fig0, fig = tackle_plot.plot(tracking, date, tacklers, carrier, offense, defense, down, yardstogo, ablos, 850, 760)

    id_text = f'gameId: {gameId} | playId: {playId}'
    return fig0, fig, id_text

@app.callback(
    Output("SoF-lr-text", "children"),
    Output("SoF-text", "children"),
    Output("plot-voronoi", "figure"),
    Output("plot-MPD", "figure"),
    [Input("date-select", "value"),
    Input("game-select", "value"),
    Input("play-select", "value")]
)

def get_vor_SoF(date, gameId, playId):

    if not playId:
        playId = dash_list[date][gameId][0]
    gameId = int(gameId)
    playId = int(playId)

    gameId = int(gameId)
    playId = int(playId)
    play = get_data.play(gameId, playId)
    carrier = play['ballCarrierId'].item()
    offense = play['possessionTeam'].item()
    defense = play['defensiveTeam'].item()

    tracking = get_data.tracking(gameId, playId)
    if tracking.at[0, 'playDirection'] == 'left':
        tracking = standardize_tracking(tracking)
    caught_frm = 6
    if gameId == 2022110608 and playId == 2351:
        caught_frm = 32

    caught_df = tracking[tracking['frameId'] == caught_frm]

    defense_caught_df = caught_df[caught_df['club'] == defense].reset_index(drop=True)
    defense_caught_xy = defense_caught_df[['x', 'y']].to_numpy()

    carrier_caught_df = caught_df[caught_df['nflId'] == carrier]
    carrier_caught_xy = carrier_caught_df[['x', 'y']].to_numpy()[0]

    # Set boundary points to avoid unbounded vertices. The points are either mirrored from carrier across
    # the sidelines or 40 yards from the carrier on both xy directions.
    boundary_points = np.array(
        [[carrier_caught_xy[0], 53.3+(53.3-carrier_caught_xy[1])],
         [carrier_caught_xy[0], 0-carrier_caught_xy[1]],
         [carrier_caught_xy[0], carrier_caught_xy[1]+40],
         [carrier_caught_xy[0], carrier_caught_xy[1]-40],
         [carrier_caught_xy[0] - 40, carrier_caught_xy[1]],
         [carrier_caught_xy[0] + 40, carrier_caught_xy[1]]] )
    points = np.vstack([defense_caught_xy, carrier_caught_xy, boundary_points])

    # Get neighbor defenders from all points
    points_nbs = voronoi_mod.get_neighbors(points, carrier_caught_xy)
    vertices = voronoi_mod.get_vertices(points_nbs, carrier_caught_xy)

    # Get areas and calculate SoF
    area_left, area_right, vertices_left, vertices_right = voronoi_mod.get_areas(vertices, carrier_caught_xy)
    log_area_left = np.log(area_left+1)
    SoF = log_area_left + area_right

    # Get offense players other than the carrier
    offense_caught_df = caught_df[caught_df['club'] == offense].reset_index(drop = True)
    offense_caught_xy = offense_caught_df[['x', 'y']].to_numpy()
    offense_caught_xy_n = np.array([p for i, p in enumerate(offense_caught_xy) if np.any(p != carrier_caught_xy)])

    # Get defender points which are not 5 yards behind carrier
    hull_points = np.array([p for i, p in enumerate(points_nbs) if
                  not np.any((p == boundary_points).all(axis = 1))
                  and p[0] > carrier_caught_xy[0] - 5])

    defense_nbs_xy = np.array([p for i, p in enumerate(points_nbs) if
                               not np.any((p == np.vstack([boundary_points, carrier_caught_xy])).all(axis=1))])

    SoF_lr_text = f'SoF_left: {round(area_left, 1)} | log(SoF_left+1): {round(log_area_left, 1)}, | SoF_right: {round(area_right, 1)}'
    SoF_text = f'SoF at caught: {round(SoF, 1)} yards\u00B2'

    defenders_MPD = {}
    for coord in defense_nbs_xy:
        x, y = coord
        matched = defense_caught_df[(defense_caught_df['x'] == x) & (defense_caught_df['y'] == y)]
        player = get_data.name(matched['nflId'].item())
        MPD = voronoi_mod.calculate_MPD(matched, carrier_caught_df)
        defenders_MPD[player] = MPD

    fig_MPD = voronoi_mod.plot_MPD(defenders_MPD)

    fig_vor = voronoi_mod.plot_vor(points_nbs, hull_points, carrier_caught_xy, boundary_points, vertices,
             vertices_left, vertices_right, offense_caught_xy_n)

    return SoF_lr_text, SoF_text, fig_vor, fig_MPD

@app.callback(
    Output("probs-text", "children"),
    Output("plot-probs", "figure"),
    [Input("date-select", "value"),
     Input("game-select", "value"),
     Input("play-select", "value")]
)
def plot_probs(date, gameId, playId):
    if not playId:
        playId = dash_list[date][gameId][0]
    gameId = int(gameId)
    playId = int(playId)

    play_data = ytg_probs_df[(ytg_probs_df['gameId'] == gameId) & (ytg_probs_df['playId'] == playId)]

    ytg_caught = play_data['ytg_caughts'].item()
    ytg_prob = play_data['ytg_probs'].item()
    survival_function = play_data['sf'].item()

    probs_text = f'yards to go after caught: {round(ytg_caught, 1)} | probability for down: {round(ytg_prob, 1)}'

    xlim_l = ytg_caught if ytg_caught < 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = survival_function.x, y= survival_function.y, mode='lines', name='Survival curve',
                             line=dict(color='darkblue', width=1)))
    fig.add_vline(x = ytg_caught, line_width=2, line_dash="dash", line_color="peru")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='yards to go after caught',
                             line=dict(color='peru', width=2, dash='dash')))

    fig.update_layout(yaxis_title='probability',
                      xaxis=dict(tickfont=dict(size=12), dtick=2, title='progress', range=[xlim_l, 40]),
                      yaxis=dict(ticks="outside",  ticklen=5, title='probability'),
                      width=900, height=400, showlegend=True, plot_bgcolor='whitesmoke',
                      margin=dict(t=2, b=0, l=0, r=0),
                      legend=dict(x=1, y=1, xanchor='right', yanchor='top', font = dict(size=12)))

    return probs_text, fig

if __name__ == '__main__':
    app.run_server(debug=True, port=1030)