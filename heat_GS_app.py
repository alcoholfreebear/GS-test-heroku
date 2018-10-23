#######
# Side-by-side heatmaps for Sitka, Alaska,
# Santa Barbara, California and Yuma, Arizona
# using a shared temperature scale.
######
# import plotly.offline as pyo
# import plotly.graph_objs as go
# from plotly import tools
# import pandas as pd


import os, sys


# plotly up ended
# ------------------------------------------------
# dash part starts
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64


import matplotlib.pyplot as plt
import networkx as nx

foldername = 'randomtest_20180915'
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = CURR_PATH + os.sep + foldername
FIGURES = CURR_PATH + os.sep + foldername
PROCESSED = CURR_PATH + os.sep + foldername
df = pd.read_pickle(DATA_RAW + os.sep + 'reg2rand.pickle')[[0]]
# df0 = pd.read_pickle(DATA_RAW + os.sep + 'reg.pickle')[[0]]
# df_res = pd.read_pickle(DATA_RAW + os.sep + 'residual_log.pkl')[[0]]

def clean_df(dfin):
    df = dfin.copy()
    df.index.names = ['game1','game2']
    df = df.reset_index()
    df.columns = ['game1', 'game2', 'values']
    df.loc[(df['game1']==df['game2']), 'values'] = 1e-8
    return df

def make_dict_df(dfin):
    df_all, df_pro, df_anti = dfin.copy(),dfin.copy(),dfin.copy()
    zweight = np.log(dfin['values'])
    df_all['values'] = zweight
    df_pro['values'] = np.where(zweight>0,zweight,0)
    df_anti['values'] = np.where(zweight<0,-zweight,0)
    return {'all': df_all, 'pro':df_pro, 'anti':df_anti}

def make_graph(dfin):
    G = nx.Graph()
    edgetuples = zip(dfin['game1'],dfin['game2'], np.exp(dfin['values']))
    G.add_weighted_edges_from(edgetuples)
    return {'graph':G, 'weight': np.exp(dfin['values'])}


def make_dict_graph(dict_df):
    dict_G_all = make_graph(dict_df['all'])
    dict_G_pro = make_graph(dict_df['pro'])
    dict_G_anti = make_graph(dict_df['anti'])
    return {'all': dict_G_all, 'pro':dict_G_pro, 'anti': dict_G_anti}


df = clean_df(df)
dict_df = make_dict_df(df)
dict_graph = make_dict_graph(dict_df)
# print(dict_df.keys())
# print(dict_graph)
del df

app = dash.Dash()

# df = pd.read_csv('../data/wheels.csv')
#
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())


plotparams = {
    'text': '#000000',
    'background': 'white',
    'nticks': len(dict_df['all']['game1'].unique()),
    'percentiles': {}
}

app.layout = html.Div([
    # radio buttons
    html.Div([
        dcc.RadioItems(
            id = 'radio-buttons',
            options=[
            {'label': 'Over All', 'value': 'all'},
            {'label': 'Pro Similarity', 'value': 'pro'},
            {'label': 'Anti Similarity', 'value': 'anti'}
                ],
        value='all'
            )], style={'paddingLeft': 200, 'color': plotparams['text']}),


    # Game similarity matrix
    html.Div([

    dcc.Graph(
        id='GS-heatmap',
        figure={
            'data': [
                go.Heatmap(
                    x=dict_df['all']['game1'],
                    y=dict_df['all']['game2'],
                    z=dict_df['all']['values'],
                    colorscale='Viridis',
                    zmin = -3, zmax = 3,  # add max/min color values to make each plot consistent
                    colorbar=dict(len=1, x = 1, y=0.5)
                )
            ],
            'layout': go.Layout(
                plot_bgcolor=plotparams['background'],
                paper_bgcolor=plotparams['background'],
                font = {
                    'color': plotparams['text']
                    },
                title = 'Game Similarity log(signal_to_noise_ratio)',
                xaxis = {'title': '','nticks': plotparams['nticks']},
                yaxis = {'title': '','nticks': plotparams['nticks']},
                hovermode='closest',
                height = 1000, width = 1050,
                margin = dict(l=250, r=100, t=100, b=250),

)
        }
    ),

    ], style={'width':'60%', 'height':'100%', 'float':'left', 'backgroundColor': plotparams['background']}),


    html.Div([
        html.Div([
            html.H1('Chose threshold:'),
            dcc.Slider(
                id='threshold-slider',
                min=0,
                max=100,
                step=5,
                #marks={i: str(i) for i in np.arange(-3, 4)},
                value=50,
            )

        ], style={'backgroundColor': plotparams['background'],
              'color': plotparams['text'],
                  'fontSize': 12,
                #'width':'100%',
                  }),

    html.H1(id='hover-text')
    ], style={'paddingTop':25,
                'paddingLeft':1100,
              'backgroundColor': plotparams['background'],
              'color': plotparams['text'],
              'fontSize': 10
              }),

    html.Div([
        html.Img(id='hover-image', height=500)
        # html.H1(id='hover-image')

    ], style={'paddingTop': 200,  'paddingLeft':1200,'paddingBottom': 200,
              'backgroundColor': plotparams['background'],
               'float':'bottom','color': plotparams['text']})

],style = {'backgroundColor': plotparams['background']})

@app.callback(
    Output('hover-text', 'children'),
    [Input('GS-heatmap', 'clickData'),
    Input('threshold-slider', 'value')],
    [State('radio-buttons', 'value')]
)
def callback_gamepairs(clickData, threshold, radiochoice):
    y=clickData['points'][0]['y']
    x=clickData['points'][0]['x']
    perc_val = np.percentile(dict_df[radiochoice]['values'].values.ravel(), float(threshold))
    return 'Game: {} \n , weight threshold: above {}% percentile'.format(y, (threshold))


@app.callback(
    Output('hover-image', 'src'),
    [Input('GS-heatmap', 'clickData'),
    Input('threshold-slider', 'value')],
    [State('radio-buttons', 'value')])
def callback_shownetwork(clickData, threshold, radiochoice):
    options ={
        'all': {'opt':'all','color':'g'},
        'pro': {'opt':'all','color':'g'},
        'anti': {'opt':'anti','color':'r'},
    }
    y=clickData['points'][0]['y']
    x=clickData['points'][0]['x']
    perc_val = np.percentile(dict_graph[options[radiochoice]['opt']]['weight'], float(threshold))
    new_g1 = nx.Graph([(u,v,d) for (u,v,d) in dict_graph[options[radiochoice]['opt']]['graph'].edges(data= True) if ((u==y)or (v==y))&(d['weight']>=(perc_val))])
    components = list(sorted(nx.connected_component_subgraphs(new_g1), key = len, reverse = True))

    f, ax = plt.subplots(1,1, figsize=(7, 5))
    plt.title(radiochoice + ' similarity network of '+y )

    if len(components)>=1:
        pos = nx.spring_layout(components[0])
        ax.patch.set_facecolor(plotparams['background'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_tick_params(color = plotparams['background'])
        ax.yaxis.set_tick_params(color=plotparams['background'])
        f.patch.set_visible(False)
        ax.axis('off')
        ax.grid(False)
        nx.draw_networkx(components[0],ax = ax, pos = pos,
                         with_labels = True, font_color = plotparams['text'],
                         edge_color = plotparams['text'],
                         font_size = 10,
                         edgelinewidth = 100,
                         node_size = 1600,
                         node_color = options[radiochoice]['color'])
        l, r = plt.xlim()
        plt.xlim(l - 0.6, r + 0.6)
    plt.tight_layout()
    figname = FIGURES + os.sep + 'network.png'
    plt.savefig(figname, dpi = 100)
    return encode_image(figname)



@app.callback(
    Output('GS-heatmap', 'figure'),
    [Input('radio-buttons', 'value')])
def figure_radiochoice(radiochoice):
    dict_choice ={
        'all': dict(zmin=-3, zmax=3),
        'pro': dict(zmin=0, zmax=3),
        'anti':dict(zmin=0, zmax=3)
    }

    figure = {
        'data': [
            go.Heatmap(
                x=dict_df[radiochoice]['game1'],
                y=dict_df[radiochoice]['game2'],
                z=dict_df[radiochoice]['values'],
                colorscale='Viridis',
                zmin=dict_choice[radiochoice]['zmin'], zmax=dict_choice[radiochoice]['zmax'],  # add max/min color values to make each plot consistent
                colorbar=dict(len=1, x=1, y=0.5)
            )
        ],
        'layout': go.Layout(
            plot_bgcolor=plotparams['background'],
            paper_bgcolor=plotparams['background'],
            font={
                'color': plotparams['text']
            },
            title='Game Similarity log(signal_to_noise_ratio)'+ radiochoice,
            xaxis={'title': '', 'nticks': plotparams['nticks']},
            yaxis={'title': '', 'nticks': plotparams['nticks']},
            hovermode='closest',
            height=1000, width=1050,
            margin=dict(l=250, r=100, t=100, b=250),

        )
    }
    return figure

if __name__ == '__main__':
    app.run_server()
