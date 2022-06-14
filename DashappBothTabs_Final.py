# Standard
#from distutils.log import debug
import os
import pandas as pd
import numpy as np

#Local modules
import localmodules.conversor as c

# Dash components
import dash
from dash import html,dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# For plotting risk indicator and for creating waterfall plot
import lime.lime_tabular
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import shap

# To import joblib file model objects
import joblib


#Random Forest and Database Load
filew = './zDatabase/XAI - Limpo_dummified_smote.csv'
file = './zDatabase/XAI - Limpo_dummified_smote_minmax.csv'
fileTrain = './zDatabase/XAI - Limpo_dummified_smote_minmax_train.csv'
rfb = joblib.load('./zDatabase/randomforestsAPP.joblib')
dataw = pd.read_csv(filew,index_col='surgycal margin',na_values='',sep=',', decimal='.')
data = pd.read_csv(file,index_col='surgycal margin',na_values='',sep=',', decimal='.') 
train: pd.DataFrame = pd.read_csv(fileTrain)
trnY: np.ndarray = train.pop('surgycal margin').values
trnX: np.ndarray = train.values
labels = pd.unique(trnY)
labels.sort()

col = ['Age.at.MRI','Prostate.volume','PSA.value.at.MRI','Index.lesion.size',
       'Capsular.contact.lenght_TLC','Smooth.capsular.bulging','Capsular.disruption','Unsharp.margin',
       'Irregular.contour','Black.estrition.periprostatic.fat','Retoprostatic.angle.obliteration',
       'Measurable.ECE','ECE.in.prostatectomy.specimen_gold.standard','Gleason.score','regra',
       'Index.lesion.PIRADS.V2_3','Index.lesion.PIRADS.V2_4','Index.lesion.PIRADS.V2_5'] #'surgycal margin'

# Start Dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions = True)

# Layout
app.layout = html.Div([
    html.Div([html.H2('Prostatic Cancer Surgical Margin Prediction Tool',
                      style={'marginLeft': 20, 'color': 'white'})],
             style={'borderBottom': 'thin black solid',
                    'backgroundColor': '#24a0ed',
                    'padding': '10px 5px'}),
    dbc.Row([
        dbc.Col([html.Div("Patient information",
                          style={'font-weight': 'bold', 'font-size': 20}),
            dbc.Row([html.Div("Patient demographics",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Patient Age at MRI(years): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='55',
                        id='Age'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([html.Div("Patient health",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Prostate volume (unit?): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='45',
                        id='Prostate_volume'
                    )
                ]), width={"size": 3}, style={'padding': '10px 10px'}),
                dbc.Col(html.Div([
                    html.Label('PSA value at MRI (ng/mL?): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='7.4',
                        id='PSA_value'
                    )
                ]), width={"size": 3}, style={'padding': '10px 10px'}),
                dbc.Col(html.Div([
                    html.Label('Lesion size (cm?): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='18',
                        id='lesion_size'
                    )
                ]), width={"size": 3}, style={'padding': '10px 10px'}),
                dbc.Col(html.Div([
                    html.Label('Lesion PIRADS.V2: '),
                    dcc.Dropdown(
                        options=[
                            {'label': '3', 'value': '3'},
                            {'label': '4', 'value': '4'},
                            {'label': '5', 'value': '5'}
                        ],
                        value='4',
                        id='PIRADS_V2'
                    )
                ]), width={"size": 3}, style={'padding': '10px 10px'}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Capsular contact length (units?): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='12',
                        id='Capsular_contact_lenght'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Smooth capsular bulging: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},#label?
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='Smooth_capsular_bulging'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Capsular disruption: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},#label?
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='Capsular_disruption'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Unsharp margin: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},#label?
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='Unsharp_margin'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Irregular contour: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},#label?
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='Irregular_contour'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Black estrition periprostatic fat: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},#label?
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='Black_estrition_periprostatic_fat'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Retoprostatic angle obliteration: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},#label?
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='Retoprostatic_angle_obliteration'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Gleason score: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Below (3+4)', 'value': '0'},
                            {'label': 'Above (4+3)', 'value': '1'}
                        ],
                        value='0',
                        id='Gleason_score'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([html.Div("ECG results",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Measurable ECE: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='Measurable_ECE'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('ECE in prostatectomy: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='ECE_in_prostatectomy'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([html.Div("Surgery related metrics",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Rule: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Followed the rule', 'value': '1'},
                            {'label': 'Did not follow the rule', 'value': '0'}
                        ],
                        value='1',
                        id='regra'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
        ], style={'padding': '10px 25px'}
        ),

        # Right hand column containing the summary information for predicted surgical margin
        dbc.Col(html.Div([
        dcc.Tabs(id="tabs", value='tab-2', children=[
            dcc.Tab(label='LIME Explanation', value='tab-1'),
            dcc.Tab(label='SHAP Explanation', value='tab-2'),
            ], style={'height': '60px'}),
            html.Div(id='tabs-content')
        ]),
             ),
    ]),
    dbc.Row(
        html.Div([
                dbc.Button(
                    "Predictive model information",
                    id="collapse-button",
                    className="mb-3",
                    color="primary",
                ),
                dbc.Collapse(
                    dbc.Card(dbc.Row([ #Colapsable details
                        dbc.Col([
                            html.Div('Predictive model information',
                                     style={'font-weight': 'bold', 'font-size': 20, 'padding': '0px 0px 20px 0px'}),
                            html.Div('Data source',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(['A cohort of 121 patients at the Hospital da Luz were assessed on multiple '
                                      'characteristics and whether the surgical margin was followed was also recorded. '
                                      'In total 28% (n=34) of patients had surgical margin caracterized as 0.'],
                                     style={'font-size': 14, 'padding': '0px 0px 20px 0px'}),
                            html.Div('Model features and cohort summary',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(['The characteristics/features of the study cohort used to develop the predictive '
                                      'model supporting this tool are shown in Table 1 to the right.'],
                                     style={'font-size': 14, 'padding': '0px 0px 20px 0px'}),
                            html.Div('Model Training',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(['The data was split into a training set (70%, n=113) and a test set (30%, '
                                      'n=49). The final model '
                                      'achieved an average AUC of 1.0 in the training set and 0.86 in the test set'
                                      'Figure 2 to the far right indicates what the model identified as '
                                      'importance of predictors of the surgical margin. The more important features are '
                                      'towards the top of the figure, which includes characteristics like affected '
                                      'PSA value at MRI, Age at MRI, prostate volume and if smooth capsular bulging'
                                      'can be or not identified.'
                                      ],
                                     style={'font-size': 14, 'padding': '0px 0px 20px 0px'}),
                            
                        ]),
                        dbc.Col([
                            html.Div('Model Performance',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(['Figure 1 contains the model performance, according to the '
                                      '4 metrics used - accuracy, recall, specificity and precision - '
                                      'and the model confusion matrix. '
                                      'Figure 3 contains the ROC chart, from which we selected the threshold '
                                      'of classification: if a patient has more than 40% probability of '
                                      'having positive surgical margin (1), then they are classified as positive. '
                                      ],
                                      style={'font-size': 14, 'padding': '0px 0px 20px 0px'}),
                            html.Div('Table 1. Cohort Table',
                                     style={'font-weight': 'bold', 'font-size': 14, 'textAlign': 'middle'}),
                            html.Div(className='container',
                                children=[html.Img(src=app.get_asset_url('Cohort_table.png'),
                                                   style={'height': '100%', 'width': '100%'})])
                            ]),
                            
                        dbc.Col([
                            html.Div('Figure 1. Model Performance',
                                     style={'font-weight': 'bold', 'font-size': 14, 'textAlign': 'middle'}),
                            html.Div(className='container',
                                     children=[html.Img(src=app.get_asset_url('Finalmodel_Performance.png'),
                                                        style={'height': '90%', 'width': '90%'}),
                                                html.Img(src=app.get_asset_url('Finalmodel_ROC_chart.png'),
                                                        style={'height': '60%', 'width': '60%'})]),
                            html.Div('Figure 2. Feature importance',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(className='container',
                                     children=[html.Img(src=app.get_asset_url('Feature_Importance.png'),
                                                        style={'height': '90%', 'width': '90%'})]),
                            ])
                        ], style={'padding': '20px 20px'})),
                    id="collapse",
                ),
            ]),
        style={'padding': '10px 25px',
               'position': 'fixed',
               'bottom': '0'},
    ),
    html.Div(id='data_patient', style={'display': 'none'}),
    ]
)

# Responsive elements: toggle button for viewing model information
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],)

def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Responsive element: create X matrix for input to model estimation
@app.callback(
    Output('data_patient', 'children'),
    [Input('Age', 'value'),
     Input('Prostate_volume', 'value'),
     Input('PSA_value', 'value'),
     Input('lesion_size', 'value'),
     Input('PIRADS_V2', 'value'),
     Input('Capsular_contact_lenght', 'value'),
     Input('Smooth_capsular_bulging', 'value'),
     Input('Capsular_disruption', 'value'),
     Input('Unsharp_margin', 'value'),
     Input('Irregular_contour', 'value'),
     Input('Black_estrition_periprostatic_fat', 'value'),
     Input('Retoprostatic_angle_obliteration', 'value'),
     Input('Gleason_score', 'value'),
     Input('Measurable_ECE', 'value'),
     Input('ECE_in_prostatectomy', 'value'),
     Input('regra', 'value'),
     ]
)

def generate_feature_matrix(Age, Prostate_volume, PSA_value, lesion_size, PIRADS_V2,
                            Capsular_contact_length, Smooth_capsular_bulging, Capsular_disruption, Unsharp_margin,
                            Irregular_contour, Black_estrition_periprostatic_fat,
                            Retoprostatic_angle_obliteration, Gleason_score,Measurable_ECE,ECE_in_prostatectomy,regra):

    # generate a new X_matrix for use in the predictive models
    column_names = ['Age.at.MRI','Prostate.volume','PSA.value.at.MRI','Index.lesion.size',
       'Capsular.contact.lenght_TLC','Smooth.capsular.bulging','Capsular.disruption','Unsharp.margin',
       'Irregular.contour','Black.estrition.periprostatic.fat','Retoprostatic.angle.obliteration',
       'Measurable.ECE','ECE.in.prostatectomy.specimen_gold.standard','Gleason.score','regra',
       'Index.lesion.PIRADS.V2_3','Index.lesion.PIRADS.V2_4','Index.lesion.PIRADS.V2_5']

    val = [Age,Prostate_volume,PSA_value,PIRADS_V2,lesion_size,Capsular_contact_length,
            Smooth_capsular_bulging,Capsular_disruption,Unsharp_margin,Irregular_contour,
            Black_estrition_periprostatic_fat,Retoprostatic_angle_obliteration,
            Measurable_ECE,ECE_in_prostatectomy,Gleason_score,regra]
    values = [float(v) for v in val]
    
    XX = c.dumm(values)
    XXX = c.minmax(XX,dataw)
    Y = [XXX[:-1]]

    x_patient = pd.DataFrame(data=Y, columns=column_names, index=[0])

    return x_patient.to_json()

@app.callback(
    [Output('Metric_LIME', 'figure'),
     Output('main_text_LIME', 'children'),
     #Output('action_header', 'children'),
     #Output('recommended_action', 'children'),
     Output('Graph_LIME', 'figure')],
    [Input('data_patient', 'children')]
)

def predict_lime_summary(data_patient):

    # read in data and predict likelihood of surgical margin
    x_new = pd.read_json(data_patient)
    prob_0 = rfb.predict_proba(x_new.to_numpy())[:, 0]*100
    prob_1 = rfb.predict_proba(x_new.to_numpy())[:, 1]*100
    
    y_val = [prob_0 if prob_0 > prob_1 else prob_1][0]
    text_val = str(np.round(y_val[0], 1)) + "%"
    clazz = ['negative' if prob_0 > prob_1 else 'positive'][0]

    # create a single bar plot showing likelihood of surgical margin
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=[''],
        x=y_val,
        marker_color='rgb(112, 128, 144)',
        orientation='h',
        width=1,
        text=text_val,
        textposition='auto',
        hoverinfo='skip'
    ))

    # add blocks for risk groups
    bot_val = 0.5
    top_val = 1

    fig1.add_shape(
        type="rect",
        x0=0,
        y0=bot_val,
        x1=0.275686 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="red"
    )
    fig1.add_shape(
        type="rect",
        x0=0.275686 * 100,
        y0=bot_val,
        x1=0.795584 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="orange"
    )
    fig1.add_shape(
        type="rect",
        x0=0.795584 * 100,
        y0=bot_val,
        x1=1 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="green"
    )
    fig1.update_layout(margin=dict(l=0, r=50, t=10, b=15), xaxis={'range': [0, 100]})

    # do lime value calculations for plot
    explainer = lime.lime_tabular.LimeTabularExplainer(trnX,class_names=['SurgycalMargin-0','SurgycalMargin-1'],feature_names=col,
                                                   categorical_features=[5,6,7,8,9,10,11,12,13,14,15,16,17], 
                                                   categorical_names=[col[i] for i in range(5,18)],kernel_width=3,verbose=True)
    exp = explainer.explain_instance(x_new.values.flatten(),rfb.predict_proba,num_features=10,num_samples=1000)
    explist = exp.as_list()
    labels_0 = [v[0] for v in explist if v[1] < 0]
    feature_importance_patient_0 = [abs(v[1]) for v in explist if v[1]<0]
    feature_importance_patient_1 = [v[1] for v in explist if v[1]>0]
    labels_1 = [v[0] for v in explist if v[1]>0]

    specs = [[{'type':'domain'}, {'type':'domain'}]]
    fig2 = make_subplots(rows=1, cols=2,subplot_titles=('SurgicalMargin-0','SurgicalMargin-1'), specs=specs) # Double pie
    fig2.add_trace(go.Pie(labels=labels_0, values=feature_importance_patient_0,scalegroup='one',name=''), 1, 1)
    fig2.add_trace(go.Pie(labels=labels_1, values=feature_importance_patient_1,scalegroup='one',name=''), 1, 2)
    fig2.update_traces(hole=.2, hoverinfo="label+percent+name")
    fig2.update(layout_showlegend=False)
    
    return fig1,\
        f"Based on the patient's profile, the predicted likelihood of a {clazz} surgical margin is {text_val}. ", \
        fig2

@app.callback(
    [Output('Metric_SHAP', 'figure'),
     Output('main_text_SHAP', 'children'),
     #Output('action_header', 'children'),
     #Output('recommended_action', 'children'),
     Output('Graph_SHAP', 'figure')],
    [Input('data_patient', 'children')]
)

def predict_shap_summary(data_patient):

    # read in data and predict likelihood of heart disease
    x_new = pd.read_json(data_patient)
    y_val = rfb.predict_proba(x_new)[:, 1]*100
    text_val = str(np.round(y_val[0], 1)) + "%"
    
    if y_val <= 40:
        risk_grp = 'surgycal margin 0'
    else:
        risk_grp = 'surgycal margin 1'

    # # assign an action related to the risk group
    # rg_actions = {'surgycal margin 0': ['Discuss with patient any single large risk factors they may have, and otherwise '
    #                            'continue supporting healthy lifestyle habits. Follow-up in 12 months'],
    #               'surgycal margin 1': ['Immediate follow-up with patient to discuss next steps including additional '
    #                             'follow-up tests, lifestyle changes and medications.']}

    # next_action = rg_actions[risk_grp][0]

    # create a single bar plot showing likelihood of heart disease
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=[''],
        x=y_val,
        marker_color='rgb(112, 128, 144)',
        orientation='h',
        width=1,
        text=text_val,
        textposition='auto',
        hoverinfo='skip'
    ))

    # add blocks for risk groups
    bot_val = 0.5
    top_val = 1

    fig1.add_shape(
        type="rect",
        x0=0,
        y0=bot_val,
        x1=0.4 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="red"
    )
    
    fig1.add_shape(
        type="rect",
        x0=0.4 * 100,
        y0=bot_val,
        x1=1 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="#3283FE"
    )
    fig1.add_annotation(
        x=0.4 / 2 * 100,
        y=0.75,
        text="Surgical Margin 0",
        showarrow=False,
        font=dict(color="black", size=13)
    )

    fig1.add_annotation(
        x=0.7 * 100,
        y=0.75,
        text="Surgical Margin 1",
        showarrow=False,
        font=dict(color="black", size=13)
    )
    fig1.update_layout(margin=dict(l=0, r=50, t=10, b=15), xaxis={'range': [0, 100]})

    # do shap value calculations for basic waterfall plot
    explainer_patient = shap.TreeExplainer(rfb)
    sp_values_patient = explainer_patient.shap_values(x_new)
    shap_values_patient = [sp_values_patient[0].round(3),sp_values_patient[1].round(3)]
    updated_fnames = x_new.T.reset_index()
    updated_fnames.columns = ['feature', 'value']
    updated_fnames['shap_original'] = pd.Series(shap_values_patient[0].flatten())
    updated_fnames['shap_abs'] = updated_fnames['shap_original'].abs()
    updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)

    # need to collapse those after first 9, so plot always shows 10 bars
    show_features = 9
    num_other_features = updated_fnames.shape[0] - show_features
    col_other_name = f"{num_other_features} other features"
    f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
    f_group.round({'shap_original':3})
    f_group['feature'] = col_other_name
    plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])

    # additional things for plotting
    plot_range = plot_data['shap_original'].cumsum().max() - plot_data['shap_original'].cumsum().min()
    plot_data['text_pos'] = np.where(plot_data['shap_original'].abs() > (1/9)*plot_range, "inside", "outside")
    plot_data['text_col'] = "white"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] < 0), 'text_col'] = "#3283FE"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] > 0), 'text_col'] = "#F6222E"

    fig2 = go.Figure(go.Waterfall(
        name="",
        orientation="h",
        measure=['absolute'] + ['relative']*show_features,
        base=explainer_patient.expected_value[0],
        textposition=plot_data['text_pos'],
        text=plot_data['shap_original'],
        textfont={"color": plot_data['text_col']},
        texttemplate='%{text:+.2f}',
        y=plot_data['feature'],
        x=plot_data['shap_original'],
        connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
        decreasing={"marker": {"color": "#3283FE"}},
        increasing={"marker": {"color": "#F6222E"}},
        hoverinfo="skip"
    ))
    fig2.update_layout(
        waterfallgap=0.2,
        autosize=False,
        width=800,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor='black',
            tickcolor='black',
            ticks='outside',
            ticklen=5
        ),
        margin={'t': 25, 'b': 50},
        shapes=[
            dict(
                type='line',
                yref='paper', y0=0, y1=1.02,
                xref='x', x0=plot_data['shap_original'].sum()+explainer_patient.expected_value,
                x1=plot_data['shap_original'].sum()+explainer_patient.expected_value,
                layer="below",
                line=dict(
                    color="black",
                    width=1,
                    dash="dot")
            )
        ]
    )
    fig2.update_yaxes(automargin=True)
    fig2.update_xaxes(automargin=True)
    # fig2.add_annotation(
    #     yref='paper',
    #     xref='x',
    #     x=explainer_patient.expected_value,
    #     y=0,
    #     text="E[f(x)] = {:.2f}".format(explainer_patient.expected_value[0]),
    #     showarrow=False,
    #     font=dict(color="black", size=14)
    # )
    # fig2.add_annotation(
    #     yref='paper',
    #     xref='x',
    #     x=plot_data['shap_original'].sum()+explainer_patient.expected_value,
    #     y=1,
    #     text="f(x) = {:.2f}".format(plot_data['shap_original'].sum()+explainer_patient.expected_value[0]),
    #     showarrow=True,
    #     font=dict(color="black", size=14)
    # )

    return fig1,\
        f"Based on the patient's profile, the predicted likelihood of a positive surgical margin is {text_val}. ", \
        fig2

# Responsive elements: tab for viewing model explanation
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
    Input('data_patient', 'children')])

def render_content(tab,data_patient):
    if tab == 'tab-1':
        prediction_lime = predict_lime_summary(data_patient)
        return html.Div([
            html.H3('LIME Explanation'),
            dbc.Row(dcc.Graph(
                id='Metric_LIME',
                style={'width': '100%', 'height': 80},
                config={'displayModeBar': False}
            ), style={'marginLeft': 15}),
            dbc.Row([html.Div(id='main_text_LIME', style={'font-size': 20, 'padding': '10px 25px'})]),
            
            dbc.Row([html.Div("Factors contributing to predicted likelihood of surgical margin",
                              style={'font-weight': 'bold', 'font-size': 18, 'padding': '10px 25px'})]),
            
            dbc.Row([html.Div(["The figure below indicates the feature importance of for each of the classified " 
                               "class of factors on the model prediction of the patient's surgical margin likelihood."
                               " The pie chart on the left represents the most important features for the classification"
                               "of Surgical Margin as negative and the pie chart on the left for the classification "
                               "as positive."],
                              style={'font-size': 16, 'padding': '10px 45px'})]),
            dbc.Row(dcc.Graph( 
                id='Graph_LIME',
                config={'displayModeBar': False}
            ), justify="center"),
        ])#,\
        #prediction_lime
        
    elif tab == 'tab-2':
        prediction_shap = predict_shap_summary(data_patient)
        return html.Div([
            html.H3('SHAP Explanation'),
            dbc.Row(dcc.Graph(
                id='Metric_SHAP',
                style={'width': '100%', 'height': 80},
                config={'displayModeBar': False}
            ), style={'marginLeft': 15}),
            dbc.Row([html.Div(id='main_text_SHAP', style={'font-size': 20, 'padding': '10px 25px'})]),
            
            dbc.Row([html.Div("Factors contributing to predicted likelihood of surgical margin",
                              style={'font-weight': 'bold', 'font-size': 18, 'padding': '10px 25px'})]),
            
            dbc.Row([html.Div(["The figure below indicates the impact (magnitude of increase or decrease in "
                               "log-odds) of factors on the model prediction of the patient's surgical margin likelihood."
                               " Blue bars indicate a decrease and red bars indicate an increase in surgical margin "
                               "likelihood. The final risk value at the top of the figure is equal to log(p/(1-p)) "
                               " where p is the predicted likelihood reported above."], 
                              style={'font-size': 16, 'padding': '10px 45px'})]),
            dbc.Row(dcc.Graph( 
                id='Graph_SHAP',
                config={'displayModeBar': False}
            ), justify="center"),     
        ])#,\
        #prediction_shap
# Start the dashboard with defined host and port.
if __name__ == '__main__':
    app.run_server(debug=True,host='127.0.0.1',port=8000)