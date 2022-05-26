# Standard
#from distutils.log import debug
import os
import pandas as pd
import numpy as np

#Local modules
import localmodules.conversor as c

# Dash components
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# For plotting risk indicator and for creating waterfall plot
import lime.lime_tabular
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# To import joblib file model objects
import joblib


# Load model and pipeline
# current_folder = os.path.dirname(__file__)
# #hd_model_obj = joblib.load(os.path.join(current_folder, 'heart_disease_prediction_model_Jul2020.pkl'))

# normally we would want the pipeline object as well, but in this example transformation is minimal so we will just
# construct the require format on the fly from data entry. Also means we don't need to rely on PyCaret here
# object has 2 slots, first is data pipeline, second is the model object
# hdpred_model = hd_model_obj[1]
# hd_pipeline = []

#Random Forest and Database Load
file = './zDatabase/XAI - Limpo_dummified_minmax_smote.csv'
fileTrain = './zDatabase/XAI - Limpo_dummified_minmax_smote_train.csv'
rfb = joblib.load('./zDatabase/randomforests.joblib')
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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                            {'label': 'Intermediate cancer significance', 'value': '3'},
                            {'label': 'High cancer significance', 'value': '4'},
                            {'label': 'Very High cancer significance', 'value': '5'}
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
                            {'label': 'Bellow 7', 'value': '0'},
                            {'label': 'Above 7', 'value': '1'}
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
        dbc.Col([html.Div("Predicted surgical margin",
                          style={'font-weight': 'bold', 'font-size': 20}),
            dbc.Row(dcc.Graph(
                id='Metric_1',
                style={'width': '100%', 'height': 80},
                config={'displayModeBar': False}
            ), style={'marginLeft': 15}),
            dbc.Row([html.Div(id='main_text', style={'font-size': 16, 'padding': '10px 25px'})]),
            
            dbc.Row([html.Div("Factors contributing to predicted likelihood of surgical margin",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            
            dbc.Row([html.Div(["The figure below indicates the feature importance of for each of the classified " 
                               "class of factors on the model prediction of the patient's surgical margin likelihood."
                               " The pie chart on the left represents the most important features for the classification"
                               "of Surgical Margin as negative and the pie chart on the left for the classification "
                               "as positive."],
                              style={'font-size': 16, 'padding': '10px 45px'})]),
            dbc.Row(dcc.Graph( 
                id='Graph',
                config={'displayModeBar': False}
            ), justify="center"),
            #dbc.Row([html.Div(id='action_header',
            #                  style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            # dbc.Row(
            #     dbc.Col([html.Div(id='recommended_action')], width={"size": 11},
            #             style={'font-size': 16, 'padding': '10px 25px',
            #                    'backgroundColor': '#E2E2E2', 'marginLeft': 25})),
            ],
            style={'padding': '10px 25px'}
        ),
    ]),
    dbc.Row(
        html.Div(
            [
                dbc.Button(
                    "Predictive model information",
                    id="collapse-button",
                    className="mb-3",
                    color="primary",
                ),
                dbc.Collapse(
                    dbc.Card(dbc.Row([ #Colapsable thingy
                        dbc.Col([
                            html.Div('Predictive model information',
                                     style={'font-weight': 'bold', 'font-size': 20, 'padding': '0px 0px 20px 0px'}),
                            html.Div('Data source',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(['A cohort of 121 patients at the Hospital da Luz were assessed on multiple '
                                      'characteristics and whether the surgical margin was followed was also recorded. '
                                      'In total 45% (n=139) of patients had heart disease.'],
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
                                      'achieved an average AUC of 1.0 in the training set and 0.86 in the '
                                      'test set. Figure 1 to the far right indicates what the model identified as '
                                      'importance of predictors of the surgical margin. The more important features are '
                                      'towards the top of the figure, which includes characteristics like affected '
                                      'PSA value at MRI, Age at MRI, prostate volume and if smooth capsular bulging'
                                      'can be or not identified.'],
                                     style={'font-size': 14, 'padding': '0px 0px 20px 0px'}),
                        ]),
                        dbc.Col([
                            html.Div('Table 1. Cohort Table',
                                     style={'font-weight': 'bold', 'font-size': 20, 'textAlign': 'middle'}),
                            html.Div(className='container',
                                children=[html.Img(src=app.get_asset_url('Cohort_table.png'),
                                                   style={'height': '100%', 'width': '100%'})])]),
                        dbc.Col([
                            html.Div('Figure 1. Feature importance',
                                     style={'font-weight': 'bold', 'font-size': 20}),
                            html.Div(className='container',
                                     children=[html.Img(src=app.get_asset_url('Feature_Importance.png'),
                                                        style={'height': '90%', 'width': '90%'})])])
                        ], style={'padding': '20px 20px'})),
                    id="collapse",
                ),
            ]
        ),
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
    [State("collapse", "is_open")],
)
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
    XXX = c.minmax(XX,data)
    XXX.pop(-1)

    x_patient = pd.DataFrame(data=[XXX],
                             columns=column_names,
                             index=[0])

    return x_patient.to_json()


@app.callback(
    [Output('Metric_1', 'figure'),
     Output('main_text', 'children'),
     #Output('action_header', 'children'),
     #Output('recommended_action', 'children'),
     Output('Graph', 'figure')],
    [Input('data_patient', 'children')]
)
def predict_hd_summary(data_patient):

    # read in data and predict likelihood of heart disease
    x_new = pd.read_json(data_patient)
    prob_0 = rfb.predict_proba(x_new.to_numpy())[:, 0]*100
    prob_1 = rfb.predict_proba(x_new.to_numpy())[:, 1]*100
    y_val = [prob_0 if prob_0>prob_1 else prob_1][0]
    text_val = str(np.round(y_val[0], 1)) + "%"
    c = ['negative' if prob_0>prob_1 else 'positive'][0]
    
    # assign a risk group
    # if y_val/100 <= 0.275685:
    #     risk_grp = 'low risk'
    # elif y_val/100 <= 0.795583:
    #     risk_grp = 'medium risk'
    # else:
    #     risk_grp = 'high risk'

    # # assign an action related to the risk group
    # rg_actions = {'low risk': ['Discuss with patient any single large risk factors they may have, and otherwise '
    #                            'continue supporting healthy lifestyle habits. Follow-up in 12 months'],
    #               'medium risk': ['Discuss lifestyle with patient and identify changes to reduce risk. '
    #                               'Schedule follow-up with patient in 3 months on how changes are progressing. '
    #                               'Recommend performing simple tests to assess positive impact of changes.'],
    #               'high risk': ['Immediate follow-up with patient to discuss next steps including additional '
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
    fig1.update_layout(margin=dict(l=0, r=50, t=10, b=10), xaxis={'range': [0, 100]})

    # do shap value calculations for basic waterfall plot
    explainer = lime.lime_tabular.LimeTabularExplainer(trnX,class_names=['SurgycalMargin-0','SurgycalMargin-1'],feature_names=col,
                                                   categorical_features=[5,6,7,8,9,10,11,12,13,14,15,16,17], 
                                                   categorical_names=[col[i] for i in range(5,18)],kernel_width=3,verbose=True)
    exp = explainer.explain_instance(x_new.values.flatten(),rfb.predict_proba,num_features=9,num_samples=1000)
    explist = exp.as_list()
    feature_importance_patient_pos = [v[1] for v in explist if v[1]>0]
    feature_importance_patient_neg = [abs(v[1]) for v in explist if v[1]<0]
    labels_pos = [v[0] for v in explist if v[1]>0]
    labels_neg = [v[0] for v in explist if v[1]<0]

    specs = [[{'type':'domain'}, {'type':'domain'}]]
    fig2 = make_subplots(rows=1, cols=2,subplot_titles=('SurgycalMargin-0','SurgycalMargin-1'), specs=specs) # Double pie
    fig2.add_trace(go.Pie(labels=labels_pos, values=feature_importance_patient_pos,name=''), 1, 1)
    fig2.add_trace(go.Pie(labels=labels_neg, values=feature_importance_patient_neg,name=''), 1, 2)
    fig2.update_traces(hole=.2, hoverinfo="label+percent+name")
    fig2.update(layout_showlegend=False)

    return fig1,\
        f"Based on the patient's profile, the predicted likelihood of a {c} surgical is margin {text_val}. ", \
        fig2


# Start the dashboard with defined host and port.
if __name__ == '__main__':
    app.run_server(debug=True,host='127.0.0.1',port=8000)