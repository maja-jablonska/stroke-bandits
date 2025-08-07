

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


data_path = "Model/datashare_aug2015.sas7bdat"
df = pd.read_sas(data_path, format="sas7bdat", encoding="latin1")


columns_needed = [
    'age', 'gender', 'pretrialexp', 'treatment', 'surv18',
    'nihss', 'randdelay', 'sbprand', 'dbprand', 'glucose',
    'gcs_score_rand', 'weight'
]
df_clean = df[columns_needed].dropna().copy()
df_clean['treatment_encoded'] = df_clean['treatment'].map({'rt-PA': 1, 'Placebo': 0})
df_clean['reward'] = df_clean['surv18'] / 548

# Train model
feature_cols = [
    'age', 'gender', 'pretrialexp', 'nihss', 'randdelay',
    'sbprand', 'dbprand', 'glucose', 'gcs_score_rand', 'weight',
    'treatment_encoded'
]
X = df_clean[feature_cols]
y = df_clean['reward']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# css
app.layout = html.Div([
    html.Div([
        html.H1("\ud83d\udcca Contextual Bandit Treatment Decision", className="text-center text-primary mb-4"),

        html.Div([
            html.Label("Select Patient Index:"),
            dcc.Slider(0, len(df_clean)-1, 1, value=0, id="patient-slider")
        ], style={"margin": "30px"}),

        html.Div(id="patient-info", className="alert alert-info", style={"fontSize": 18}),
        html.Div(id="prediction-output", className="alert alert-success", style={"fontSize": 20}),

        dcc.Graph(id="reward-bar-plot")
    ], style={
        "background": "linear-gradient(to bottom right, #f0f8ff, #e6f7ff)",
        "padding": "40px",
        "borderRadius": "12px",
        "boxShadow": "0px 0px 10px rgba(0,0,0,0.1)",
        "maxWidth": "900px",
        "margin": "auto"
    }),

    html.Footer("Built with \ud83d\udc99 using Dash, Plotly & Python", style={
        "textAlign": "center",
        "padding": "10px",
        "fontSize": "14px",
        "color": "#888"
    })
])

# Callback
@app.callback(
    Output("patient-info", "children"),
    Output("prediction-output", "children"),
    Output("reward-bar-plot", "figure"),
    Input("patient-slider", "value")
)
def update_output(patient_index):
    row = df_clean.iloc[patient_index]
    context = row[[
        'age', 'gender', 'pretrialexp', 'nihss', 'randdelay',
        'sbprand', 'dbprand', 'glucose', 'gcs_score_rand', 'weight'
    ]].values

    input_0 = np.append(context, 0).reshape(1, -1)
    input_1 = np.append(context, 1).reshape(1, -1)

    pred_0 = model.predict(input_0)[0]
    pred_1 = model.predict(input_1)[0]
    best = "rt-PA" if pred_1 > pred_0 else "Placebo"

    info = f"Patient {patient_index} \u2014 Age: {row['age']}, Gender: {row['gender']}, NIHSS: {row['nihss']}"
    result = f"\ud83d\udc8a Predicted Reward \u2192 Placebo: {pred_0:.2f}, rt-PA: {pred_1:.2f} \u2192 Best: {best}"

    fig = go.Figure(data=[
        go.Bar(
            name="Placebo",
            x=["Placebo"],
            y=[pred_0],
            marker_color='lightgray',
            text=[f'{pred_0:.2f}'],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"Placebo<br>Predicted Reward: {pred_0:.2f}"]
        ),
        go.Bar(
            name="rt-PA",
            x=["rt-PA"],
            y=[pred_1],
            marker_color='lightblue',
            text=[f'{pred_1:.2f}'],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"rt-PA<br>Predicted Reward: {pred_1:.2f}"]
        )
    ])
    fig.update_layout(
        title="Predicted Rewards",
        yaxis=dict(title='Reward', range=[0, 1]),
        xaxis=dict(title='Treatment'),
        plot_bgcolor='white',
        bargap=0.4,
        hovermode='x unified',
        transition=dict(duration=500)
    )

    return info, result, fig

#running
if __name__ == "__main__":
    app.run(debug=True)
