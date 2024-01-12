import dash
from dash import dcc, html
import pandas as pd
import os
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
import json
from dash import dash_table
import plotly.express as px

# Load the data
root_dir = r"C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA"
analysis_dir = "PRINCE_ANALYSIS"
df_sum = pd.read_excel(os.path.join(root_dir, analysis_dir, "plate_summary2.xlsx"))
df_sum["treatment"] = df_sum["treatment"].replace("1P 100N", "001P100N")
df_sum["treatment"] = df_sum["treatment"].replace("001P100N", "1P/100N/100C")
df_sum["treatment"] = df_sum["treatment"].replace("1P100N100C", "1P/100N/100C")

df_sum["root"] = df_sum["root"].replace("Carrot", "carrot")
df_sum["root"] = df_sum["root"].replace("Carrot Toby", "carrot")


# Dash app
app = JupyterDash(__name__, assets_folder=os.path.join(root_dir, analysis_dir))

app.layout = html.Div(
    [
        # Left Column
        html.Div(
            [
                dcc.Dropdown(
                    id="treatment-dropdown",
                    options=[
                        {"label": i, "value": i}
                        for i in df_sum["treatment"].dropna().unique()
                    ],
                    value=df_sum["treatment"].dropna().unique()[0],
                ),
                dcc.Dropdown(
                    id="root-dropdown",
                    options=[
                        {"label": i, "value": i}
                        for i in df_sum["root"].dropna().unique()
                    ],
                    value=df_sum["root"].dropna().unique()[0],
                ),
                dcc.Dropdown(
                    id="fungus-dropdown",
                    options=[
                        {"label": i, "value": i}
                        for i in df_sum["fungus"].dropna().unique()
                    ],
                    value=df_sum["fungus"].dropna().unique()[0],
                ),
                dcc.Checklist(
                    id="suitable-for-coarse-grain-checklist",
                    options=[
                        {"label": i, "value": i}
                        for i in df_sum["Suitable for coarse grain"].dropna().unique()
                    ],
                    inline=True,
                    value=df_sum["Suitable for coarse grain"]
                    .dropna()
                    .unique()
                    .tolist(),
                ),
                html.Div(id="video-output-list"),
            ],
            style={"flex": "1", "width": "30%", "margin-right": "10px"},
        ),
        # Right Column
        html.Div(
            [
                # Video Player
                html.Div(
                    id="video-player", style={"margin-bottom": "20px", "width": "100%"}
                ),
                # Graph Controls + Graph
                html.Div(
                    [
                        dcc.Dropdown(
                            id="xaxis-column",
                            style={"width": "45%", "display": "inline-block"},
                        ),
                        dcc.Dropdown(
                            id="yaxis-column",
                            style={
                                "width": "45%",
                                "display": "inline-block",
                                "margin-left": "10px",
                            },
                        ),
                        dcc.Graph(id="data-graph"),
                        dash_table.DataTable(
                            id="table",
                            style_table={"height": "300px", "overflowY": "auto"},
                        ),
                    ],
                    style={"flex": "1", "width": "100%"},
                ),
            ],
            style={"flex": "2", "width": "68%", "flex-direction": "column"},
        ),
    ],
    style={
        "display": "flex",
        "justify-content": "space-between",
        "padding": "10px",
        "flex-direction": "row",
    },
)


@app.callback(
    Output("video-output-list", "children"),
    [
        Input("treatment-dropdown", "value"),
        Input("root-dropdown", "value"),
        Input("fungus-dropdown", "value"),
        Input("suitable-for-coarse-grain-checklist", "value"),
    ],
)
def update_video_list(treatment, root, fungus, coarse_grain_values):
    print(filter_videos_for_list(treatment, root, fungus, coarse_grain_values))
    return filter_videos_for_list(treatment, root, fungus, coarse_grain_values)


def filter_videos_for_list(treatment, root, fungus, coarse_grain_values):
    mask = (
        (df_sum["treatment"] == treatment)
        & (df_sum["root"] == root)
        & (df_sum["fungus"] == fungus)
        # & (df_sum["Suitable for coarse grain"].isin(coarse_grain_values))
    )
    unique_ids = df_sum[mask]["unique_id"].tolist()
    video_filenames = [
        uid
        for uid in unique_ids
        if os.path.exists(
            os.path.join(root_dir, analysis_dir, str(uid), f"{uid}_stitched.mp4")
        )
    ]
    return html.Ul(
        [
            html.Li(html.A(uid, href="#", id={"type": "video-link", "uid": uid}))
            for uid in video_filenames
        ]
    )


@app.callback(
    Output("data-graph", "figure"),
    [
        Input("xaxis-column", "value"),
        Input("yaxis-column", "value"),
        Input("table", "data"),
    ],
)
def update_graph(x_col, y_col, table_data):
    if not x_col or not y_col or not table_data:
        return dash.no_update

    df = pd.DataFrame(table_data)
    fig = px.scatter(df, x=x_col, y=y_col)
    return fig


@app.callback(
    [
        Output("video-player", "children"),
        Output("table", "data"),
        Output("xaxis-column", "options"),
        Output("yaxis-column", "options"),
    ],
    [Input({"type": "video-link", "uid": dash.dependencies.ALL}, "n_clicks")],
)
def play_video(clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]
    uid = json.loads(clicked_id)["uid"]

    video_path = os.path.join(str(uid), f"{uid}_stitched.mp4")
    video_path = "/assets/" + video_path.replace("\\", "/")

    # Look for Analysis_* folder
    analysis_folder = next(
        (
            folder
            for folder in os.listdir(os.path.join(root_dir, analysis_dir, str(uid)))
            if folder.startswith("Analysis_")
        ),
        None,
    )

    if analysis_folder:
        json_path = os.path.join(
            root_dir, analysis_dir, str(uid), analysis_folder, "time_plate_info.json"
        )
        if os.path.exists(json_path):
            df = pd.read_json(json_path).transpose()
    else:
        df = pd.DataFrame()

    column_options = [{"label": col, "value": col} for col in df.columns]
    return (
        html.Video(src=video_path, controls=True, style={"width": "100%"}),
        df.to_dict("records"),
        column_options,
        column_options,
    )


if __name__ == "__main__":
    app.run_server(debug=False)
