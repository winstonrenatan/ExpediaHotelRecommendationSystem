# Import dash
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Import csv
import csv
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

# Refering to a table that describe which hotel_cluster have which facilities
cluster_hotel = pd.read_csv ('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/GitHub Documentation/csv_data/hotel_cluster_csv.csv', sep=',')

##########################################################################


######################### RANDOM FOREST  #################################
# Read data
features = pd.read_csv('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/GitHub Documentation/csv_data/clean_train.csv', sep=',')

# Labels are the values we want to predict
labels = np.array(features['hotel_cluster'])
# Remove the labels from the features
# axis 1 refers to the columns
features = features.drop('hotel_cluster', axis=1)

# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

# Instantiate model with 1 decision trees
rf = RandomForestRegressor(n_estimators=1, random_state=42)
# Train the model on training data
rfTrain = rf.fit(train_features, train_labels)


#########################################################################


#################### SUPPORT VECTOR MACHINE #############################
# Read data
df = pd.read_csv('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/GitHub Documentation/csv_data/clean_train_one_percent.csv', sep=',')

# Labels are the values we want to predict
labels = np.array(df['hotel_cluster'])
# Remove the labels from the features
# axis 1 refers to the columns
df = df.drop('hotel_cluster', axis=1)

# Saving feature names for later use
df_list = list(df.columns)
# Convert to numpy array
df = np.array(df)

# Split the data into training and testing sets
train_df, test_df, train_labels, test_labels = train_test_split(
    df, labels, test_size=0.25, random_state=109)

# Create a svr regression
# Radial Basis Function Kernel
clf = svm.SVR(kernel='rbf', gamma='auto')  

# Train the model using the training sets
svm = clf.fit(train_df, train_labels)

#############################################################################

################## HOTEL INFORMATION PART ###################################

# To determine whether the hotel have certain facility or not
def YesNo(number):
    # Hotel have the facility
    if(number==1):
        return "YES"
    # Hotel do not have the facility
    else:
        return "NO"

# Give out hotel information according to the cluster
def hotel_info (hotel_clus):
    # Print information of the hotel according to its cluster
    hotel_star="HOTEL STAR: {}. \n".format(cluster_hotel.iloc[hotel_clus-1, 1])
    hotel_wifi="WIFI: {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 2]))
    hotel_pool="POOL ACCESS: {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 3]))
    hotel_restaurant="RESTAURANT:  {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 4]))
    hotel_bar="BAR AND ALCOHOL: {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 5]))
    hotel_aircon="AIR CONDITIONER: {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 6]))
    final_info = hotel_star + hotel_wifi + hotel_pool + hotel_restaurant + hotel_bar + hotel_aircon
    return final_info


#############################################################################

################## DASH PART ################################################

header = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Expedia Hotel Recommendation System"),
            html.P("Welcome to Expedia Hotel recommendation System."),
            html.Hr(),
        ], md=12)
    ])
], className="mt-12 p-2")

body = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Input Your Details Below :"),
            html.H6("Continents: 1 = North America | 2 = South America | 3 = Asia | 4 = Australia | 5 = Europe | 6 = Africa"),
            html.H6("There are only 195 countries in the world."),
            html.H6("Using Mobile? 1 = You use Mobile || 0 = You do not use Mobile"),
            html.H6("Package? 1 = This is a package || 0 = This is not a package"),
            html.Hr(),
        ], md=12)
    ]),
    # User Information
    dbc.Row([
        dbc.Col([
            html.H2("#Your Information"),
        ], md=12),

        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Site Name"),
                dbc.Input(id="site_name",
                          placeholder="Input here ..", type="text")
            ])
        ], md=4),

        # Posa continent
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Continent"),
                dbc.Input(id="posa_continent", placeholder="Input here ..",
                        type="number", min=1, max=6)
            ])
        ], md=4),

        # Country location
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Country Location"),
                dbc.Input(id="user_location_country", placeholder="Input here ..",
                            type="number", min=1, max=195)
            ])
        ], md=4),

        # Country Region
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Country Region"),
                dbc.Input(id="user_location_region",
                          placeholder="Input here ..", type="text")
            ])
        ], md=4),

        # User City
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("City"),
                dbc.Input(id="user_location_city",
                          placeholder="Input here ..", type="text")
            ])
        ], md=4),

        # Destination Distance
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Destination Distance"),
                dbc.Input(id="orig_destination_distance",
                          placeholder="Input here ..", type="text")
            ])
        ], md=4),
    ]),

    # Additional Information
    dbc.Row([
        dbc.Col([
            html.H2("#Additional Information"),
        ], md=12),

        # User ID
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("User ID"),
                dbc.Input(id="user_id", placeholder="Input here ..", type="text")
            ])
        ], md=3),

        # Is Mobile
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Access Using Mobile?"),
                dbc.Input(id="is_mobile", placeholder="Input here ..",
                          type="number", min=0, max=1)
            ])
        ], md=3),

        # Is_Package
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Is Package"),
                dbc.Input(id="is_package", placeholder="Input here ..",
                          type="number", min=0, max=1)
            ])
        ], md=3),

        # channel
        dbc.Col([
            dbc.FormGroup([
                dbc.Label("Channel"),
                dbc.Label(". "),
                dbc.Input(id="channel", placeholder="Input here ..", type="text")
            ])
        ], md=3),
    ]),
    dbc.Row([
            # Family Member
            dbc.Col([
                html.H2("#Family Member"),
                dbc.FormGroup([
                    dbc.Label("Adult: Minimum 1 and Maximum 10"),
                    dbc.Input(id="srch_adults_cnt", placeholder="Input here ..",
                              type="number", min=1, max=10, step=1)
                ]),
                dbc.FormGroup([
                    dbc.Label("Child"),
                    dbc.Input(id="srch_children_cnt", placeholder="Input here ..",
                              type="number", min=0, max=10, step=1)
                ]),
                dbc.FormGroup([
                    dbc.Label("Room"),
                    dbc.Input(id="srch_rm_cnt", placeholder="Input here ..",
                              type="number", min=1, max=10, step=1)
                ])
            ], md=4),

            # Destination
            dbc.Col([
                html.H2("#Destination"),
                dbc.FormGroup([
                    dbc.Label("Destination ID"),
                    dbc.Input(id="srch_destination_id", placeholder="Input here ..",
                              type="text")
                ]),
                dbc.FormGroup([
                    dbc.Label("Destination Type"),
                    dbc.Input(id="srch_destination_type_id", placeholder="Input here ..",
                              type="text")
                ]),
                dbc.FormGroup([
                    dbc.Label("Similar Event Context"),
                    dbc.Input(id="cnt", placeholder="Input here ..",
                              type="text")
                ]),
            ], md=4),

            # Hotel
            dbc.Col([
                html.H2("#Hotel"),
                dbc.FormGroup([
                    dbc.Label("Hotel Continent:"),
                    dbc.Input(id="hotel_continent", placeholder="Input here ..",
                              type="number", min=1, max=6)
                ]),
                dbc.FormGroup([
                    dbc.Label("Hotel Country:"),
                    dbc.Input(id="hotel_country", placeholder="Input here ..",
                              type="number", min=1, max=195)
                ]),
                dbc.FormGroup([
                    dbc.Label("Hotel Market:"),
                    dbc.Input(id="hotel_market", placeholder="Input here ..",
                              type="number")
                ]),
            ], md=4),
            ]),
    html.Hr(),
    dbc.Button("Submit", color="primary", block=True,
               id="submit-button", className="ml-auto", n_clicks=0),
    html.Br(),

    # RESULT
    dbc.Row([
        dbc.Col([
            html.H3("YOUR HOTEL RECOMMENDATION")
        ], md=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Plaintext(id="hotel_recommendation", style={'font-size': '20px'})
        ], md=12)
    ])
], className="mt-4")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([header, body])


@app.callback(
    Output('hotel_recommendation', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('site_name', 'value'),
     State('posa_continent', 'value'),
     State('user_location_country', 'value'),
     State('user_location_region', 'value'),
     State('user_location_city', 'value'),
     State('orig_destination_distance', 'value'),
     State('user_id', 'value'),
     State('is_mobile', 'value'),
     State('is_package', 'value'),
     State('channel', 'value'),
     State('srch_adults_cnt', 'value'),
     State('srch_children_cnt', 'value'),
     State('srch_rm_cnt', 'value'),
     State('srch_destination_id', 'value'),
     State('srch_destination_type_id', 'value'),
     State('cnt', 'value'),
     State('hotel_continent', 'value'),
     State('hotel_country', 'value'),
     State('hotel_market', 'value'),
     ]
)

# To get the input and prints out the final output
def predictHC(*args):
    values = list(args)
    values.pop(0)
    
    # Prediction using Random Forest
    prediction1 = rfTrain.predict([values])
    prediction1 = int(prediction1)
    outputRF = "Predicted in Random Forest Model is {}. \n".format(prediction1)
    hotelRFinfo = hotel_info(prediction1)
    outputRF = outputRF + hotelRFinfo
    
    # Prediction using Support Vector Machine
    prediction2 = svm.predict([values])
    prediction2 = int(prediction2)
    outputSVM = "Predicted in Support Vector Regression is {}. \n".format(prediction2)
    hotelSVMinfo = hotel_info(prediction2)
    outputSVM = outputSVM + hotelSVMinfo

    # Final output string
    outputFinal = outputRF + "\n" + outputSVM
    return outputFinal

if __name__ == "__main__":
    app.run_server()