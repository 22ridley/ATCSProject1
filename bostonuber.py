import warnings

import bokeh as bokeh
import seaborn as sns
from bokeh.models import HoverTool
import matplotlib
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas
import pickle
import matplotlib.ticker as tkr
import pylab
import geopandas
import adjustText as aT
from mpldatacursor import datacursor
import folium
import mapclassify
from geopandas import GeoDataFrame
import numpy as np
import descartes
from shapely.geometry import Point, Polygon
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def custom_round(x, base=5):
    return int(base * round(float(x)/base))

def numfmt(x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x / 2000)
    return s

def price_vs_location(df):
    df = df.replace(to_replace=["North End"], value=0)
    df = df.replace(to_replace=["North Station", "West End"], value=1)
    df = df.replace(to_replace=["Haymarket Square", "Financial District", "Downtown"], value=2)
    df = df.replace(to_replace=["Beacon Hill"], value=3)
    df = df.replace(to_replace=["Theatre District", "Chinatown"], value=4)
    df = df.replace(to_replace=["Back Bay"], value=5)
    df = df.replace(to_replace=["South Station", "South End"], value=6)
    df = df.replace(to_replace=["Boston University", "Fenway"], value=7)
    df = df.replace(to_replace=["Northeastern University", "Roxbury"], value=8)

    # replacing brand name
    df.loc[df['cab_type'] == 'Lyft', ['cab_type']] = 0
    df.loc[df['cab_type'] == 'Uber', ['cab_type']] = 1

    # replacing car specific
    df.loc[df['name'] == 'Shared', ['name']] = 0
    df.loc[df['name'] == 'Lyft', ['name']] = 1
    df.loc[df['name'] == 'Lyft XL', ['name']] = 2
    df.loc[df['name'] == 'Lux', ['name']] = 3
    df.loc[df['name'] == 'Lux Black', ['name']] = 4
    df.loc[df['name'] == 'Lux Black XL', ['name']] = 5
    df.loc[df['name'] == 'UberPool', ['name']] = 6
    df.loc[df['name'] == 'UberX', ['name']] = 7
    df.loc[df['name'] == 'UberXL', ['name']] = 8
    df.loc[df['name'] == 'Black', ['name']] = 9
    df.loc[df['name'] == 'Black SUV', ['name']] = 10
    df.loc[df['name'] == 'WAV', ['name']] = 11

    convert_dict = {'source': int, 'destination': int, 'cab_type': int, 'name': int}
    df = df.astype(convert_dict)

    dfDT = df[(df.destination == 2)]  # 'Downtown'
    meanDT = dfDT["prices"].mean()
    dfWE = df[(df.destination == 1)]  # 'West End'
    meanWE = dfWE["prices"].mean()
    dfRB = df[(df.destination == 8)]  # Roxbury
    meanRB = dfRB["prices"].mean()
    dfFW = df[(df.destination == 7)]  # 'Fenway'
    meanFW = dfFW["prices"].mean()
    dfCT = df[(df.destination == 4)]  # 'Chinatown'
    meanCT = dfCT["prices"].mean()
    dfBB = df[(df.destination == 5)]  # 'Back Bay'
    meanBB = dfBB["prices"].mean()
    dfNE = df[(df.destination == 0)]  # 'North End'
    meanNE = dfNE["prices"].mean()
    dfBH = df[(df.destination == 3)]  # 'Beacon Hill'
    meanBH = dfBH["prices"].mean()
    dfSE = df[(df.destination == 6)]  # 'South End'
    meanSE = dfSE["prices"].mean()

    a_list = [['Downtown', meanDT], ['West End', meanWE], ['Roxbury', meanRB], ['Fenway', meanFW],
              ['Chinatown', meanCT], ['Back Bay', meanBB], ['North End', meanNE], ['Beacon Hill', meanBH],
              ['South End', meanSE], ['Leather District', meanDT], ['Bay Village', meanBB]]
    meandf = pandas.DataFrame(a_list, columns=['Name', 'MeanPrice'])

    crs = {'init': 'epsg:4326'}
    hoods_map = geopandas.read_file('Boston_Hoods/Boston_Neighborhoods.shp')
    hoods_map['District'] = hoods_map['Name']

    merged = hoods_map.set_index('Name').join(meandf.set_index('Name'))
    merged = merged.dropna()

    fig, ax = plt.subplots(1, figsize=(6, 6))
    merged.plot(column='MeanPrice', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.5', legend=True)

    merged["center"] = merged["geometry"].centroid
    merged_points = merged.copy()
    merged_points.set_geometry("center", inplace=True)
    texts = []
    for x, y, label in zip(merged_points.geometry.x, merged_points.geometry.y, merged_points['District']):
        texts.append(plt.text(x, y, label, fontsize=6))
    aT.adjust_text(texts, force_points=0.3, force_text=0.8, expand_points=(1, 1), expand_text=(1, 1),
                   arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))

    plt.title('Average Rideshare Prices in Boston', fontsize=15, fontweight='bold')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show()


def hour_vs_surge(dfplot):
    dfplot = dfplot[(dfplot.name == 'Lyft')]
    dfplot = dfplot.iloc[:3000, :]
    dfplot['hour'] = dfplot['hour'].apply(lambda x: custom_round(x, base=4))
    r = sns.violinplot(x='hour', y='surge_multiplier', data=dfplot)
    r.set_title('Surge Multiplier Throughout the Day')
    r.set_xlabel('Hour')
    r.set_ylabel('Surge Multiplier')
    r.set_xticklabels(['12:00AM', '4:00AM', '8:00AM', '12:00PM', '4:00PM', '8:00PM', '12:00AM'], size=7)
    plt.show()


def hour_vs_price(dfplot):
    prices = dfplot[~dfplot['prices'].isna()]
    plt.hist([prices[prices['cab_type'] == 'Uber']['hour'], prices[prices['cab_type'] == 'Lyft']['hour']], alpha=0.4,
             label=['Uber', 'Lyft'])
    plt.legend(loc='upper right')
    plt.title('Average Price Throughout the Day by Brand')
    plt.xlabel('Hour')
    plt.ylabel('Average Price')
    plt.xticks([0, 5, 10, 15, 20], ['12:00AM', '5:00AM', '10:00AM', '3:00PM', '8:00PM'], rotation='horizontal')
    yfmt = tkr.FuncFormatter(numfmt)
    plt.gca().yaxis.set_major_formatter(yfmt)
    plt.show()


def ride_vs_price(dfplot):
    sns.set(style="darkgrid")
    p = sns.boxplot(x='name', y='prices', data=dfplot)
    p.set_xlabel("Ride Type", fontsize=10)
    p.set_ylabel("Average Price", fontsize=10)
    p.set_xticklabels(
        ['Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL', 'UberPool', 'UberX', 'UberXL', 'Black',
         'Black SUV', 'WAV'], size=5)
    p.set_title("Ride Type and Average Price", fontsize=20)
    plt.show()


def weather_vs_surge(df):
    #this data comes from november and december only in 2018
    df = df[(df.name == 'Lyft')]
    df = df.iloc[:2000, :]
    #Simplifing the labels
    df.loc[df['long_summary'].str.contains('fog'), ['short_summary']] = ' Foggy '
    df.loc[df['long_summary'].str.contains('cloud'), ['short_summary']] = ' Cloudy '
    df.loc[df['long_summary'].str.contains('rain'), ['short_summary']] = ' Rain '
    df.loc[df['short_summary'] == ' Light Rain ', ['short_summary']] = ' Rain '
    df.loc[df['short_summary'] == ' Drizzle ', ['short_summary']] = ' Rain '
    df.loc[df['short_summary'].str.contains('Cloudy'), ['short_summary']] = ' Cloudy '
    df.loc[df['short_summary'] == ' Possible Drizzle ', ['short_summary']] = ' Rain '
    #df['precipIntensity'] = df['precipIntensity'].apply(lambda x: custom_round(x, base=0.04))
    r = sns.violinplot(x='short_summary', y='surge_multiplier', data=df)
    r.set_title('Weather and Surge Multiplier')
    r.set_xlabel('Weather', fontsize=10)
    r.set_xticklabels(
        ['Cloudy', 'Foggy', 'Rainy', 'Clear', 'Snowy'], size=7)
    r.set_ylabel('Surge Multiplier')
    #r.set_xticklabels(['12:00AM', '4:00AM', '8:00AM', '12:00PM', '4:00PM', '8:00PM', '12:00AM'], size=7)
    plt.show()


def price_predictor(df):
    warnings.filterwarnings('ignore')
    df = df.iloc[:100000, :]
    df = df.drop(columns=['short_summary', 'long_summary', 'month', 'temperature', 'day', 'precipIntensity', 'precipProbability'])

    df = df.replace(to_replace=["North End"], value=0)
    df = df.replace(to_replace=["North Station", "West End"], value=1)
    df = df.replace(to_replace=["Haymarket Square", "Financial District", "Downtown"], value=2)
    df = df.replace(to_replace=["Beacon Hill"], value=3)
    df = df.replace(to_replace=["Theatre District", "Chinatown"], value=4)
    df = df.replace(to_replace=["Back Bay"], value=5)
    df = df.replace(to_replace=["South Station", "South End"], value=6)
    df = df.replace(to_replace=["Boston University", "Fenway"], value=7)
    df = df.replace(to_replace=["Northeastern University", "Roxbury"], value=8)

    # replacing brand name
    df.loc[df['cab_type'] == 'Lyft', ['cab_type']] = 0
    df.loc[df['cab_type'] == 'Uber', ['cab_type']] = 1

    # replacing car specific
    df.loc[df['name'] == 'Shared', ['name']] = 0
    df.loc[df['name'] == 'Lyft', ['name']] = 1
    df.loc[df['name'] == 'Lyft XL', ['name']] = 2
    df.loc[df['name'] == 'Lux', ['name']] = 3
    df.loc[df['name'] == 'Lux Black', ['name']] = 4
    df.loc[df['name'] == 'Lux Black XL', ['name']] = 5
    df.loc[df['name'] == 'UberPool', ['name']] = 6
    df.loc[df['name'] == 'UberX', ['name']] = 7
    df.loc[df['name'] == 'UberXL', ['name']] = 8
    df.loc[df['name'] == 'Black', ['name']] = 9
    df.loc[df['name'] == 'Black SUV', ['name']] = 10
    df.loc[df['name'] == 'WAV', ['name']] = 11

    convert_dict = {'source': int, 'destination': int, 'cab_type': int, 'name': int}
    df = df.astype(convert_dict)

    df['prices'] = df['prices'].apply(np.int64)
    df['prices'] = df['prices'].apply(lambda x: custom_round(x, base=5))
    df = df[(df.prices <= 35)]
    #print(df.head())
    array = df.values
    X = array[:, :-1]  # every row, all columns  excluding the last
    y = array[:, -1]  # every row, only the last column
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    filename = 'finalized_model.sav'
    #pickle.dump(model, open(filename, 'wb'))
    predictions = model.predict(X_validation)

    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


def distances_only(df):
    warnings.filterwarnings('ignore')
    df = df.iloc[:200000, :]
    df = df.drop(columns=['short_summary', 'long_summary', 'month', 'temperature', 'day', 'precipIntensity',
                          'precipProbability', 'name', 'cab_type', 'surge_multiplier', 'hour', 'prices'])

    df = df.replace(to_replace=["North End"], value=0)
    df = df.replace(to_replace=["North Station", "West End"], value=1)
    df = df.replace(to_replace=["Haymarket Square", "Financial District", "Downtown"], value=2)
    df = df.replace(to_replace=["Beacon Hill"], value=3)
    df = df.replace(to_replace=["Theatre District", "Chinatown"], value=4)
    df = df.replace(to_replace=["Back Bay"], value=5)
    df = df.replace(to_replace=["South Station", "South End"], value=6)
    df = df.replace(to_replace=["Boston University", "Fenway"], value=7)
    df = df.replace(to_replace=["Northeastern University", "Roxbury"], value=8)

    convert_dict = {'source': int, 'destination': int}
    df = df.astype(convert_dict)
    df['distance'] = df['distance'].apply(np.int64)
    df['distance'] = df['distance'].apply(lambda x: custom_round(x, base=1))
    #print(df.head())
    array = df.values
    X = array[:, :-1]  # every row, all columns  excluding the last
    y = array[:, -1]  # every row, only the last column
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    filename = 'distance_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    predictions = model.predict(X_validation)

    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


if __name__ == '__main__':
    pandas.set_option('display.expand_frame_repr', False)
    url = "rideshare_kaggle.csv"
    header = ["id","timestamp","hour","day","month","datetime","timezone","source","destination","cab_type","product_id","name","price","distance","surge_multiplier","latitude","longitude","temperature","apparentTemperature","short_summary","long_summary","precipIntensity","precipProbability","humidity","windSpeed","windGust","windGustTime","visibility","temperatureHigh","temperatureHighTime","temperatureLow","temperatureLowTime","apparentTemperatureHigh","apparentTemperatureHighTime","apparentTemperatureLow","apparentTemperatureLowTime","icon","dewPoint","pressure","windBearing","cloudCover","uvIndex","visibility.1","ozone","sunriseTime","sunsetTime","moonPhase","precipIntensityMax","uvIndexTime","temperatureMin","temperatureMinTime","temperatureMax","temperatureMaxTime","apparentTemperatureMin","apparentTemperatureMinTime","apparentTemperatureMax","apparentTemperatureMaxTime"]

    df = read_csv(url, names=header, low_memory=False)
    df = df.iloc[1:,:]
    df['prices'] = df['price']
    df = df.drop(columns=['windSpeed','latitude','longitude','product_id','price',"id","timestamp","datetime","timezone","apparentTemperature","humidity","windGust","windGustTime","visibility","temperatureHigh","temperatureHighTime","temperatureLow","temperatureLowTime","apparentTemperatureHigh","apparentTemperatureHighTime","apparentTemperatureLow","apparentTemperatureLowTime","icon","dewPoint","pressure","windBearing","cloudCover","uvIndex","visibility.1","ozone","sunriseTime","sunsetTime","moonPhase","precipIntensityMax","uvIndexTime","temperatureMin","temperatureMinTime","temperatureMax","temperatureMaxTime","apparentTemperatureMin","apparentTemperatureMinTime","apparentTemperatureMax","apparentTemperatureMaxTime"])

    df = df.dropna()

    convert_dict = {'hour': int, 'day': int, 'month': int, 'prices': float, 'distance': float,
                    'surge_multiplier': float, 'temperature': float, 'precipProbability': float, 'precipIntensity': float}
    df = df.astype(convert_dict)

    '''Different Plots, in this order'''
    #ride_vs_price(df) #maybe skip this one
    #hour_vs_price(df)
    #hour_vs_surge(df)
    #weather_vs_surge(df)
    #price_vs_location(df)

    price_predictor(df)
    #distances_only(df)