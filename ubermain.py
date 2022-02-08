import csv

import inline as inline
import matplotlib
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas
import geopandas
from geopandas import GeoDataFrame
import numpy as np
import descartes
from shapely.geometry import Point, Polygon
import seaborn as sns
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
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

#%matplotlib inline

def custom_round(x, base=5):
    return int(base * round(float(x)/base))

def haversine_vectorize(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    newlon = lon2 - lon1
    newlat = lat2 - lat1
    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2
    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist #6367 for distance in KM for miles use 3958
    return km

if __name__ == "__main__":
    url = "uber.csv"
    header = ['index', 'key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    #creates a dataframe
    dataset = read_csv(url, names=header)
    dataset = dataset.iloc[1:, 2:]
    dataset[['year','month', 'day']] = dataset.pickup_datetime.str.split("-", 2, expand=True)
    dataset[['day', 'time', 'timezone']] = dataset.day.str.split(" ", 2, expand=True)

    pandas.set_option('display.expand_frame_repr', False)
    #print(dataset.loc[[1]])
    '''
    dataset.boxplot(column=['fare_amount'], by=['year'])
    pyplot.show()
    '''
    pred = dataset.iloc[:, 0:11]
    pred = pred.dropna()
    pred[['hour','minutes']] = pred.time.str.split(":", 1, expand=True)
    convert_dict = {'fare_amount':float, 'pickup_longitude':float, 'pickup_latitude':float, 'dropoff_latitude':float, 'dropoff_longitude':float, 'passenger_count':int, 'year':int, 'month':int, 'day':int, 'hour':int}
    pred = pred.astype(convert_dict)
    pred['distance'] = haversine_vectorize(pred['pickup_longitude'],pred['pickup_latitude'],pred['dropoff_longitude'],pred['dropoff_latitude'])
    pred = pred.drop(columns=['pickup_datetime', 'time', 'minutes']) #'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'
    pred['fare_amount'] = pred['fare_amount'].apply(np.int64)
    pred['fare_amount'] = pred['fare_amount'].apply(lambda x: custom_round(x, base=5))

    crs = {'init': 'epsg:4326'}
    geometry = [Point(xy) for xy in zip(pred['pickup_longitude'],pred['pickup_latitude'])]
    gdf = geopandas.GeoDataFrame(pred, geometry=geometry)
    #print(pred.loc[[1]])

    country_map = geopandas.read_file('cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    gdf.plot(cmap='Blues', ax=country_map.plot())
    # add title to graph
    plt.title('Uber Fares Across the United States', fontsize = 15, fontweight ='bold')
    # set latitiude and longitude boundaries for map display
    plt.xlim(-175, -50)
    plt.ylim(25, 75)
    # show map
    plt.show()
    '''
    print(country_map.head())
    fig,ax=plt.subplots(figsize=(15,5))
    country_map.plot(ax=ax)
    plt.show()

    array = pred.values
    X = array[:, 1:]  # every row, only columns 0 to 3
    y = array[:, 0]  # every row, only column 4 (class, which is the answer)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    predictions = np.around(predictions/5, decimals=0)*5

    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))'''