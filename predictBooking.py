import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, svm, decomposition
from sklearn.metrics import accuracy_score, roc_auc_score
import datetime

def parseData(data):
	#datetime processing
	dates = pd.to_datetime(pd.Series(data['date_time']))
	retData = pd.DataFrame({'day':[i.day for i in dates]}) #start our df that will be returned
	retData['month'] = [i.month for i in dates]
	retData['year'] = [i.year for i in dates]
	retData['hour'] = [i.hour for i in dates]
	retData['minute'] = [i.minute for i in dates]
	retData['second'] = [i.second for i in dates]

	encoded = ['site_name', 'user_location_country', 'user_location_region', 'user_location_city','srch_destination_id', 'hotel_country', 'hotel_id']
	direct = ['user_location_latitude', 'user_location_longitude', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel','srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'prop_is_branded', 'prop_starrating', 'cnt', 'is_booking']
	oneHotEncoded = ['distance_band', 'hist_price_band', 'popularity_band']
	for column in encoded: #variables that need to be encoded
		retData[column] = data[column].astype('category').cat.codes

	for column in direct:
		retData[column] = data[column]

	for attribute in oneHotEncoded:
		encodedDF = pd.get_dummies(data[attribute])
		encodedDF.columns = [name + attribute for name in encodedDF.columns]
		retData = retData.join(encodedDF)

	dates = pd.to_datetime(pd.Series(data['srch_ci']))
	retData['day_ci'] = [i.day for i in dates] #start our df that will be returned
	retData['month_ci'] = [i.month for i in dates]
	retData['year_ci'] = [i.year for i in dates]

	dates = pd.to_datetime(pd.Series(data['srch_co']))
	retData['day_co'] = [i.day for i in dates] #start our df that will be returned
	retData['month_co'] = [i.month for i in dates]
	retData['year_co'] = [i.year for i in dates]
	retData = retData.dropna()

	return retData


df = pd.read_csv('ASADataFest2017Data/data.txt', sep='\t')
df = df.dropna() #remove rows with null values for now

X = parseData(df[0:1000])
Y = X['is_booking'] #getLabels

X = X.drop('is_booking',1)
X = pd.DataFrame(preprocessing.scale(X))

pca = decomposition.PCA(n_components=5)
pca.fit(X)
X = pca.transform(X)

dataSplit = int(len(X)*.8)


rf = RandomForestClassifier(1000)
rf.fit(X[:dataSplit],Y[:dataSplit])
predictions = rf.predict(X[dataSplit:])

print accuracy_score(Y[dataSplit:], predictions)
print roc_auc_score(Y[dataSplit:], predictions)

svm = svm.SVC(class_weight={1: 10})
svm.fit(X[:dataSplit],Y[:dataSplit])
predictions = svm.predict(X[dataSplit:])

print accuracy_score(Y[dataSplit:], predictions)
print roc_auc_score(Y[dataSplit:], predictions)


