import pandas as pd
from sklearn.ensemble import *
from sklearn import preprocessing, svm, decomposition
from sklearn.metrics import accuracy_score, roc_auc_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import *
from sklearn.utils import shuffle
from dateutil import parser
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

import matplotlib.pyplot as plt
import numpy as np
import datetime
import multiprocessing

def parseData(inputData):

	data = inputData.as_matrix()
	#labels used to name the Pandas DF after we are done
	labels = ['day', 'month', 'year', 'hour', 'minute', 'second', 'srch_ci_day', 'srch_ci_month', 'srch_ci_year', 'srch_co_day', 'srch_co_month', 'srch_co_year', 'user_location_country', 'user_location_region', 'user_location_city', 'site_name','srch_destination_id', 'hotel_country', 'hotel_id', 'srch_destination_name', 'user_location_latitude', 'user_location_longitude', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel','srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'prop_is_branded', 'prop_starrating', 'cnt', 'is_booking', 'srch_destination_latitude', 'srch_destination_longitude'] 
	
	#non-numeric attributes that we will encode
	encoded = [ 'user_location_country', 'user_location_region', 'user_location_city', 'site_name','srch_destination_id', 'hotel_country', 'hotel_id', 'srch_destination_name']
	# set of unique values for each encoded value
	encodedValues = list(map((lambda inputs : list(set( data[:,list(inputData).index(inputs)]))), encoded)) 

	#attributes that we are going to add directly without any extra parsing
	direct = ['user_location_latitude', 'user_location_longitude', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel','srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'prop_is_branded', 'prop_starrating', 'cnt', 'is_booking', 'srch_destination_latitude', 'srch_destination_longitude']
	
	#the popular_* attributes that we can directly add
	#only reason i didnt put this with the other direct adds is that I didnt want to enumerate them in labels
	popularAttributes = [x for x in list(inputData) if 'popular_' in x]
	labels.extend(popularAttributes)

	#attributes that we will need dummy variables for
	oneHotEncoded = ['distance_band', 'hist_price_band', 'popularity_band', 'srch_destination_type_id']
	#how many different values are in each one hot attibute
	oneHotEncodedValues =  list(map((lambda inputs : list(set( data[:,list(inputData).index(inputs)]))), oneHotEncoded)) 
	for iterator, column in enumerate(oneHotEncoded):
		#oneHotEncoded words may have duplicate values, we dont want to set these as repeating column values
		encodedWordArray = [oneHotEncoded[iterator]]*len(oneHotEncodedValues[iterator])
		possibleValues = oneHotEncodedValues[iterator]
		newColNames = [str(m)+ '_' + str(n) for m,n in zip(encodedWordArray,possibleValues)]
		labels.extend(newColNames)

	parsedDF = []
	#adding a column at a time makes it easier to skip messy data
	def generateDF(data):
		for i in range(0, len(data)): 

			if len(parsedDF)%100 == 0:
				print(len(parsedDF))

			example = []
			#check for bad values as we go along, if there are any, just throw the row away
			#need to also add the values in the same order as our labels

			try:
				searchDate = parser.parse(data[i][list(inputData).index('date_time')])
			except:
				print('search date parsing error in example', i, '.Skipping this example.')
				continue
			example.append(searchDate.day)
			example.append(searchDate.month)
			example.append(searchDate.year)
			example.append(searchDate.hour)
			example.append(searchDate.minute)
			example.append(searchDate.second)

			try:
				checkInDate = parser.parse(data[i][list(inputData).index('srch_ci')])
			except:
				print('check in date parsing error in example', i, '.Skipping this example.')
				continue
			example.append(checkInDate.day)
			example.append(checkInDate.month)
			example.append(checkInDate.year)

			try:
				checkOutDate = parser.parse(data[i][list(inputData).index('srch_co')])
			except:
				print('check out date parsing error in example', i, '.Skipping this example.')
				continue
			example.append(checkOutDate.day)
			example.append(checkOutDate.month)
			example.append(checkOutDate.year)


			#access encoded indicies and use the index of each encoded attribute as the value
			for iterator,columns in enumerate(encoded):
				example.append(encodedValues[iterator].index(data[i][list(inputData).index(columns)]))

			#add the direct attributes
			for columns in direct:
				example.append(data[i][list(inputData).index(columns)])

			#add the popular attributes
			for columns in popularAttributes:
				example.append(data[i][list(inputData).index(columns)])

			#add the one hot attributes
			for iterator,columns in enumerate(oneHotEncoded):
				#intialize the array that we will extend to our example
				oneHotArray = np.zeros(len(oneHotEncodedValues[iterator]))
				#use the position in oneHotEncodedValues and the value we have for the oneHotEncodedVariable as the index that we set
				oneHotArray[oneHotEncodedValues[iterator].index(data[i][list(inputData).index(columns)])] = 1			
				example.extend(oneHotArray)

			parsedDF.append(example)

	threads = multiprocessing.cpu_count()
	pool = ThreadPool(4)
	dfPieces = [data[i:i + int(len(data)/threads)] for i in range(0, len(data), int(len(data)/threads))]
	results = pool.map(generateDF, dfPieces)

	parsedDF = pd.DataFrame(parsedDF, columns = labels)
	parsedDF.dropna()
	return parsedDF


# df = pd.read_csv('ASADataFest2017Data/data.txt', sep='\t')
df = pd.read_csv('combined_small.txt', sep='\t')

#remove rows with null values for now
df = df.dropna()
df = shuffle(df)

df = df.reset_index()
X_raw = parseData(df)
# print(X_raw)
Y = X_raw['is_booking'] #getLabels

X_raw = X_raw.drop('is_booking',1)
X_raw = pd.DataFrame(preprocessing.scale(X_raw))
X = X_raw

pca = decomposition.PCA(n_components=3)
pca.fit(X_raw)
X = pca.transform(X_raw)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Y = Y.tolist()

ax.scatter(X[:,0], X[:,1], X[:,2])
# for i in range(0, len(X)):
# 	# print(i)
# 	if Y[i] == 1:
# 		ax.scatter(X[i,0], X[i,1], X[i,2], c = 'r')
# 	else:
# 		ax.scatter(X[i,0], X[i,1], X[i,2], c = 'b')

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

plt.show()

# dataSplit = int(len(X)*.8)

# rf = AffinityPropagation(preference=-50)
# rf.fit(X[:dataSplit],Y[:dataSplit])
# predictions = rf.predict(X[dataSplit:])

# print(accuracy_score(Y[dataSplit:], predictions))
# print(roc_auc_score(Y[dataSplit:], predictions))




