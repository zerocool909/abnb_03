# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:03:18 2019

@author: adraj
"""


import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# Point to data
path = os.getcwd() +'\\Airbnb\\'
train_data = pd.read_csv(path + 'train_airbnb_2905.csv', header = 0, encoding='latin1') #define encoding type to match output from excel
test_data = pd.read_csv(path + 'test_airbnb_2905.csv', header = 0, encoding='latin1') #define encoding type to match output from excel
sentiment_data = pd.read_csv(path + 'description_sentiment_all_data.csv', header = 0, encoding='latin1')
#polarity_data = pd.read_csv(path + 'name_sentiment_all_data.csv', header = 0, encoding='latin1')

train_data = train_data[(train_data['log_price'] != 0)]

ntrain = len(train_data)
ntest = len(test_data)

data = pd.concat([train_data, test_data],sort=False,ignore_index=True)
data = pd.merge(data,sentiment_data[['id','vader_pos']], on='id', how = 'left')
#data = pd.merge(data,polarity_data[['id','polarity_textblob']], on='id', how = 'left')

df= data.copy()

#position of AirBnB property

def lat_center(row):
    if (row['city']=='NYC'):
            lat= 40.72
    if(row['city']=='LA'):
            lat= 34.0522
    if(row['city']=='SF'):
            lat = 37.7749
    if(row['city']=='DC'):
            lat = 38.9072
    if(row['city']=='Chicago'):
            lat = 41.8781 
    if(row['city']=='Boston'):
            lat = 42.3601  
    return lat        
    
    
def long_center(row):
    if (row['city']=='NYC'):
            long = -74
    if(row['city']=='LA'):
            long = -118.2437
    if(row['city']=='SF'):
            long = -122.4194
    if(row['city']=='DC'):
            long = -77.0369
    if(row['city']=='Chicago'):
            long= -87.6298 
    if(row['city']=='Boston'):
            long= -71.0589 
    return long        

df['lat_center']=df.apply(lambda row: lat_center(row), axis=1)
df['long_center']=df.apply(lambda row: long_center(row), axis=1)
df['distance_from_city_center']=np.sqrt((df['lat_center']-df['latitude'])**2+(df['long_center']-df['longitude'])**2)


#divide cities into clusters
def find_clusters(df_city):
    X=df_city[['latitude','longitude']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=.123, min_samples = 4)
    clusters = dbscan.fit_predict(X_scaled)
    return clusters
# remove nans while merging the dataframes
def remove_nan_values(string):
        string_values = string.split(',')
        values = len(string_values)
        final_string = ""
        for i in range(values):
            if string_values[i]!='nan':
                final_string = string_values[i]
                break
        return  final_string 

df_city_Chicago = df[df.city=='Chicago']
df_city_LA = df[df.city=='LA']
df_city_SF = df[df.city=='SF']
df_city_NYC = df[df.city=='NYC']
df_city_DC = df[df.city=='DC']
df_city_Boston = df[df.city=='Boston']

df_city_Chicago['cords'] = find_clusters(df_city_Chicago)
df_city_Chicago['cords'] = 'city_chicago_cluster_' + df_city_Chicago['cords'].astype(str)
df_city_Boston['cords'] = find_clusters(df_city_Boston)
df_city_Boston['cords'] = 'city_boston_cluster_' + df_city_Boston['cords'].astype(str)
df_city_LA['cords'] = find_clusters(df_city_LA)
df_city_LA['cords'] = 'city_la_cluster_' + df_city_LA['cords'].astype(str)
df_city_SF['cords'] = find_clusters(df_city_SF)
df_city_SF['cords'] = 'city_sf_cluster_' + df_city_SF['cords'].astype(str)
df_city_NYC['cords'] = find_clusters(df_city_NYC)
df_city_NYC['cords'] = 'city_nyc_cluster_' + df_city_NYC['cords'].astype(str)
df_city_DC['cords'] = find_clusters(df_city_DC)
df_city_DC['cords'] = 'city_dc_cluster_' + df_city_DC['cords'].astype(str)

data_city_cluster = pd.merge(df,df_city_DC[['id','cords']], on='id', how = 'left')
data_city_cluster = pd.merge(data_city_cluster,df_city_NYC[['id','cords']], on='id', how = 'left')
data_city_cluster = pd.merge(data_city_cluster,df_city_SF[['id','cords']], on='id', how = 'left')
data_city_cluster = pd.merge(data_city_cluster,df_city_LA[['id','cords']], on='id', how = 'left')
data_city_cluster = pd.merge(data_city_cluster,df_city_Boston[['id','cords']], on='id', how = 'left')
data_city_cluster = pd.merge(data_city_cluster,df_city_Chicago[['id','cords']], on='id', how = 'left')

all_city_cluster = data_city_cluster.iloc[:,-6:]
all_city_cluster['cluster_data'] = all_city_cluster.iloc[:,0:6].apply(lambda x: ",".join(x.astype(str)), axis=1)
all_city_cluster['clean_cluster_data'] = all_city_cluster['cluster_data'].apply(lambda x:remove_nan_values(x))

df['city_cluster'] = all_city_cluster['clean_cluster_data']  
 

#Amenities available

def extract_list_val(s):
    for c in ['{','}','"']:
        s=s.replace(c,'')
    for c in [':','-','.','&','\'']:
        s=s.replace(c,'')
    s=s.replace('matress','mattress')
    return s.split(',')
        
df['amenities']=df.apply(lambda x: extract_list_val(x.amenities), axis=1)


def replace_missing_text(string):
    total_list=[]
    miss = 'missing'
    for i in range (len(string)):
        if miss not in string[i]:
            total_list.append(string[i])            
    return ( ','.join(total_list) )   

df['amenities'] = df.apply(lambda x:replace_missing_text(x.amenities),axis=1)
 
def count_amenities(string):
    total_list=string.split(',')
    return len(total_list)
           
df['amenities_count'] = df.apply(lambda x:count_amenities(x.amenities),axis=1)



tfidf_vec = TfidfVectorizer(analyzer='word', stop_words='english' ,strip_accents='ascii')
tfidf_dense = tfidf_vec.fit_transform(df['amenities']).todense()
new_cols = tfidf_vec.get_feature_names()
# remove the text column as the word 'text' may exist in the words and you'll get an error
df = df.drop('amenities',axis=1)
# join the tfidf values to the existing dataframe
df = df.join(pd.DataFrame(tfidf_dense, columns=new_cols))

##property type        

property_dictionary = {'Apartment':['Condominium','Timeshare','Loft','Serviced apartment','Guest suite'],
         'House':['Vacation home','Villa','Townhouse','In-law','Casa particular'],
         'HotelType1':['Dorm','Hostel','Guesthouse'],
         'HotelType2':['Boutique hotel','Bed & Breakfast'],
         'Other':['Island','Castle','Yurt','Hut','Chalet','Treehouse',
                  'Earth House','Tipi','Cave','Train','Parking Space','Lighthouse',
                 'Tent','Boat','Cabin','Camper/RV','Bungalow']
        }
properties = {i : k for k, v in property_dictionary.items() for i in v}
df['property_type']= df['property_type'].replace(properties) 


#cancellation_policy
cancellation_policy_dictionary = {'strict':['super_strict_30','super_strict_60','long_term']}
cancellation = {i : k for k, v in cancellation_policy_dictionary.items() for i in v}
df['cancellation_policy']= df['cancellation_policy'].replace(cancellation) 


def bool_to_int(s):
    if s=='t':
        return 1
    else: 
        return 0

def remove_percentage_sign(s):
    if pd.isnull(s)==False:
        return float(s.replace('%',''))
    else: 
        return s    

df['host_response_rate']=df.apply(lambda x: remove_percentage_sign(x.host_response_rate), axis=1)
df['host_response_rate'].fillna(df['host_response_rate'].mode()[0], inplace=True)

df['neighbourhood'].fillna("city_neighbour_df_"+df['city'], inplace=True)    
all_cals = df['neighbourhood'].value_counts()
df['instant_bookable'] = df.apply(lambda x: bool_to_int(x.instant_bookable), axis=1)
df['cleaning_fee'] = df['cleaning_fee'].astype(int)
df['cleaning_fee'].value_counts()
df['room_type']=df['room_type'].str.replace(' ','_')
df['bed_type']=df['bed_type'].str.replace(' ','_')

#test dummy encoding for categorical values
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    print('Encoding Text Dummy Complete for {}'.format(name)) 
    
encode_text_dummy(df, 'property_type')
encode_text_dummy(df, 'room_type')
encode_text_dummy(df, 'bed_type')
encode_text_dummy(df, 'cancellation_policy')
encode_text_dummy(df, 'city')
encode_text_dummy(df, 'city_cluster')



df['review_scores_rating'].fillna(df['review_scores_rating'].mode()[0],inplace=True)
df['number_of_reviews'].fillna(df['number_of_reviews'].mode()[0], inplace=True)

#timeline data
df['host_since'] = pd.to_datetime(df.host_since, format='%m/%d/%Y' ,errors='coerce')
df['first_review'] = pd.to_datetime(df.first_review, format='%m/%d/%Y',errors='coerce')
df['last_review'] = pd.to_datetime(df.last_review, format='%m/%d/%Y',errors='coerce')

df['host_since'] = df['host_since'].fillna(df['first_review'])
df['host_since'] = df['host_since'].fillna(df['last_review']) 
df['host_since'] = df['host_since'].fillna('2017-01-01 00:00:00')  

df['host_since_year'] = pd.to_datetime(df.host_since, format='%Y-%m-%d %H:%M:%S').dt.year
df['years_being_host'] = 2017 - df['host_since_year']

#delete columns to be not used by our model 
del df['id']
del df['description']
del df['first_review']
del df['host_has_profile_pic']
del df['host_identity_verified']
del df['host_since']
del df['last_review']
del df['thumbnail_url']
del df['zipcode']
del df['lat_center']
del df['long_center']
del df['host_since_year']
del df['name']
del df['bathrooms']
del df['bedrooms']
del df['beds']
del df['amenities_count']
del df['neighbourhood']


#dividing df into test and train set
train_final =df[:ntrain]
test_final=df[ntrain:]
del test_final['log_price']

#null_columns=train_final.columns[train_final.isnull().any()]
#train_final[null_columns].isnull().sum()

X = train_final.copy()
y = X.pop('log_price')

#train test split
train, test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

train_data = lgb.Dataset(train, label=y_train)
test_data = lgb.Dataset(test, label=y_test)

#parameters assigned post grid-search on model
param = {'objective': 'regression',
         'boosting': 'gbdt',  
         'metric': 'root_mean_squared_error',
         'learning_rate': 0.08, 
         'num_iterations': 500,
         'num_leaves': 45,#35
         'max_depth': 10,
         'min_data_in_leaf': 18,#12
         'bagging_fraction': 0.85,
         'bagging_freq': 10,#1
         'feature_fraction': 0.7
         }

lgbm = lgb.train(params=param,
                 verbose_eval=50,
                 train_set=train_data,
                 valid_sets=[test_data])

y_pred_lgbm = lgbm.predict(test)
print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, y_pred_lgbm)), 4)))


##feature importance
feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importance(importance_type='gain'),X.columns)), columns=['Value','Feature'])
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:20])
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

##final prediction on test set
final_pred = lgbm.predict(test_final)
pd.DataFrame(final_pred, columns=['log_price']).to_csv('Airbnb_prediction_lgb_clusters.csv')

########################
####################Models not selected due to low RMSE Score######################
########################


###Model selection with cross validation
####################################################


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# Base models
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3 ))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=30, max_features='sqrt',
                                   min_samples_leaf=45, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=30, 
                             min_child_weight=1.7817, n_estimators=500,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=150,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.23,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =18, min_sum_hessian_in_leaf = 11)



score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv (GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

model_lgb.fit(train, y_train)
lgb_pred =model_lgb.predict(test_final.values)
pd.DataFrame(lgb_pred, columns=['log_price']).to_csv('Airbnb_prediction_lgbcv_sa.csv')

###################################
############Catboost###############
###################################

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=0)
model=CatBoostRegressor(iterations=1500, depth=7, learning_rate=0.08, loss_function='RMSE')
model.fit(X_train, y_train,eval_set=(X_valid, y_valid),plot=True);





