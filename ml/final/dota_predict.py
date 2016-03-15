import pandas

# Final examp script by Dmitry Ippolitov for coursera YandexDataSchool course "Introduction to machine learning".

# function to prepare data for analysis
def prepare_data(remove_categorial=False,heroes_2_bagofwords=False,scale=False,test=False):
  # load data
  if test==True:
    filename='features_test.csv'
  else:
    filename='features.csv'

  data = pandas.read_csv(filename, index_col='match_id')

  # load predicted value in column 'radiant_win'
  if test==False:
    y = data['radiant_win'].tolist()

  # remove game ending data (duration,radiant_win,tower_status_radiant, tower_status_dire, barracks_status_radiant, barracks_status_dire), last 6 columns
  if test==False:
    data = data.drop(data.columns[102:],axis=1)

  # remove timestamp
  data = data.drop(data.columns[0:1],axis=1)
  # replace NaN values with zeros
  data = data.fillna(0)

  # generate bag of words
  if heroes_2_bagofwords==True:
    import numpy as np
    X_pick = np.zeros((data.shape[0], 112))

    for i, match_id in enumerate(data.index):
      for p in xrange(5):
        X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

  # remove categorialdata
  if remove_categorial==True:
    categorial_cols = ['lobby_type']
    #append another cols2 categorial_colls
    for i in range(1,6): 
       categorial_cols+=["r%s_hero" % i, "d%s_hero" %  i]

    #drop categorial colls
    for col in categorial_cols: data=data.drop(col,axis=1)

  # scale if needed (we use this for logistic regression)
  if scale==True:
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_scaled = sc.fit_transform(data)
  

  # concat bagofwords with scaled data if needed
  if heroes_2_bagofwords==True and scale==True:
    X = np.hstack((X_scaled,X_pick))
  elif scale==True:
    X = X_scaled
  else: # gradient boosting don't need scaler
    X = data.values.tolist()

  if test==True: return X
  else: return X,y

# grid_search and cross_validation
def grid_search(X,y,est,grid):
  # initialize KFold cross validator
  from sklearn.cross_validation import KFold
  cv = KFold(len(y),5,shuffle=True)

  from sklearn.grid_search import GridSearchCV
  # user verbose=10 for measure time 
  grid_search = GridSearchCV(est,grid,cv=cv,verbose=10,scoring='roc_auc')

  # perform grid_search
  grid_search.fit(X,y)

  # print .grid_scores_
  from pprint import pprint
  pprint(grid_search.grid_scores_)

  return grid_search

# gradient boosting run
def gb_run():
  X,y = prepare_data()
  
  # initialize GradientBoostingClassifier
  from sklearn.ensemble import GradientBoostingClassifier
  est = GradientBoostingClassifier()
  grid = { 'n_estimators': [10,20,30] } 
  return grid_search(X,y,est,grid)

# logistic regression run
def log_reg_run(remove_categorial=False,heroes_2_bagofwords=False):
  X,y = prepare_data(remove_categorial=remove_categorial,heroes_2_bagofwords=heroes_2_bagofwords,scale=True)

  from sklearn.linear_model import LogisticRegression
  est = LogisticRegression(penalty='l2')
  grid = {'C': [0.0001,0.001, 0.01,0.1,1,10,100,1000] }

  return grid_search(X,y,est,grid)

# DEBUG FUNCTION: save grid_search result for futher usage
def save_est(filename,est):
  import pickle
  f = open(filename,'w')
  f.write(pickle.dumps(est))
  f.close()

def __main__():
  # run gradient boosting
  gb = gb_run() #; save_est('gb.clf',gb);
  # run log reg
  lr1 = log_reg_run() #; save_est('lr1.clf',lr1) 
  # run log reg without categorial
  lr2 = log_reg_run(remove_categorial=True) #; save_est('lr2.clf',lr2)
  # run log reg with bag of words
  lr3 = log_reg_run(remove_categorial=True,heroes_2_bagofwords=True) #; save_est('lr3.clf',lr3)

  # let's try to process test data with best_estimator
  est = lr3.best_estimator_
  X = prepare_data(remove_categorial=True,heroes_2_bagofwords=True,scale=True,test=True)
  result = est.predict_proba(X)

  # select maximal predict and minimal
  predict_radiant_win = [x[1] for x in result]
  print("max predict: %s\n" % max(predict_radiant_win))
  print("min predict: %s\n" % min(predict_radiant_win))


__main__()

