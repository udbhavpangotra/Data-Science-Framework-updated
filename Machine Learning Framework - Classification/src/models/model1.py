columns = ['Pclass',  'Age','Embarked','Parch','SibSp','Fare','CabinType','Ticket','SameFirstName']
cat_features = ['Pclass','Embarked','CabinType','Ticket',]

models_f = []
num_folds=9
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2021) # create folds 
X_train = train_female[columns]
y_train = train_female['Survived']
for n_fold, (train_idx, valid_idx) in enumerate (folds.split(X_train,  y_train)):
    train_X, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
    valid_X, valid_y = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
    dataset = Pool(train_X, train_y, cat_features)
    evalset = Pool(valid_X, valid_y, cat_features)
    model_female = CatBoostClassifier(
        task_type="GPU", 
        depth=7,
        max_ctr_complexity=5,
        #border_count=1024, 
        iterations=50000,
        od_wait=500,od_type='Iter',       
        #l2_leaf_reg=0.01,
        learning_rate=0.0035,
        min_data_in_leaf=3
    
        )
    model_female.fit(dataset, plot=False, verbose=500,eval_set=evalset)
    models_f.append(model_female)
    y_pred_female = model_female.predict(train_female[columns])
    print(metrics.accuracy_score(train_female['Survived'], y_pred_female))