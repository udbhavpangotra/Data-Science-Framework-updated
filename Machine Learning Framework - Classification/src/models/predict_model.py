%%time
columns = ['Pclass',  'Age','Embarked','Parch','SibSp','Fare','CabinType','Ticket','SameFirstName']
m_columns_f = []
for idx,m in enumerate(models_f):
    new_column = 'fm_{}'.format(idx)
    m_columns_f.append(new_column)
    test[new_column] = m.predict(test[columns])
    print(new_column, end=' ')
print()
m_columns_m = []
columns = ['Pclass',  'Age','Embarked','Parch','SibSp','Fare','CabinType','Ticket']
for idx,m in enumerate(models_m):
    new_column = 'm_{}'.format(idx)
    m_columns_m.append(new_column)
    test[new_column] = m.predict(test[columns])
    print(new_column, end=' ')