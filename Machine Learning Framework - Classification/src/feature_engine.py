from dataset import dataset_reading

train_df,test_df = dataset_reading(train_data_location,test_data_location)

ctypes = {
    'Survived':np.int8,
    'Pclass':np.int8,
    'Name':np.str,
    'Embarked':np.str,  
    'SibSp':np.int8,
    'Parch':np.int8,
}

           
train = train_df
test = test_df


    train['Embarked'] = train['Embarked'].fillna('No')
    test['Embarked'] = test['Embarked'].fillna('No')

    train['Cabin'] = train['Cabin'].fillna('_')
    test['Cabin'] = test['Cabin'].fillna('_')

    train.Ticket = train.Ticket.map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')
    test.Ticket = test.Ticket.map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')

    train['CabinType'] = train['Cabin'].apply(lambda x:x[0])
    test['CabinType'] = test['Cabin'].apply(lambda x:x[0])

    train['Age'].fillna(round(train['Age'].mean()), inplace=True,)
    test['Age'].fillna(round(test['Age'].mean()), inplace=True,)
    train['Age'] = train['Age'].apply(round)
    test['Age'] = test['Age'].apply(round)
    train['Age'] = train['Age'].astype(np.int8)
    test['Age'] = test['Age'].astype(np.int8)


    train['Fare'].fillna(round(train['Fare'].mean()), inplace=True,)
    test['Fare'].fillna(round(test['Fare'].mean()), inplace=True,)

    train['FirstName'] = train['Name'].apply(lambda x:x.split(', ')[0])
    train['SecondName'] = train['Name'].apply(lambda x:x.split(', ')[1])

    test['FirstName'] = test['Name'].apply(lambda x:x.split(', ')[0])
    test['SecondName'] = test['Name'].apply(lambda x:x.split(', ')[1])

    train['n'] = 1
    test['n'] = 1

    gb = train.groupby('FirstName')
    df_names = gb['n'].sum()
    train['SameFirstName'] = train['FirstName'].apply(lambda x:df_names[x])

    gb = test.groupby('FirstName')
    df_names = gb['n'].sum()
    test['SameFirstName'] = test['FirstName'].apply(lambda x:df_names[x])

    train['SameFirstName'] = train['SameFirstName'].apply(lambda x:-1 if x>10 else x)
    test['SameFirstName'] = test['SameFirstName'].apply(lambda x:-1 if x>10 else x)

    train_female = train[train.Sex=='female']
    train_male = train[train.Sex=='male']