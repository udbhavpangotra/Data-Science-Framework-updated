from dataset import dataset_reading

train_df,test_df = dataset_reading(train_data_location,test_data_location)


def basic_eda(train_df, test_df):

    print('Rows and Columns in train dataset:', train_df.shape)
    print('Rows and Columns in test dataset:', test_df.shape)

    print('Missing values in train dataset:', sum(train_df.isnull().sum()))
    print('Missing values in test dataset:', sum(test_df.isnull().sum()))
    
    print('Missing values per columns in train dataset')
    for col in train_df.columns:
        temp_col = train_df[col].isnull().sum()
        print(f'{col}: {temp_col}')
    print()
    print('Missing values per columns in test dataset')
    for col in test_df.columns:
        temp_col = test_df[col].isnull().sum()
        print(f'{col}: {temp_col}')
        
        
    print('first five rows of train dataset')
    train_df.head()
    print('first five rows of test dataset')
    test_df.head()
    
    
    
