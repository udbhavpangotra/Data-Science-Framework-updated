import __init__

def dataset_reading(train_data_location,test_data_location):
    
    import pandas as pd
    import numpy as np
    print('location provided for the test file' + loc_test)
    print('location provided for the train file' + loc_train)
    '''
    This python file is meant for reading your train and test raw files
    You can alter the location of the file in this py file as per your machine.

    There are sample codes for reading csv, json, reading from bigquery, reading from aws, reading from oracle as well
    You will need to provide the auth. code for them
    This will be the only file where you interact with the raw data! 
    You should consume the raw data in one go and then store it as a dataframe_object/ j5 file/ json document/ processed csv

    '''


    train = pd.read_csv(train_data_location,dtype=ctypes,index_col='PassengerId')
    test = pd.read_csv(test_data_location,dtype=ctypes,index_col='PassengerId')

    return train, test


# SAMPLE CODE FOR READING FROM BIGQUERY
"""
from google.cloud import bigquery

bqclient = bigquery.Client()

# Download query results.
query_string = """
# THIS SECTION NEEDS TO BE IN CODE NOT COMMENTS
"""
SELECT
CONCAT(
    'https://stackoverflow.com/questions/',
    CAST(id as STRING)) as url,
view_count
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE tags like '%google-bigquery%'
ORDER BY view_count DESC
"""
"""

dataframe = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # API is used by default.
        create_bqstorage_client=True,
    )
)
print(dataframe.head())

"""


# SAMPLE CODE FOR READING FROM AWS 

"""
!python -m pip install boto3 pandas "s3fs<=0.4"
!pip install boto3

import io
import os

import boto3
import pandas as pd


AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
)

books_df = pd.DataFrame(
    data={"Title": ["Book I", "Book II", "Book III"], "Price": [56.6, 59.87, 74.54]},
    columns=["Title", "Price"],
)


with io.StringIO() as csv_buffer:
    books_df.to_csv(csv_buffer, index=False)

    response = s3_client.put_object(
        Bucket=AWS_S3_BUCKET, Key="files/books.csv", Body=csv_buffer.getvalue()
    )

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 put_object response. Status - {status}")
    else:
        print(f"Unsuccessful S3 put_object response. Status - {status}")
        
"""



# Demo script for reading a CSV file from S3 into a pandas data frame using the boto3 library


"""
import os

import boto3
import pandas as pd


AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
)

response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key="files/books.csv")

status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

if status == 200:
    print(f"Successful S3 get_object response. Status - {status}")
    books_df = pd.read_csv(response.get("Body"))
    print(books_df)
else:
    print(f"Unsuccessful S3 get_object response. Status - {status}")

"""

# READING A JSON FILE
"""

import json
with open('E:/datasets/patients.json', 'w') as f:
    json.dump(patients, f)
    
with open('E:/datasets/cars.json', 'w') as f:
    json.dump(cars, f)  
    
"""    
