def vote(r, columns):
    ones = 0
    zeros = 0
    for i in columns:
        if r[i]==0:
            zeros+=1
        else:
            ones+=1
    if ones>zeros:
        return 1
    else:
        return 0

test['model_female'] = test.apply(lambda x:vote(x,m_columns_f),axis=1)
test['model_male'] = test.apply(lambda x:vote(x,m_columns_m),axis=1)