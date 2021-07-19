def _s(r):
    if r.Sex=='male':
        return r.model_male
    else:
        return r.model_female
    
submission['Survived'] = test.apply(lambda x:_s(x),axis=1)

submission.to_csv('result.csv')