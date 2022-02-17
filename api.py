

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class inputs(BaseModel):
    CNV: int
    RPPA: int
    miRNA: int 
    
##insert loading trained model

##loadedmodel = 


def survivability(data:inputs):
    recieved = data.dict()
    CNV = recieved['CNV']
    RPPA = recieved['RPPA']
    miRNA = recieved['miRNA']
    Result = loadedmodel.predict([[CNV, RPPA, miRNA]]).tolist()[0]
    
    return {'Prediction': Result}

