import uvicorn
from fastapi import FastAPI
import pickle
from pydantic import BaseModel


class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float


app = FastAPI()
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {"message": "Bank Note Authentication"}


@app.post('/predict')
def predict_banknote(data: BankNote):
    data = data.dict()

    print(data)

    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    print(prediction)

    if prediction[0] == 1:
        return {'message': 'Fake Note'}
    elif prediction[0] == 0:
        return {'message': 'Original Note'}


if __name__ == "__main__":
    uvicorn.run(app)

# To Run Application paste the below query in terminal:
# uvicorn app:app --reload
