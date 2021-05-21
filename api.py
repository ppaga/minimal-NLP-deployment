from fastapi import Depends, FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Pipeline


app = FastAPI()

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline_classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def get_classifier():
    return pipeline_classifier


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    probability: float


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest,
            classifier: Pipeline = Depends(get_classifier)):
    result_dict = classifier(request.text)
    sentiment, probability = result_dict[0]['label'], result_dict[0]['score']
    return SentimentResponse(
        sentiment=sentiment, probability=probability
    )
