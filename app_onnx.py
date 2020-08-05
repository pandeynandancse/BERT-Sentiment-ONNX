import config
import torch
import flask
import time
from flask import Flask
from flask import request
from model import BERTBaseUncased
import functools
import torch.nn as nn
#remove joblib for onnx
#import joblib
import onnxruntime as ort

app = Flask(__name__)


MODEL = ort.InferenceSession("model.onnx")


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    
    return tensor.cpu().numpy()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    #unsueeze(0) means add 1 more dimension at index 0 ,  so that batch size can be one via adding one more dimension
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    # type of ids,mask,token_type_ids is torch
    #but onnx does not accepts torch tensors 
    # so convert them into numpy arrays
    onnx_input = {
        'ids': to_numpy(ids),
        'mask':to_numpy(mask),
        'token_type_ids':to_numpy(token_type_ids)
    }

    output = MODEL.run(None,onnx_input) #some numerical value that may be greater than 1 
    #because outputs is two-dimensional with only one value i.e. [[outputs]] so use [0][0] to get actual value
    
    #so sigmoid it so that output value can be in range of (0,1)
    return sigmoid(output[0][0])


@app.route("/predict")
def predict():
    #parse the sentence from request
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    # sentence_prediction("nice video lecture")
    app.run(host='127.0.0.1',port = 9901)