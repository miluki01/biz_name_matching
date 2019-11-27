import torch
from torch.autograd import Variable
import torch.nn.functional as F
from flask import Flask, request, jsonify
import sys
import os

from models import NameMatchingModel
from inference import string2vec
from utils import parse_config

CONFIG_PATH = os.path.join('config.yml')

app = Flask(__name__)
config = parse_config(CONFIG_PATH)

try:
    with open(os.path.join('data', config['data_path']['common_words_path']), 'r') as f:
        common_words = [word.strip() for word in f.readlines()]

except IOError:
    sys.exit('stopword file is not found')
try:
    model = NameMatchingModel()
    model.load_state_dict(torch.load(os.path.join('saved_models', config['data_path']['model_path'])))
except IOError:
    sys.exit('Cant find model: {}'.format(config['data_path']['model_path']))

@app.route("/api/name_matching/v2/", methods = ['POST'])
def neural_matching():

    content = request.get_json()

    print(content)

    result = apply_query(content)

    return jsonify(result)

def score(s1, s2):
    with torch.no_grad():
        vector = string2vec(s1, s2, common_words)
        tensor_vector = Variable(torch.FloatTensor(vector), requires_grad=True)

        result = model(tensor_vector)
        score = (torch.exp(result) * 100).type(torch.IntTensor).numpy()[0].tolist()[1]

    return score

def apply_query(json_):

    for query in json_:
        keyword = query['keyword']
        for company in query['company']:
            company['score'] = score(company['name'], keyword)

    return json_

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8000')
    # print(config['data_path']['common_words_path'])
