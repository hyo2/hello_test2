from flask import Flask, request, jsonify
# from flask_restful import Resource, Api
# from flask_restful import reqparse
import sys
import json
# import manage

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

application = Flask(__name__)
# api = Api(application) # 이거 왜 한거임? ? ?

# CPU
device = torch.device('cpu')

# BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# 입력 데이터셋 정리
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

# 분류 모델
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=5, ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)
    
# BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay) (optimizer와 schedule 설정)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# 학습한 모델 불러오기
PATH = '/workspace/mood_api/model/'
model = torch.load(PATH + 'our_emotions_bert_model.pt', map_location = 'cpu') # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'our_emotions_bert_model_state_dict.pt'), map_location = 'cpu')  # state_dict를 불러 온 후, 모델에 저장

checkpoint = torch.load(PATH + 'our_emotions_bert_all.tar', map_location = 'cpu') # dict 불러오기
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0: # 분노
                test_eval.append("angry")
            elif np.argmax(logits) == 1: # 슬픔
                test_eval.append("sad")
            elif np.argmax(logits) == 2: # 중립
                test_eval.append("soso")
            elif np.argmax(logits) == 3: # 행복
                test_eval.append("happy")
            elif np.argmax(logits) == 4: # 즐거움
                test_eval.append("joy")

        return test_eval[0]

@application.route("/")
def hello():
    return "Hello goorm!"    
    
# GET // 밑에 GET, POST로 전부 되면 걍 없애자~
@application.route("/contents")  # get diary contents - 걍 requset.json으로 되는 거면 없애자
def get_contents(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('diary_id', type=int)
    parser.add_argument('content', type=str)
    args = parser.parse_args()
    
    diary_id = args['diary_id']
    content = args['content']
    
    return jsonify({'diary_id':diary_id, 'contents':content})

#POST
@application.route("/result", methods=['GET','POST']) # post diary mood
def process():
    # content 부분만 파싱해서 받기 되는지 찾기!
    diary = request.json  # 이쪽에서 파싱해서 가져올 수 있이면 위에 꺼 없애도 될 듯?
    diary_id = diary['diary_id']
    content = diary['contents']
    diary_mood = predict(content)
    return jsonify({'diary_id':diary_id, 'diary_mood':diary_mood}) # db 쪽에 물어보자,,,어떻게 받아올지 결정하기

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=int(sys.argv[1]))