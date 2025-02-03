
from preproc import preproc
from model import get_model, get_model_with_params, BertModelInitialization
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time
import os

# Sentiment classification 학습
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def accuracy(preds, labels):
    f_pred = np.argmax(preds, axis=1).flatten()
    f_labels = labels.flatten()
    return np.sum(f_pred == f_labels) / len(f_labels)

seed_val = 2022
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

from transformers import AutoTokenizer

whole_dataset = pd.read_csv('ratings.txt', delimiter="\t")

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

train_dataloader, validation_dataloader, test_dataloader = preproc(tokenizer, whole_dataset)

BertModelInitialization()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, optimizer, scheduler, epochs = get_model_with_params(len(train_dataloader), device)

model.zero_grad()

for epoch_i in range(epochs):
    print("")
    print('========{:}번째 Epoch / 전체 {:}회 ========'.format(epoch_i + 1, epochs))
    print('훈련 중')

    total_loss = 0 
    sum_loss = 0
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader)):

        if step % 50 == 0:
            print("{}번째 까지의 평균 loss : {}".format(step, sum_loss/50))
            sum_loss = 0

        batch = tuple(t.to(device) for t in batch)   
        b_input_ids, b_input_mask, b_labels = batch  

        outputs = model(b_input_ids,attention_mask=b_input_mask, labels = b_labels)

        loss = outputs[0]
        total_loss += loss.item() # 총 로스 계산
        sum_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 그래디언트 클리핑
        optimizer.step() 
        scheduler.step()
        model.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    #### Validaiton ####

    print("")
    print("검증 중")

    model.eval()

    eval_accuracy = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids,attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))


PATH = "model.pt"
torch.save(model.state_dict(), PATH)

print("")
print("Training complete!")

print("")
print("테스트 중")

model.eval()

eval_accuracy = 0
nb_eval_steps = 0

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids,attention_mask=b_input_mask)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))