import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import traceback

# preprocessing용 함수 모음. 실행시 잘 구현됐는지 확인 가능.

def data_processing(raw_data):
    transformed_data = raw_data['label']
    processed_data = pd.concat([raw_data['document'], transformed_data], axis=1)

    processed_data.columns = ['sentence', 'label']

    return processed_data

def data_to_token_ids(tokenizer, single_sentence):
    special_token_added = "[CLS] " + str(single_sentence) + " [SEP]"

    tokenized_sentence = tokenizer.tokenize(special_token_added)

    token_ids = [tokenizer.convert_tokens_to_ids(tokenized_sentence)]

    MAX_LEN = 128 # padding
    token_ids_padded = pad_sequences(token_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    token_ids_flatten = token_ids_padded.flatten()
    return token_ids_flatten

def token_ids_to_mask(token_ids):

    mask = [float(i>0) for i in token_ids] # 0보다 큰 숫자만 유효.

    return mask

def tokenize_processed_data(tokenizer, processed_dataset):
    labels = processed_dataset['label'].to_numpy()

    # processed_dataset의 'sentence' 데이터를 id 리스트로 토큰화
    tokenized_data = [data_to_token_ids(tokenizer, processed_data) for processed_data in processed_dataset['sentence']]

    # 토큰화한 id 리스트 각각을 mask로 변환
    attention_masks = [token_ids_to_mask(token_ids) for token_ids in tokenized_data]

    return tokenized_data, labels, attention_masks

def split_into_train_test(whole_data, whole_label, whole_masks):
    print("length of whole_data : " + str(len(whole_data)))
    # data split : train/test
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(whole_data, whole_label, random_state=2022, test_size=0.1)

    # mask split
    train_masks, test_masks, _, _ = train_test_split(whole_masks, whole_data, random_state=2022, test_size=0.1)

    return train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks

def split_into_train_validation(train_data, train_label, train_masks):
    print("length of train_data : " + str(len(train_data)))

    # data split : train/val
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_data, train_label,  random_state=2022,  test_size=0.1)

    # mask_split
    train_masks, validation_masks, _, _ = train_test_split(train_masks,  train_data,  random_state=2022,  test_size=0.1)

    return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks

def data_to_tensor(inputs, labels, masks):
    inputs_tensor = torch.tensor(inputs)
    labels_tensor = torch.tensor(labels)
    masks_tensor = torch.tensor(masks)
    return inputs_tensor, labels_tensor, masks_tensor

def tensor_to_dataloader(inputs, labels, masks, mode):
    from torch.utils.data import RandomSampler, SequentialSampler

    batch_size=32
    data = TensorDataset(inputs, masks, labels)

    if mode == "train":
        sampler = RandomSampler(data) # mini-batch 내부 구성이 다양할수록 전체 dataset(모집단)를 잘 대표
    else:
        sampler = SequentialSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def preproc(tokenizer, whole_dataset):
    # whole_dataset을 전처리
    processed_dataset = data_processing(whole_dataset)

    # 전처리한 전체 데이터를 토큰화
    tokenized_dataset, labels, attention_masks = tokenize_processed_data(tokenizer, processed_dataset)

    # 토큰화한 전체 데이터를 train용과 test용으로 분리
    train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = split_into_train_test(tokenized_dataset, labels, attention_masks)
    # 토큰화한 train용 데이터를 train용과 validation용으로 분리
    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = split_into_train_validation(train_inputs, train_labels, train_masks)

    # train용, validation용, test용 데이터 각각을 텐서로 변환
    train_inputs, train_labels, train_masks = data_to_tensor(train_inputs, train_labels, train_masks)
    validation_inputs, validation_labels, validation_masks = data_to_tensor(validation_inputs, validation_labels, validation_masks)
    test_inputs, test_labels, test_masks = data_to_tensor(test_inputs, test_labels, test_masks)

    # train용, validation용, test용 텐서를 dataloader로 변환
    train_dataloader = tensor_to_dataloader(train_inputs, train_labels, train_masks, "train")
    validation_dataloader = tensor_to_dataloader(validation_inputs, validation_labels, validation_masks, "validation")
    test_dataloader = tensor_to_dataloader(test_inputs, test_labels, test_masks, "test")

    return train_dataloader, validation_dataloader, test_dataloader

def test2(tokenized_data):
    real_data = [2, 1706, 6664, 5729, 6983,  517, 7990, 6493, 7828, 5943, 4928, 1861, 5783, 2235,
                 6527,   54, 7227, 6160, 3010, 6559, 7828, 2846, 7095, 3394, 6946,   54, 5782, 6150,
                 3093, 6653, 7010, 5384, 3647, 2846, 6116, 4147, 6441,  517, 5693, 5693, 7828, 4768,
                 5330,  743, 5451, 6903, 4147, 7869, 6198, 4102, 2034, 7170, 7792, 4709, 7879, 7328,
                 54, 1185, 6049, 5782, 5439, 5007, 3647, 2680, 5330, 3135, 7271, 5782, 5760, 5384,
                 1861, 3100,   54, 1569, 4196, 3093, 6653, 7013, 2571,   54,    3,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0]
    return (tokenized_data[2022] == real_data).all()


def test3(train_inputs, validation_inputs, test_masks):
    if len(train_inputs) != 162000 or len(validation_inputs) != 18000 or len(test_masks) != 20000:
        return False
    real_data = [2, 3765, 6954, 4207, 7850, 4446, 6395, 5761, 4102, 3977, 6881, 6701,   54, 2368,
                 517, 7265, 6827, 6701,   54,    3,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0]
    return (train_inputs[2022] == real_data).all()


def test4(train_inputs):
    real_input = torch.tensor([2, 3301, 6553, 6410,  517, 6193, 7591, 4179, 6141, 6255, 4244, 5439,
                               4012,  517, 6193, 7591, 1370, 5347, 5782, 5330, 2573, 6844, 7495, 1844,
                               6190, 1734, 6978, 7968, 7720, 7086,  517, 6193, 7591, 4179, 7788,  517,
                               6394, 5833, 6141, 7318, 6149, 7086, 3524, 7227, 5859, 7136, 5546, 5850,
                               2034, 7170, 7095, 1369, 5760, 1420,   55,    3,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,    0])

    return torch.equal(train_inputs[12345], real_input)


def test5(train_dataloader):
    real_input_ids = torch.tensor([2,  529,   54, 2860, 6295, 7640, 5371, 3594, 7837,  553,   54,  773,
                                   6383, 7095, 5037, 6645, 7837, 4501, 5957, 6629, 7288, 3714, 7207, 5357,
                                   589,   54, 2417, 5398, 6882, 3357,  631,  529, 7220,    3,    0,    0,
                                   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                   0,    0,    0,    0,    0,    0,    0,    0])

    for step, batch in enumerate(tqdm(train_dataloader)):

        if step < 1234:
            continue
        if step > 1234:
            break

        b_input_ids, b_input_mask, b_labels = batch

    return torch.equal(b_input_ids[5], real_input_ids)

def preproc_test(tokenizer, whole_dataset):

    print("================={}번째 테스트 시작===================".format(1))
    # whole_dataset을 전처리
    try:
      processed_dataset = data_processing(whole_dataset)
    except:
      print(traceback.format_exc())
      return 0
    print("================={}번째 테스트 성공===================\n".format(1))


    print("================={}번째 테스트 시작===================".format(2))
    # 전처리한 전체 데이터를 토큰화
    try:
      tokenized_dataset, labels, attention_masks = tokenize_processed_data(tokenizer, processed_dataset)
    except:
      print(traceback.format_exc())
      return 20
    if not test2(tokenized_dataset):
      return 20
    print("================={}번째 테스트 성공===================\n".format(2))


    print("================={}번째 테스트 시작===================".format(3))
    # 토큰화한 전체 데이터를 train용과 test용으로 분리
    try:
      train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = split_into_train_test(tokenized_dataset, labels, attention_masks)
      # 토큰화한 train용 데이터를 train용과 validation용으로 분리
      train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = split_into_train_validation(train_inputs, train_labels, train_masks)
    except:
      print(traceback.format_exc())
      return 40
    if not test3(train_inputs, validation_inputs, test_masks):
      return 40
    print("================={}번째 테스트 성공===================\n".format(3))


    print("================={}번째 테스트 시작===================".format(4))
    # train용, validation용, test용 데이터 각각을 텐서로 변환
    try:
      train_inputs, train_labels, train_masks = data_to_tensor(train_inputs, train_labels, train_masks)
      validation_inputs, validation_labels, validation_masks = data_to_tensor(validation_inputs, validation_labels, validation_masks)
      test_inputs, test_labels, test_masks = data_to_tensor(test_inputs, test_labels, test_masks)
    except:
      print(traceback.format_exc())
      return 60
    if not test4(train_inputs):
      return 60
    print("================={}번째 테스트 성공===================\n".format(4))


    print("================={}번째 테스트 시작===================".format(5))
    # train용, validation용, test용 텐서를 dataloader로 변환
    try:
      train_dataloader = tensor_to_dataloader(train_inputs, train_labels, train_masks, "train")
      validation_dataloader = tensor_to_dataloader(validation_inputs, validation_labels, validation_masks, "validation")
      test_dataloader = tensor_to_dataloader(test_inputs, test_labels, test_masks, "test")
    except:
      print(traceback.format_exc())
      return 80
    if not test5(train_dataloader):
      return 80
    print("================={}번째 테스트 성공===================\n".format(5))


    return 100

def main():
    from transformers import AutoTokenizer
    whole_dataset = pd.read_csv('ratings.txt', delimiter="\t")

    # KoBERTTokenizer를 불러옴 <- 기존의 tokenization.py 말고 AutoTokenizer 사용. (기존 코드 미지원)
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

    score = preproc_test(tokenizer, whole_dataset)
    print("현재 점수 : {}/100점".format(score))

if __name__ == '__main__':

    # 시드 고정
    seed_val = 2022
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    main()