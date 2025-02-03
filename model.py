import torch
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification, BertConfig

# BertModel 초기화
def BertModelInitialization():
    PATH = "model.pt"
    model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)

    torch.save(model.state_dict(), PATH)

def get_model(device):
    PATH = "model.pt"

    model = BertForSequenceClassification.from_pretrained('monologg/kobert')

    model.load_state_dict(torch.load(PATH)) # PATH에 저장된 모델을 불러오기
    model = model.to(device) # 불러온 모델을 device에 올리기

    return model

def get_model_with_params(num_data, device):
    model = get_model(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)
    epochs = 3

    total_steps = num_data * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    return model, optimizer, scheduler, epochs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BertModelInitialization()
    print(get_model_with_params(200000, device))

if __name__ == '__main__':
    main()
