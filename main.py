import torch

from preproc import data_to_token_ids, token_ids_to_mask
from model import get_model
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(device)

while True:
    input_str = input()

    # q를 입력하면 while 루프 끝
    if input_str == "q":
        break

    # sentence 2 token
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    tokenized_data = [data_to_token_ids(tokenizer, input_str)]
    attention_masks = [token_ids_to_mask(token_ids) for token_ids in tokenized_data]

    # token 2 tensor
    inputs_tensor = torch.tensor(tokenized_data).to(device)
    masks_tensor = torch.tensor(attention_masks).to(device)

    outputs = model.forward(input_ids = inputs_tensor,attention_mask = masks_tensor)

    logits = outputs[0][0]

    negpos = logits.detach().cpu().numpy()

    if negpos[0] > negpos[1]:
        print("당신이 입력한 문장은 부정입니다.\n")

    else: print("당신이 입력한 문장은 긍정입니다.\n")