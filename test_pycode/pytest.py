print(123)
print("HelloWorld")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('lora_name')
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('model_size')
parser.add_argument('--use_wandb', action="store_true")

args = parser.parse_args()

if args.use_wandb:
    print('使用线上wandb进行监控')
else:
    print('不使用线上wandb进行监控')

print(args)
print('-----------------')


from transformers import AutoTokenizer
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么我可以帮你的吗？"},
    {"role": "user", "content": "你知道地球为什么是圆的吗？"}   #  如果最后一个对话时user 就会自动补上 <|im_start|>assistant
]
tokenizer = AutoTokenizer.from_pretrained('../model')
text = tokenizer.apply_chat_template(messages, tokenize=False)

print(text)
print('-----------------')
messages = [
    {"role": "user", "content": "你好"}
]
new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # 这个参数没啥作用，取决于最后一条message是否是user
        )

print(new_prompt)



import json
samples = []
path = '../dataset/pretrain_hq.jsonl'
# path = './dataset/sft_mini_512.jsonl'
# path = './dataset/dpo.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        # print(f'line_num = {line_num}, line = {line}')
        data = json.loads(line.strip())
        # print(f'data = {data}')
        samples.append(data)
        # print(f'samples = {samples}')
        if line_num == 32:
            break
for conv in samples:
    print(conv)
    break
# print(samples)