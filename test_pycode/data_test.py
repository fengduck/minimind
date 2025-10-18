
import json
samples = []
# path = './dataset/dpo.jsonl'
# path = '../dataset/dpo.jsonl'
path = '../dataset/r1_mix_1024.jsonl'
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