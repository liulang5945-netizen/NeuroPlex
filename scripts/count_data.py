"""快速统计 zh_texts.jsonl 的行数和字符数。"""
f = open('data/corpus/zh_texts.jsonl', 'r', encoding='utf-8')
n = 0
chars = 0
non_empty = 0
while True:
    line = f.readline()
    if not line:
        break
    n += 1
    s = line.strip()
    if len(s) >= 10:
        non_empty += 1
        chars += len(s)
    if n % 1000000 == 0:
        print(f'  scanned {n/1e6:.0f}M lines, {non_empty} non-empty, {chars/1e6:.1f}M chars')
f.close()
print(f'\nTotal: {n} lines, {non_empty} non-empty(>=10), {chars/1e6:.1f}M chars')
print(f'Estimated tokens: ~{chars/1.7/1e6:.0f}M')
print(f'Per neuron (10 split): ~{chars/1.7/1e6/10:.0f}M tokens')
print(f'Data/param ratio (36M): ~{chars/1.7/1e6/10/36:.1f}')
