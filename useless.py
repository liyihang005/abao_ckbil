from collections import Counter

def extract_characters(sentences, target_phrase):
    target_counter = Counter(target_phrase)
    extracted_characters = {char: [] for char in target_counter}

    for sentence in sentences:
        for char in target_counter:
            if target_counter[char] > 0 and char in sentence:
                extracted_characters[char].append(sentence[:1])  # 提取句子中的第一个汉字
                target_counter[char] -= 1

    # 检查是否提取了足够的汉字来组成目标句子
    if all(count == 0 for count in target_counter.values()):
        result = "".join(extracted_characters[char][0] for char in target_phrase)
        return result
    else:
        return None

# 示例句子
sentences = [
    "今天天气不错",
    "学习使人进步",
    "好好学习天天向上",
    "中秋节快乐",
]

# 目标短语
target_phrase = "今天学习快乐"

# 提取汉字
result = extract_characters(sentences, target_phrase)

if result:
    print("提取的短语:", result)
else:
    print("无法从给定句子中提取目标短语")
