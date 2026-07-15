"""
清洗预训练数据
==============
删除"复制输入"的垃圾数据，保留高质量部分
"""
import json
import os

INPUT_FILE = "taiji_data/training_data/pretrain_all.jsonl"
OUTPUT_FILE = "taiji_data/training_data/pretrain_cleaned.jsonl"

# Block 2: lines 47567-347537 (Read and understand - text echo)
# Block 3: lines 347538-547482 (Summarize - text echo)
BAD_RANGES = [
    (47567, 347537),   # Block 2
    (347538, 547482),  # Block 3
]

def is_bad_line(line_num):
    """检查是否在垃圾数据范围内"""
    for start, end in BAD_RANGES:
        if start <= line_num <= end:
            return True
    return False

def main():
    print("=" * 50)
    print("清洗预训练数据")
    print("=" * 50)

    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    total_lines = 0
    kept_lines = 0
    removed_lines = 0

    with open(INPUT_FILE, encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            total_lines += 1

            if is_bad_line(i):
                removed_lines += 1
                continue

            f_out.write(line)
            kept_lines += 1

            if kept_lines % 10000 == 0:
                print(f"  已处理 {kept_lines:,} 条...")

    print()
    print("=" * 50)
    print("清洗完成！")
    print("=" * 50)
    print(f"  原始数据：{total_lines:,} 条")
    print(f"  保留数据：{kept_lines:,} 条")
    print(f"  删除数据：{removed_lines:,} 条")
    print(f"  删除比例：{removed_lines/total_lines*100:.1f}%")
    print()
    print(f"输出文件：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()
