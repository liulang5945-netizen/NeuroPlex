"""
运行数据生成器扩充训练数据
"""
import sys
import json
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("=" * 60)
    print("运行统一数据生成器扩充训练数据")
    print("=" * 60)

    try:
        from taiji.data.unified_data_generator import (
            gen_identity_data,
            gen_knowledge_data,
            gen_math_data,
            gen_logic_data,
            gen_code_data,
            gen_conversation_data,
            gen_reflection_data,
            gen_planning_data,
            gen_safety_data,
            gen_creative_data,
            gen_error_handling_data,
            gen_software_engineering_data,
        )

        output_dir = Path("taiji_data/training_data/sft")
        output_dir.mkdir(parents=True, exist_ok=True)

        all_data = []

        # 生成各类数据
        generators = [
            ("身世记忆", gen_identity_data, 1000),
            ("知识问答", gen_knowledge_data, 10000),
            ("数学推理", gen_math_data, 5000),
            ("逻辑推理", gen_logic_data, 3000),
            ("代码能力", gen_code_data, 5000),
            ("多轮对话", gen_conversation_data, 3000),
            ("自我反思", gen_reflection_data, 1000),
            ("任务规划", gen_planning_data, 2000),
            ("安全伦理", gen_safety_data, 1000),
            ("创意写作", gen_creative_data, 2000),
            ("错误处理", gen_error_handling_data, 1000),
            ("软件工程", gen_software_engineering_data, 2000),
        ]

        for name, gen_func, target in generators:
            print(f"\n生成 {name} 数据 (目标: {target} 条)...")
            try:
                data = gen_func(target)
                print(f"  实际生成: {len(data)} 条")
                all_data.extend(data)
            except Exception as e:
                print(f"  生成失败: {e}")

        # 保存
        output_file = output_dir / "generated_unified_data.jsonl"
        print(f"\n保存到: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"总计生成: {len(all_data)} 条")
        print(f"保存完成: {output_file}")

        # 抽样检查
        print("\n抽样检查:")
        print("-" * 60)
        import random
        samples = random.sample(all_data, min(5, len(all_data)))
        for i, sample in enumerate(samples):
            msgs = sample.get('messages', [])
            if len(msgs) >= 3:
                print(f"\n样本 {i+1}:")
                print(f"  Q: {msgs[1]['content'][:100]}")
                print(f"  A: {msgs[2]['content'][:100]}")

    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保 taiji.data.unified_data_generator 模块存在")
    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
