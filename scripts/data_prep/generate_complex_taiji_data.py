"""
生成态极专属的复杂工具调用与长对话训练数据
==========================================
解决开源数据集工具格式不匹配、且缺乏态极特定多步工作流的问题。
本脚本将程序化生成大量包含：
1. 包含错误纠正、多步联动的复杂 ReAct 轨迹 (Taiji专属工具)
2. 强上下文依赖的多轮编程与逻辑对话
"""
import json
import random
import os
from pathlib import Path

# 态极的核心工具集
TOOLS = [
    "search", "read_file", "write_file", "execute_python", 
    "list_directory", "grep_search", "get_current_time", "calculate"
]

def generate_multi_step_react(num_samples=5000):
    """生成多步、带纠错逻辑的复杂 ReAct 数据"""
    samples = []
    
    # 场景库
    scenarios = [
        {
            "task": "帮我看看 src 目录下有没有 main.py，并运行它。",
            "steps": [
                {
                    "thought": "用户要求我检查 src 目录并运行 main.py。我首先需要列出 src 目录的内容，确认文件是否存在。",
                    "action": "list_directory",
                    "args": {"path": "src"},
                    "obs": "文件列表: ['utils.py', 'config.json', 'main.py']\n状态: 成功"
                },
                {
                    "thought": "我看到了 main.py 存在于 src 目录下。接下来我需要读取它的内容或者直接运行它。为了保险起见，我先用 execute_python 运行它。",
                    "action": "execute_python",
                    "args": {"code": "import subprocess\nresult = subprocess.run(['python', 'src/main.py'], capture_output=True, text=True)\nprint(result.stdout)"},
                    "obs": "Traceback (most recent call last):\n  File \"src/main.py\", line 2, in <module>\n    import requests\nModuleNotFoundError: No module named 'requests'"
                },
                {
                    "thought": "运行报错了，提示找不到 'requests' 模块。这说明环境里缺少依赖，或者代码有特定的运行要求。我应该读取这个文件看看它的逻辑。",
                    "action": "read_file",
                    "args": {"path": "src/main.py"},
                    "obs": "import requests\nimport json\n\ndef fetch_data():\n    pass\n# rest of code..."
                },
                {
                    "thought": "确实是引入了 requests。我需要向用户报告这个错误，并建议他们安装依赖。",
                    "answer": "我在 `src` 目录下找到了 `main.py` 并尝试运行了它。但是运行失败了，错误提示缺少 `requests` 库。\n\n你可以通过运行 `pip install requests` 来安装缺失的依赖，然后再试一次。需要我帮你写一个 requirements.txt 吗？"
                }
            ]
        },
        {
            "task": "我想知道当前目录下所有 markdown 文件里，提到 'API_KEY' 的地方在哪里。",
            "steps": [
                {
                    "thought": "用户需要在当前目录的所有 markdown 文件中搜索 'API_KEY' 这个词。我应该使用 grep_search 工具进行全局搜索。",
                    "action": "grep_search",
                    "args": {"pattern": "API_KEY", "dir_path": ".", "include_pattern": "*.md"},
                    "obs": "匹配结果:\nREADME.md:45: 请在环境变量中设置您的 API_KEY。\ndocs/setup.md:12: export API_KEY='your_secret_key_here'"
                },
                {
                    "thought": "我已经找到了包含 'API_KEY' 的位置，分别在 README.md 的第 45 行和 docs/setup.md 的第 12 行。我现在可以把结果总结给用户。",
                    "answer": "我帮您搜索了所有的 Markdown 文件，提到 'API_KEY' 的地方有两处：\n\n1. **`README.md` (第45行)**: `请在环境变量中设置您的 API_KEY。`\n2. **`docs/setup.md` (第12行)**: `export API_KEY='your_secret_key_here'`\n\n需要我为您读取这两个文件的完整上下文吗？"
                }
            ]
        },
        {
            "task": "计算 124 的 3 次方，然后把结果写到 result.txt 文件里。",
            "steps": [
                {
                    "thought": "用户要求我先计算 124 的 3 次方，然后将结果保存到文件中。我先使用 calculate 工具进行计算。",
                    "action": "calculate",
                    "args": {"expression": "124 ** 3"},
                    "obs": "计算结果: 1906624"
                },
                {
                    "thought": "计算成功，124 的 3 次方是 1906624。接下来我需要使用 write_file 工具将这个结果写入 result.txt。",
                    "action": "write_file",
                    "args": {"path": "result.txt", "content": "124 的 3 次方计算结果是: 1906624\n"},
                    "obs": "文件写入成功: result.txt"
                },
                {
                    "thought": "文件写入完成。我可以回复用户任务已完成。",
                    "answer": "我已经为您计算完毕。124 的 3 次方等于 1,906,624。并且我已经将这个结果成功写入到了当前目录的 `result.txt` 文件中。您可以随时查看。"
                }
            ]
        }
    ]

    # 通过变量替换和微调扩充数据集
    for _ in range(num_samples):
        base_scenario = random.choice(scenarios)
        messages = [{"role": "system", "content": "你是态极，一个能够执行复杂多步工具调用的 AI 助手。你遵循 Thought-Action-Observation 的推理框架。"}]
        
        # 变体生成逻辑：在原始文本中加入细微变化，避免数据过于单一
        task = base_scenario["task"]
        if random.random() > 0.5:
            task = f"你好，{task}"
            
        messages.append({"role": "user", "content": task})
        
        assistant_content = ""
        for step in base_scenario["steps"]:
            if "action" in step:
                assistant_content += f"Thought: {step['thought']}\nAction: {step['action']}\nAction Input: {json.dumps(step['args'], ensure_ascii=False)}\n"
                messages.append({"role": "assistant", "content": assistant_content.strip()})
                messages.append({"role": "tool", "name": step["action"], "content": f"Observation: {step['obs']}"})
                assistant_content = "" # Reset for next phase
            else:
                # 最终回答
                assistant_content += f"Thought: {step['thought']}\n\n{step['answer']}"
                messages.append({"role": "assistant", "content": assistant_content.strip()})
                
        samples.append({"messages": messages})
        
    return samples

def generate_long_context_conversations(num_samples=2000):
    """生成具有长上下文依赖的对话（例如追踪变量名、代码演进）"""
    samples = []
    
    # 构建多轮逻辑深度的模板
    for _ in range(num_samples):
        messages = [{"role": "system", "content": "你是态极，一个擅长处理多轮长上下文代码辅助的 AI。"}]
        
        var_name = random.choice(["user_auth_token", "db_connection_pool", "cache_redis_client", "max_retry_limit"])
        module_name = random.choice(["auth.py", "database.py", "cache.py", "config.py"])
        
        turns = [
            ("user", f"我在写一个项目，现在需要在一个叫 `{module_name}` 的文件里初始化一个变量叫做 `{var_name}`，可以帮我写个基础的骨架吗？"),
            ("assistant", f"当然可以。在 `{module_name}` 中初始化 `{var_name}` 的基础骨架如下：\n\n```python\n# {module_name}\n\n{var_name} = None\n\ndef init_{module_name.split('.')[0]}():\n    global {var_name}\n    # TODO: 实现具体的初始化逻辑\n    {var_name} = \"initialized\"\n    print(f\"{var_name} ready.\")\n```\n你可以根据具体需求替换里面的 TODO 部分。"),
            ("user", "我现在想加一个异常处理，如果初始化失败，抛出特定的异常 CustomInitError。"),
            ("assistant", f"好的，我们需要先定义这个异常，然后在 `init_{module_name.split('.')[0]}()` 函数中加入 try-except 块。代码如下：\n\n```python\n# {module_name}\n\nclass CustomInitError(Exception):\n    pass\n\n{var_name} = None\n\ndef init_{module_name.split('.')[0]}():\n    global {var_name}\n    try:\n        # TODO: 实现具体的初始化逻辑\n        # 模拟可能失败的操作\n        # raise RuntimeError(\"连接超时\")\n        {var_name} = \"initialized\"\n        print(f\"{var_name} ready.\")\n    except Exception as e:\n        raise CustomInitError(f\"初始化失败: {{e}}\")\n```"),
            ("user", "如果在这个文件之外的地方调用，怎么确保它是单例的？"),
            ("assistant", f"为了确保在其他文件调用时它是单例的（只初始化一次），我们可以修改 `{module_name}` 中的逻辑，利用 Python 模块本身只加载一次的特性，或者写一个获取实例的方法：\n\n```python\n# {module_name}\n\nclass CustomInitError(Exception):\n    pass\n\n{var_name} = None\n\ndef get_instance():\n    global {var_name}\n    if {var_name} is None:\n        try:\n            # 执行初始化\n            {var_name} = \"initialized_singleton\"\n        except Exception as e:\n            raise CustomInitError(f\"初始化失败: {{e}}\")\n    return {var_name}\n```\n这样外部只需要调用 `get_instance()` 即可。"),
            ("user", "等等，最开始我们在文件顶部定义了那个变量为 None 对吧？如果多线程同时访问 get_instance 怎么办？"),
            ("assistant", f"非常敏锐！是的，我们最初将 `{var_name}` 设置为 `None`。在多线程环境下，如果两个线程同时检查 `if {var_name} is None:`，它们可能都会看到 `None` 并执行两次初始化操作。这就破坏了单例。\n\n我们需要引入线程锁 (`threading.Lock`) 来解决这个问题：\n\n```python\nimport threading\n\nclass CustomInitError(Exception):\n    pass\n\n{var_name} = None\n_lock = threading.Lock()\n\ndef get_instance():\n    global {var_name}\n    if {var_name} is None:  # 第一重检查\n        with _lock:\n            if {var_name} is None:  # 第二重检查 (双重检查锁定)\n                try:\n                    {var_name} = \"initialized_thread_safe_singleton\"\n                except Exception as e:\n                    raise CustomInitError(f\"初始化失败: {{e}}\")\n    return {var_name}\n```"),
            ("user", "太棒了，那么之前抛出的那个自定义异常名字叫什么来着？我能在外部捕获它吗？"),
            ("assistant", f"那个自定义异常的名字叫 `CustomInitError`。是的，你完全可以在外部捕获它。\n\n假设你在另一个文件 `main.py` 中调用它，你可以这样做：\n\n```python\nfrom {module_name.split('.')[0]} import get_instance, CustomInitError\n\ntry:\n    instance = get_instance()\n    print(\"获取成功:\", instance)\nexcept CustomInitError as e:\n    print(\"捕获到我们自定义的初始化错误:\", e)\n```\n这里我们成功串联了你在第一轮要求的 `{var_name}` 初始化，第二轮加的 `CustomInitError`，以及第四轮的线程安全单例模式。")
        ]
        
        for role, content in turns:
            messages.append({"role": role, "content": content})
            
        samples.append({"messages": messages})
        
    return samples

def main():
    print("=" * 60)
    print("生成态极专属的复杂工具调用与长对话数据")
    print("=" * 60)
    
    output_dir = Path("taiji_data/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成复杂多步 ReAct 轨迹
    react_file = output_dir / "taiji_complex_react.jsonl"
    print("正在生成态极专属复杂 ReAct 数据...")
    react_samples = generate_multi_step_react(6000)
    with open(react_file, "w", encoding="utf-8") as f:
        for sample in react_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  成功生成 {len(react_samples)} 条，保存至 {react_file}")
    
    # 2. 生成长上下文代码追踪对话
    long_ctx_file = output_dir / "taiji_long_context.jsonl"
    print("\n正在生成强上下文依赖的长对话数据...")
    long_samples = generate_long_context_conversations(4000)
    with open(long_ctx_file, "w", encoding="utf-8") as f:
        for sample in long_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  成功生成 {len(long_samples)} 条，保存至 {long_ctx_file}")
    
    print("=" * 60)
    print("态极专属高质量数据生成完毕！")
    print("=" * 60)

if __name__ == "__main__":
    main()
