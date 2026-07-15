"""
态极训练数据下载器
==================

从 HuggingFace 下载 Agent 训练数据，转换为态极的 ReAct 格式。

数据源：
1. AgentInstruct — ReAct 轨迹数据（THUDM）
2. hermes-function-calling — 函数调用数据（NousResearch）
3. Glaive Function Calling — 合成工具调用数据

使用方式：
    python scripts/download_training_data.py
"""
import os
import sys
import json
import logging
import hashlib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DataDownloader")

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "taiji_data" / "training_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_hf_dataset(repo_id: str, filename: str, output_path: str) -> bool:
    """从 HuggingFace 下载数据集文件"""
    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"下载 {repo_id}/{filename}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(Path(output_path).parent),
        )
        logger.info(f"下载完成: {path}")
        return True
    except ImportError:
        logger.warning("需要安装 huggingface_hub: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.warning(f"下载失败: {e}")
        return False


def download_via_url(url: str, output_path: str) -> bool:
    """通过 URL 直接下载"""
    import urllib.request
    try:
        logger.info(f"下载 {url}...")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"保存到 {output_path}")
        return True
    except Exception as e:
        logger.warning(f"下载失败: {e}")
        return False


def convert_agent_instruct(input_path: str, output_path: str) -> int:
    """
    将 AgentInstruct 数据转换为态极 ReAct 格式。

    AgentInstruct 格式：
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }

    态极 ReAct 格式：
    {
        "type": "react",
        "task": "...",
        "steps": [
            {"thought": "...", "action": "...", "action_args": {...}, "observation": "...", "final_answer": "..."}
        ]
    }
    """
    count = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 可能是 JSONL 格式
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    converted = []
    for item in data:
        try:
            conversations = item.get("conversations", [])
            if len(conversations) < 2:
                continue

            task = ""
            steps = []
            current_step = {}

            for conv in conversations:
                role = conv.get("from", "")
                content = conv.get("value", "")

                if role == "human" and not task:
                    task = content
                elif role == "gpt":
                    # 尝试解析 ReAct 格式
                    if "Thought:" in content:
                        thought = content.split("Thought:")[1].split("Action:")[0].strip() if "Action:" in content else content.split("Thought:")[1].strip()
                        current_step["thought"] = thought
                    if "Action:" in content:
                        action_part = content.split("Action:")[1].split("Action Input:")[0].strip() if "Action Input:" in content else content.split("Action:")[1].strip()
                        current_step["action"] = action_part
                    if "Action Input:" in content:
                        input_part = content.split("Action Input:")[1].strip()
                        try:
                            current_step["action_args"] = json.loads(input_part)
                        except:
                            current_step["action_args"] = {"input": input_part}
                    if "Observation:" in content:
                        obs = content.split("Observation:")[1].strip()
                        current_step["observation"] = obs
                    if "Final Answer:" in content:
                        current_step["final_answer"] = content.split("Final Answer:")[1].strip()
                        if current_step:
                            steps.append(current_step)
                            current_step = {}
                    elif current_step.get("thought") and current_step.get("action"):
                        steps.append(current_step)
                        current_step = {}

            if task and steps:
                converted.append({
                    "type": "react",
                    "task": task[:500],
                    "steps": steps[:10],
                })
                count += 1
        except Exception:
            continue

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"转换完成: {count} 条 → {output_path}")
    return count


def convert_function_calling(input_path: str, output_path: str) -> int:
    """
    将函数调用数据转换为态极工具调用格式。

    输入格式（hermes-function-calling）：
    {
        "system": "...",
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "...", "function_call": {...}}
        ],
        "tools": [...]
    }

    输出格式（态极）：
    {
        "type": "tool_call",
        "task": "...",
        "tools": [...],
        "steps": [{"thought": "...", "action": "...", "action_args": {...}, "observation": "..."}]
    }
    """
    count = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    converted = []
    for item in data:
        try:
            conversations = item.get("conversations", [])
            tools = item.get("tools", [])

            task = ""
            steps = []

            for conv in conversations:
                role = conv.get("from", "")
                content = conv.get("value", "")
                func_call = conv.get("function_call")

                if role == "human" and not task:
                    task = content
                elif role == "gpt":
                    if func_call:
                        steps.append({
                            "thought": content[:200] if content else "",
                            "action": func_call.get("name", ""),
                            "action_args": json.loads(func_call.get("arguments", "{}")) if isinstance(func_call.get("arguments"), str) else func_call.get("arguments", {}),
                        })

            if task and steps:
                converted.append({
                    "type": "tool_call",
                    "task": task[:500],
                    "tools": [t.get("function", {}).get("name", "") for t in tools[:20]],
                    "steps": steps[:10],
                })
                count += 1
        except Exception:
            continue

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"转换完成: {count} 条 → {output_path}")
    return count


def generate_synthetic_react_data(count: int = 500) -> int:
    """
    生成合成 ReAct 训练数据（用于补充）。

    使用预定义的工具和场景生成训练样本。
    """
    import random

    tools = [
        {"name": "search", "desc": "搜索互联网", "args": {"input": "关键词"}},
        {"name": "read_webpage", "desc": "阅读网页", "args": {"input": "URL"}},
        {"name": "read_local_file", "desc": "读取文件", "args": {"input": "文件路径"}},
        {"name": "write_file", "desc": "写入文件", "args": {"input": "路径 | 内容"}},
        {"name": "execute_python", "desc": "执行Python", "args": {"input": "代码"}},
        {"name": "calculator", "desc": "数学计算", "args": {"input": "表达式"}},
        {"name": "list_directory", "desc": "列目录", "args": {"input": "路径"}},
        {"name": "create_plan", "desc": "创建计划", "args": {"input": "任务描述"}},
        {"name": "data_query", "desc": "查询数据", "args": {"input": "文件 | 操作"}},
        {"name": "text_summarize", "desc": "文本摘要", "args": {"input": "文本"}},
    ]

    templates = [
        {
            "task": "搜索关于{topic}的最新信息，并总结要点",
            "steps": [
                {"thought": "我需要搜索{topic}的相关信息", "action": "search", "action_args": {"input": "{topic}"}},
                {"thought": "搜索完成，让我阅读详细内容", "action": "read_webpage", "action_args": {"input": "https://example.com/{topic}"}},
                {"thought": "现在总结要点", "action": "text_summarize", "action_args": {"input": "{content}"}},
            ],
        },
        {
            "task": "读取{file}文件，分析其中的数据",
            "steps": [
                {"thought": "先读取文件内容", "action": "read_local_file", "action_args": {"input": "{file}"}},
                {"thought": "分析数据结构", "action": "data_query", "action_args": {"input": "{file} | describe"}},
            ],
        },
        {
            "task": "编写一个{lang}程序，实现{feature}",
            "steps": [
                {"thought": "先规划代码结构", "action": "create_plan", "action_args": {"input": "实现{feature}"}},
                {"thought": "编写代码", "action": "write_file", "action_args": {"input": "{filepath} | {code}"}},
                {"thought": "测试代码", "action": "execute_python", "action_args": {"input": "{test_code}"}},
            ],
        },
        {
            "task": "计算{expr}的值",
            "steps": [
                {"thought": "使用计算器工具", "action": "calculator", "action_args": {"input": "{expr}"}},
            ],
        },
    ]

    topics = ["人工智能", "量子计算", "机器学习", "深度学习", "自然语言处理", "计算机视觉",
              "区块链", "云计算", "物联网", "大数据", "网络安全", "自动驾驶"]
    files = ["data.csv", "config.json", "report.txt", "analysis.py", "README.md"]
    langs = ["Python", "JavaScript", "Go", "Rust"]
    features = ["排序算法", "数据可视化", "API接口", "文件处理", "网络爬虫"]
    exprs = ["2**10", "factorial(10)", "sqrt(144)", "log2(1024)", "mean([1,2,3,4,5])"]

    output_path = DATA_DIR / "synthetic_react_data.jsonl"
    total_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(count):
            template = random.choice(templates)
            topic = random.choice(topics)
            file = random.choice(files)
            lang = random.choice(langs)
            feature = random.choice(features)
            expr = random.choice(exprs)

            task = template["task"].format(
                topic=topic, file=file, lang=lang,
                feature=feature, expr=expr
            )

            steps = []
            for step_template in template["steps"]:
                step = {}
                for k, v in step_template.items():
                    if isinstance(v, str):
                        step[k] = v.format(
                            topic=topic, file=file, lang=lang,
                            feature=feature, expr=expr,
                            filepath=f"agent_workspace/{file}",
                            code=f"# {feature} implementation",
                            test_code=f"print('Testing {feature}')",
                            content=f"关于{topic}的内容摘要",
                        )
                    else:
                        step[k] = v
                steps.append(step)

            item = {
                "type": "react",
                "task": task,
                "steps": steps,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_count += 1

    logger.info(f"生成 {total_count} 条合成数据 → {output_path}")
    return total_count


def download_agent_instruct_parquet() -> int:
    """下载 AgentInstruct parquet 数据并转换为 ReAct 格式"""
    import urllib.request
    base_url = "https://huggingface.co/datasets/THUDM/AgentInstruct/resolve/main/data/"
    files = [
        "alfworld-00000-of-00001-302ad687bb3817a4.parquet",
        "db-00000-of-00001-916a87c4725da8c0.parquet",
        "kg-00000-of-00001-9e159f6d0557d229.parquet",
        "mind2web-00000-of-00001-fc25d47330eea0fc.parquet",
        "os-00000-of-00001-971539c34fcc7500.parquet",
        "webshop-00000-of-00001-9f2ae60445e11b4e.parquet",
    ]

    output_dir = DATA_DIR / "agent_instruct"
    output_dir.mkdir(exist_ok=True)

    total = 0
    output_path = DATA_DIR / "agent_instruct_react.jsonl"

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for fname in files:
            local_path = output_dir / fname
            if not local_path.exists():
                url = base_url + fname
                try:
                    logger.info(f"下载 {fname}...")
                    urllib.request.urlretrieve(url, str(local_path))
                except Exception as e:
                    logger.warning(f"下载 {fname} 失败: {e}")
                    continue

            # 读取 parquet
            try:
                import pandas as pd
                df = pd.read_parquet(str(local_path))
                logger.info(f"  {fname}: {len(df)} 条")

                for _, row in df.iterrows():
                    try:
                        # 将 parquet 行转换为 ReAct 格式
                        item = {
                            "type": "react",
                            "task": str(row.get("task", row.get("instruction", "")))[:500],
                            "steps": [],
                        }
                        # 尝试提取 steps
                        if "steps" in row:
                            item["steps"] = row["steps"] if isinstance(row["steps"], list) else []
                        elif "trajectory" in row:
                            item["steps"] = row["trajectory"] if isinstance(row["trajectory"], list) else []

                        if item["task"]:
                            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            total += 1
                    except Exception:
                        continue
            except ImportError:
                logger.warning("需要安装 pandas 和 pyarrow: pip install pandas pyarrow")
                break
            except Exception as e:
                logger.warning(f"读取 {fname} 失败: {e}")
                continue

    logger.info(f"AgentInstruct 转换完成: {total} 条")
    return total


def generate_tool_call_data(count: int = 300) -> int:
    """生成工具调用训练数据"""
    import random

    tool_templates = [
        {
            "task": "帮我搜索{topic}的最新进展",
            "tools": ["search"],
            "steps": [
                {"thought": "用户想了解{topic}，我需要搜索", "action": "search", "action_args": {"input": "{topic} 最新进展"}},
            ],
        },
        {
            "task": "读取{file}并告诉我内容",
            "tools": ["read_local_file"],
            "steps": [
                {"thought": "先读取文件", "action": "read_local_file", "action_args": {"input": "{file}"}},
            ],
        },
        {
            "task": "计算{expr}",
            "tools": ["calculator"],
            "steps": [
                {"thought": "使用计算器", "action": "calculator", "action_args": {"input": "{expr}"}},
            ],
        },
        {
            "task": "查看当前目录下有哪些文件",
            "tools": ["list_directory"],
            "steps": [
                {"thought": "列目录", "action": "list_directory", "action_args": {"input": "."}},
            ],
        },
        {
            "task": "创建一个新文件{file}，内容为{content}",
            "tools": ["write_file"],
            "steps": [
                {"thought": "写入文件", "action": "write_file", "action_args": {"input": "{file} | {content}"}},
            ],
        },
        {
            "task": "查看今天几号",
            "tools": ["datetime"],
            "steps": [
                {"thought": "获取当前时间", "action": "datetime", "action_args": {"input": ""}},
            ],
        },
        {
            "task": "访问{url}看看内容",
            "tools": ["url_check"],
            "steps": [
                {"thought": "抓取网页", "action": "url_check", "action_args": {"input": "{url} | fetch"}},
            ],
        },
        {
            "task": "分析{file}中的数据",
            "tools": ["data_query"],
            "steps": [
                {"thought": "查询数据", "action": "data_query", "action_args": {"input": "{file} | describe"}},
            ],
        },
        {
            "task": "帮我规划一个{project}项目",
            "tools": ["create_plan"],
            "steps": [
                {"thought": "创建计划", "action": "create_plan", "action_args": {"input": "开发{project}"}},
            ],
        },
        {
            "task": "帮我搜索{topic}，然后总结要点",
            "tools": ["search", "text_summarize"],
            "steps": [
                {"thought": "先搜索", "action": "search", "action_args": {"input": "{topic}"}},
                {"thought": "搜索完成，总结要点", "action": "text_summarize", "action_args": {"input": "{topic}相关结果"}},
            ],
        },
    ]

    topics = ["Python编程", "机器学习", "深度学习", "Web开发", "数据库", "云计算", "网络安全", "数据分析"]
    files = ["data.csv", "config.json", "README.md", "main.py", "report.txt"]
    urls = ["https://example.com", "https://github.com", "https://news.ycombinator.com"]
    exprs = ["2+3*4", "sqrt(144)", "factorial(10)", "log2(1024)", "sin(pi/2)"]
    projects = ["Web应用", "数据分析工具", "聊天机器人", "文件管理器", "API服务"]
    contents = ["Hello World", "# README\n\nThis is a project.", "data = [1, 2, 3]"]

    output_path = DATA_DIR / "tool_call_training.jsonl"
    total = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(count):
            template = random.choice(tool_templates)
            topic = random.choice(topics)
            file = random.choice(files)
            url = random.choice(urls)
            expr = random.choice(exprs)
            project = random.choice(projects)
            content = random.choice(contents)

            task = template["task"].format(
                topic=topic, file=file, url=url,
                expr=expr, project=project, content=content
            )
            steps = []
            for step in template["steps"]:
                s = {}
                for k, v in step.items():
                    if isinstance(v, str):
                        s[k] = v.format(
                            topic=topic, file=file, url=url,
                            expr=expr, project=project, content=content
                        )
                    else:
                        s[k] = v
                steps.append(s)

            item = {
                "type": "tool_call",
                "task": task,
                "tools": template["tools"],
                "steps": steps,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1

    logger.info(f"生成 {total} 条工具调用数据 → {output_path}")
    return total


def main():
    """主函数：下载并转换训练数据"""
    logger.info("=" * 60)
    logger.info("态极训练数据下载器")
    logger.info("=" * 60)

    total = 0

    # 1. 生成合成 ReAct 数据（无外部依赖）
    logger.info("\n[1/3] 生成合成 ReAct 训练数据...")
    total += generate_synthetic_react_data(500)

    # 2. 下载 AgentInstruct（parquet 格式）
    logger.info("\n[2/3] 下载 AgentInstruct 数据集...")
    react_path = DATA_DIR / "agent_instruct_react.jsonl"
    if not react_path.exists():
        total += download_agent_instruct_parquet()
    else:
        logger.info("AgentInstruct 已存在，跳过下载")

    # 3. 生成工具调用训练数据（无外部依赖）
    logger.info("\n[3/3] 生成工具调用训练数据...")
    tool_call_path = DATA_DIR / "tool_call_training.jsonl"
    if not tool_call_path.exists():
        total += generate_tool_call_data(300)
    else:
        logger.info("工具调用数据已存在，跳过")

    # 统计
    logger.info("\n" + "=" * 60)
    logger.info(f"总计: {total} 条训练数据")
    logger.info(f"数据目录: {DATA_DIR}")

    # 列出所有数据文件
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        size = f.stat().st_size
        with open(f, 'r', encoding='utf-8') as fh:
            lines = sum(1 for _ in fh)
        logger.info(f"  {f.name}: {lines} 条 ({size/1024:.1f} KB)")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
