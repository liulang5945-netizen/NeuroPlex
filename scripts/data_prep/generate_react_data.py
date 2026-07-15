"""生成 ReAct 工具调用训练数据"""
import json
import os

output_path = 'taiji_data/training_data/react_data.jsonl'

templates = [
    {
        'task': '搜索Python教程',
        'thought': '用户想学习Python，我需要搜索相关教程。',
        'action': 'search',
        'args': {'query': 'Python入门教程'},
        'obs': 'Python官方教程、菜鸟教程、廖雪峰Python教程等。',
        'answer': '以下是Python入门教程推荐：\n1. Python官方教程 - 最权威\n2. 菜鸟教程 - 适合中文用户\n3. 廖雪峰Python教程 - 深入浅出'
    },
    {
        'task': '帮我搜索最新的AI新闻',
        'thought': '用户想了解AI最新动态，需要搜索新闻。',
        'action': 'search',
        'args': {'query': '2026年AI最新新闻'},
        'obs': 'OpenAI发布GPT-5、谷歌Gemini更新、开源模型Llama4发布等。',
        'answer': '2026年AI领域最新动态：\n1. 大语言模型持续进化\n2. AI Agent成为主流\n3. 多模态能力大幅提升\n4. 开源模型快速发展'
    },
    {
        'task': '读取config.json文件',
        'thought': '用户想查看配置文件，我需要读取它。',
        'action': 'read_file',
        'args': {'path': 'config.json'},
        'obs': '{"name": "taiji", "version": "1.6", "model": "1B"}',
        'answer': '配置文件内容：\n- 名称: taiji\n- 版本: 1.6\n- 模型: 1B参数'
    },
    {
        'task': '列出当前目录的文件',
        'thought': '用户想查看目录结构，我需要列出文件。',
        'action': 'list_directory',
        'args': {'path': '.'},
        'obs': 'README.md, requirements.txt, taiji/, api/, frontend/',
        'answer': '当前目录包含：\n- README.md - 项目说明\n- requirements.txt - 依赖\n- taiji/ - 核心代码\n- api/ - API服务\n- frontend/ - 前端'
    },
    {
        'task': '计算斐波那契数列前10项',
        'thought': '用户需要计算斐波那契数列，我用Python执行。',
        'action': 'execute_python',
        'args': {'code': 'a, b = 0, 1\nresult = []\nfor _ in range(10):\n    result.append(a)\n    a, b = b, a+b\nprint(result)'},
        'obs': '[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]',
        'answer': '斐波那契数列前10项：0, 1, 1, 2, 3, 5, 8, 13, 21, 34\n\n规律：每个数等于前两个数之和。'
    },
    {
        'task': '写一个排序函数',
        'thought': '用户需要排序函数，我用Python实现。',
        'action': 'execute_python',
        'args': {'code': 'def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\nprint(quicksort([3,6,8,10,1,2,1]))'},
        'obs': '[1, 1, 2, 3, 6, 8, 10]',
        'answer': '快速排序函数已实现，测试结果：[3,6,8,10,1,2,1] -> [1,1,2,3,6,8,10]'
    },
    {
        'task': '记住我的生日是10月1日',
        'thought': '用户想让我记住生日，我需要保存到记忆。',
        'action': 'remember',
        'args': {'content': '用户生日：10月1日'},
        'obs': '已保存到记忆',
        'answer': '好的，我已经记住了！你的生日是10月1日。'
    },
    {
        'task': '你还记得我的名字吗',
        'thought': '用户询问是否记得名字，我需要从记忆中检索。',
        'action': 'recall',
        'args': {'query': '用户名字'},
        'obs': '记忆中没有找到相关信息',
        'answer': '抱歉，我目前没有关于你名字的记忆。你可以告诉我，我会记住的！'
    },
    {
        'task': '现在几点了',
        'thought': '用户想知道当前时间，我需要获取时间。',
        'action': 'get_current_time',
        'args': {},
        'obs': '2026-06-15 10:30:45',
        'answer': '现在是2026年6月15日 10:30:45。'
    },
    {
        'task': '计算123乘以456',
        'thought': '用户需要数学计算，我用计算工具。',
        'action': 'calculate',
        'args': {'expression': '123 * 456'},
        'obs': '56088',
        'answer': '123 x 456 = 56088'
    },
    {
        'task': '帮我写一个Python脚本计算圆的面积',
        'thought': '用户需要圆面积计算脚本，我先写代码再执行验证。',
        'action': 'execute_python',
        'args': {'code': 'import math\ndef circle_area(r):\n    return math.pi * r ** 2\nfor r in [1,2,5,10]:\n    print(f"半径{r}: {circle_area(r):.2f}")'},
        'obs': '半径1: 3.14\n半径2: 12.57\n半径5: 78.54\n半径10: 314.16',
        'answer': '圆面积公式: S = pi * r^2\n测试结果：半径1->3.14, 半径5->78.54, 半径10->314.16'
    },
    {
        'task': '搜索机器学习框架对比',
        'thought': '用户想了解ML框架，我搜索最新对比。',
        'action': 'search',
        'args': {'query': 'PyTorch TensorFlow JAX 对比'},
        'obs': 'PyTorch主导研究，TensorFlow在生产占优，JAX在高性能计算崛起。',
        'answer': '主流ML框架对比：\n- PyTorch: 灵活、研究友好\n- TensorFlow: 生态完善、部署方便\n- JAX: 高性能、函数式编程'
    },
    {
        'task': '查看系统内存使用情况',
        'thought': '用户想了解内存状态，我执行代码检查。',
        'action': 'execute_python',
        'args': {'code': 'import psutil\nm = psutil.virtual_memory()\nprint(f"总计: {m.total//1024//1024}MB")\nprint(f"使用率: {m.percent}%" )'},
        'obs': '总计: 16384MB\n使用率: 45.2%',
        'answer': '系统内存状态：\n- 总计: 16GB\n- 使用率: 45.2%\n- 状态: 正常'
    },
    {
        'task': '帮我写一个Hello World程序',
        'thought': '用户需要简单程序，我用Python写。',
        'action': 'execute_python',
        'args': {'code': 'print("Hello, World!")'},
        'obs': 'Hello, World!',
        'answer': '```python\nprint("Hello, World!")\n```\n输出: Hello, World!'
    },
    {
        'task': '搜索Docker入门教程',
        'thought': '用户想学Docker，我搜索教程。',
        'action': 'search',
        'args': {'query': 'Docker容器入门教程中文'},
        'obs': 'Docker官方文档、菜鸟教程Docker篇、Docker从入门到实践等。',
        'answer': 'Docker入门教程推荐：\n1. Docker官方文档 - 最权威\n2. 菜鸟教程 - 中文友好\n3. Docker从入门到实践 - 社区维护'
    },
    {
        'task': '计算1到100的和',
        'thought': '用户需要数学计算，我用Python执行。',
        'action': 'execute_python',
        'args': {'code': 'print(sum(range(1, 101)))'},
        'obs': '5050',
        'answer': '1+2+3+...+100 = 5050\n\n公式: n*(n+1)/2 = 100*101/2 = 5050'
    },
    {
        'task': '搜索Git常用命令',
        'thought': '用户想学Git，我搜索常用命令。',
        'action': 'search',
        'args': {'query': 'Git常用命令大全'},
        'obs': 'git clone, git add, git commit, git push, git pull, git branch等。',
        'answer': 'Git常用命令：\n- git clone: 克隆仓库\n- git add: 添加暂存\n- git commit: 提交更改\n- git push: 推送远程\n- git pull: 拉取更新\n- git branch: 分支管理'
    },
    {
        'task': '计算10的阶乘',
        'thought': '用户需要阶乘计算，我用Python执行。',
        'action': 'execute_python',
        'args': {'code': 'import math\nprint(math.factorial(10))'},
        'obs': '3628800',
        'answer': '10! = 10 x 9 x 8 x ... x 1 = 3628800'
    },
    {
        'task': '查看Python版本',
        'thought': '用户想了解Python版本，我执行代码查看。',
        'action': 'execute_python',
        'args': {'code': 'import sys\nprint(sys.version)'},
        'obs': '3.11.0 (main, Oct 2024, 19:00:00)',
        'answer': '当前Python版本: 3.11.0'
    },
    {
        'task': '搜索Rust编程语言特点',
        'thought': '用户想了解Rust，我搜索其特点。',
        'action': 'search',
        'args': {'query': 'Rust编程语言特点优势'},
        'obs': '内存安全、零成本抽象、并发安全、高性能、现代化工具链。',
        'answer': 'Rust语言核心特点：\n1. 内存安全 - 编译时检查\n2. 零成本抽象 - 性能媲美C++\n3. 并发安全 - 防止数据竞争\n4. 现代化工具 - cargo包管理'
    },
    {
        'task': '列出taiji目录结构',
        'thought': '用户想看项目结构，我列出目录。',
        'action': 'list_directory',
        'args': {'path': 'taiji/'},
        'obs': 'core/, life/, agent/, brain/, tools/, train/, safety/, multimodal/',
        'answer': 'taiji项目结构：\n- core/ - 核心推理引擎\n- life/ - 生命系统\n- agent/ - 智能体\n- brain/ - 大脑皮层\n- tools/ - 工具集\n- train/ - 训练系统\n- safety/ - 安全系统\n- multimodal/ - 多模态'
    },
]

# Convert to Taiji message format
samples = []
for t in templates:
    args_str = json.dumps(t['args'], ensure_ascii=False) if t['args'] else ''
    sample = {
        'messages': [
            {'role': 'system', 'content': '你是态极，一个会使用工具的AI助手。'},
            {'role': 'user', 'content': t['task']},
            {'role': 'assistant', 'content': f"<think>{t['thought']}</think><tool_call>{t['action']}({args_str})</tool_call>"},
            {'role': 'system', 'content': f"<tool_result>{t['obs']}</tool_result>"},
            {'role': 'assistant', 'content': t['answer']},
        ]
    }
    samples.append(sample)

# Save
with open(output_path, 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

print(f'Generated {len(samples)} ReAct samples -> {output_path}')
