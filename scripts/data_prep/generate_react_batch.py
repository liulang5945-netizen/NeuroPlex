"""批量生成 ReAct 工具调用训练数据"""
import json
import random

output_path = 'taiji_data/training_data/react_data.jsonl'

# 态极实际工具定义
TOOLS = {
    'search': {'desc': '搜索网页获取信息', 'args': ['query'], 'category': '网络'},
    'read_file': {'desc': '读取文件内容', 'args': ['path'], 'category': '文件'},
    'write_file': {'desc': '写入文件', 'args': ['path', 'content'], 'category': '文件'},
    'execute_python': {'desc': '执行Python代码', 'args': ['code'], 'category': '代码'},
    'list_directory': {'desc': '列出目录文件', 'args': ['path'], 'category': '文件'},
    'search_files': {'desc': '搜索文件', 'args': ['pattern'], 'category': '文件'},
    'get_current_time': {'desc': '获取当前时间', 'args': [], 'category': '系统'},
    'calculate': {'desc': '数学计算', 'args': ['expression'], 'category': '工具'},
    'web_browse': {'desc': '浏览网页获取内容', 'args': ['url'], 'category': '网络'},
    'remember': {'desc': '保存信息到记忆', 'args': ['content'], 'category': '记忆'},
    'recall': {'desc': '从记忆中检索', 'args': ['query'], 'category': '记忆'},
}

# 任务模板
TASKS = [
    # 搜索任务
    {'task': '搜索{topic}的最新消息', 'tool': 'search', 'args': lambda: {'query': f'{random.choice(["AI", "科技", "Python", "机器学习", "区块链", "量子计算"])}最新消息'}, 'thought': '用户想了解{topic}的最新动态，需要搜索。', 'obs': '搜索结果：找到相关新闻和文章。', 'answer': '根据搜索结果，{topic}领域最近有以下重要进展...'},
    {'task': '帮我查一下{topic}的教程', 'tool': 'search', 'args': lambda: {'query': f'{random.choice(["Python", "Git", "Docker", "React", "Vue", "Rust"])}入门教程'}, 'thought': '用户需要学习{topic}，我搜索相关教程。', 'obs': '找到多个优质教程资源。', 'answer': '以下是{topic}的推荐教程：\n1. 官方文档 - 最权威\n2. 菜鸟教程 - 适合入门\n3. 实战项目 - 边学边做'},
    {'task': '搜索{topic}和{topic2}的区别', 'tool': 'search', 'args': lambda: {'query': f'{random.choice(["Python", "Java", "Go", "Rust"])} vs {random.choice(["C++", "JavaScript", "TypeScript"])} 对比'}, 'thought': '用户想比较两种技术，需要搜索对比资料。', 'obs': '找到详细的对比分析文章。', 'answer': '两者的主要区别：\n- 语法：一个简洁，一个严谨\n- 性能：各有优势\n- 生态：应用场景不同'},

    # 文件操作
    {'task': '读取README文件', 'tool': 'read_file', 'args': lambda: {'path': 'README.md'}, 'thought': '用户想查看项目说明，我读取README。', 'obs': '# 项目名称\n\n这是一个示例项目，用于演示功能。\n\n## 安装\npip install -r requirements.txt\n\n## 使用\npython main.py', 'answer': 'README文件内容：\n- 项目名称和简介\n- 安装方法：pip install\n- 使用方法：python main.py'},
    {'task': '查看配置文件内容', 'tool': 'read_file', 'args': lambda: {'path': 'config.json'}, 'thought': '用户想查看配置，我读取配置文件。', 'obs': '{"name": "taiji", "version": "1.6", "debug": false}', 'answer': '配置文件内容：\n- 项目名：taiji\n- 版本：1.6\n- 调试模式：关闭'},
    {'task': '列出src目录的文件', 'tool': 'list_directory', 'args': lambda: {'path': 'src/'}, 'thought': '用户想查看源码目录结构。', 'obs': 'main.py, utils.py, config.py, models/, tests/', 'answer': 'src目录包含：\n- main.py - 主程序\n- utils.py - 工具函数\n- config.py - 配置\n- models/ - 模型目录\n- tests/ - 测试目录'},
    {'task': '搜索Python文件', 'tool': 'search_files', 'args': lambda: {'pattern': '*.py'}, 'thought': '用户想找到所有Python文件。', 'obs': 'main.py, utils.py, test.py, config.py', 'answer': '找到以下Python文件：\n- main.py - 主程序\n- utils.py - 工具函数\n- test.py - 测试文件\n- config.py - 配置文件'},

    # 代码执行
    {'task': '计算{expr}', 'tool': 'execute_python', 'args': lambda: {'code': f'print({random.choice(["2**10", "sum(range(100))", "3.14159*5**2", "100!", "fibonacci(20)"])})'}, 'thought': '用户需要数学计算，我用Python执行。', 'obs': f'{random.choice(["1024", "4950", "78.54", "3628800", "6765"])}', 'answer': '计算结果：{obs}'},
    {'task': '帮我写一个{func}函数', 'tool': 'execute_python', 'args': lambda: {'code': f'def {random.choice(["sort", "fibonacci", "prime_check", "binary_search", "merge"])}(arr):\n    # 实现\n    pass\nprint("函数已定义")'}, 'thought': '用户需要一个函数，我用Python实现。', 'obs': '函数已定义', 'answer': '函数已创建，可以用于...'},
    {'task': '运行这段代码看看结果', 'tool': 'execute_python', 'args': lambda: {'code': 'import sys\nprint(f"Python {sys.version}")\nprint(f"Platform: {sys.platform}")'}, 'thought': '用户想运行代码查看环境信息。', 'obs': 'Python 3.11.0\nPlatform: win32', 'answer': '运行结果：\n- Python版本：3.11.0\n- 平台：Windows'},
    {'task': '生成一个随机密码', 'tool': 'execute_python', 'args': lambda: {'code': 'import random, string\npwd = "".join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=16))\nprint(pwd)'}, 'thought': '用户需要随机密码，我用Python生成。', 'obs': 'aB3$kL9#mN2@pQ5!', 'answer': '已生成16位随机密码，包含大小写字母、数字和特殊字符。'},

    # 记忆操作
    {'task': '记住{info}', 'tool': 'remember', 'args': lambda: {'content': f'用户{random.choice(["喜欢Python", "是学生", "在北京", "在学AI", "生日是5月1日"])}'}, 'thought': '用户让我记住信息，我保存到记忆。', 'obs': '已保存到记忆', 'answer': '好的，我已经记住了！'},
    {'task': '你还记得{info}吗', 'tool': 'recall', 'args': lambda: {'query': f'{random.choice(["我的名字", "我喜欢什么", "我在哪里", "我的生日"])}'}, 'thought': '用户询问记忆，我需要检索。', 'obs': f'{random.choice(["记忆中找到：用户喜欢Python", "未找到相关信息"])}', 'answer': '{obs}'},

    # 时间和计算
    {'task': '现在几点了', 'tool': 'get_current_time', 'args': lambda: {}, 'thought': '用户想知道当前时间。', 'obs': '2026-06-15 14:30:00', 'answer': '现在是2026年6月15日 14:30:00。'},
    {'task': '今天星期几', 'tool': 'get_current_time', 'args': lambda: {}, 'thought': '用户想知道今天是星期几。', 'obs': '2026-06-15 星期日', 'answer': '今天是2026年6月15日，星期日。'},
    {'task': '计算{num1}+{num2}', 'tool': 'calculate', 'args': lambda: {'expression': f'{random.randint(100,999)} + {random.randint(100,999)}'}, 'thought': '用户需要加法计算。', 'obs': f'{random.randint(200,1998)}', 'answer': '计算结果：{obs}'},
    {'task': '计算{num}的平方根', 'tool': 'calculate', 'args': lambda: {'expression': f'sqrt({random.choice([16, 25, 36, 49, 64, 81, 100, 144, 225])})'}, 'thought': '用户需要计算平方根。', 'obs': f'{random.choice([4, 5, 6, 7, 8, 9, 10, 12, 15])}', 'answer': '平方根是：{obs}'},

    # 网页浏览
    {'task': '打开{url}看看内容', 'tool': 'web_browse', 'args': lambda: {'url': f'https://{random.choice(["example.com", "python.org", "github.com"])}'}, 'thought': '用户想浏览网页，我获取内容。', 'obs': '网页标题：示例网站\n主要内容：这是一个示例页面...', 'answer': '网页内容：\n- 标题：示例网站\n- 主要内容：这是一个示例页面'},

    # 多步骤任务
    {'task': '帮我分析这个数据文件', 'tool': 'read_file', 'args': lambda: {'path': 'data.csv'}, 'thought': '用户想分析数据文件，我先读取内容。', 'obs': 'name,age,score\nAlice,25,85\nBob,30,92\nCharlie,22,78', 'answer': '数据分析结果：\n- 共3条记录\n- 平均年龄：25.7岁\n- 平均分数：85分\n- 最高分：92（Bob）'},
    {'task': '帮我写一个Web服务器', 'tool': 'execute_python', 'args': lambda: {'code': 'from http.server import HTTPServer, SimpleHTTPRequestHandler\nserver = HTTPServer(("localhost", 8000), SimpleHTTPRequestHandler)\nprint("服务器启动在 http://localhost:8000")\n# server.serve_forever()'}, 'thought': '用户需要Web服务器，我用Python标准库实现。', 'obs': '服务器启动在 http://localhost:8000', 'answer': 'Web服务器代码已生成，使用Python标准库http.server。\n\n运行方法：\n1. 保存代码为server.py\n2. 执行python server.py\n3. 访问http://localhost:8000'},
    {'task': '帮我调试这段代码的错误', 'tool': 'execute_python', 'args': lambda: {'code': 'def divide(a, b):\n    return a / b\n\ntry:\n    result = divide(10, 0)\nexcept ZeroDivisionError as e:\n    print(f"错误：{e}")'}, 'thought': '用户代码有错误，我执行看看什么问题。', 'obs': '错误：division by zero', 'answer': '代码错误是除以零（ZeroDivisionError）。\n\n修复方法：\n1. 添加参数检查：if b == 0: return None\n2. 使用try-except捕获异常\n3. 使用math.inf处理无穷大'},
]

# 生成样本
samples = []
for task_info in TASKS:
    try:
        args = task_info['args']()
        args_str = json.dumps(args, ensure_ascii=False) if args else ''
        obs = task_info['obs']
        answer = task_info['answer'].replace('{obs}', obs)

        # 处理任务描述中的变量
        task = task_info['task']
        for key in ['topic', 'topic2', 'func', 'expr', 'info', 'num', 'num1', 'num2', 'url']:
            if f'{{{key}}}' in task:
                task = task.replace(f'{{{key}}}', '...')

        thought = task_info['thought']
        for key in ['topic', 'topic2']:
            if f'{{{key}}}' in thought:
                thought = thought.replace(f'{{{key}}}', '...')

        sample = {
            'messages': [
                {'role': 'system', 'content': '你是态极，一个会使用工具的AI助手。当需要使用工具时，用思考和工具调用的方式完成任务。'},
                {'role': 'user', 'content': task},
                {'role': 'assistant', 'content': f'<think>{thought}</think><tool_call>{task_info["tool"]}({args_str})</tool_call>'},
                {'role': 'system', 'content': f'<tool_result>{obs}</tool_result>'},
                {'role': 'assistant', 'content': answer},
            ]
        }
        samples.append(sample)
    except Exception as e:
        print(f'Error processing task: {e}')

# 生成变体 - 用不同的问法问同样的问题
variants = []
variant_templates = [
    ('帮我查一下{query}', 'search'),
    ('我想知道{query}', 'search'),
    ('{query}怎么学', 'search'),
    ('读取文件{path}', 'read_file'),
    ('看看{path}里面有什么', 'read_file'),
    ('执行代码{code}', 'execute_python'),
    ('运行一下{code}', 'execute_python'),
    ('记住{info}', 'remember'),
    ('保存这个信息：{info}', 'remember'),
    ('现在几点', 'get_current_time'),
    ('当前时间是什么', 'get_current_time'),
    ('算一下{expr}', 'calculate'),
    ('{expr}等于多少', 'calculate'),
]

queries = ['Python教程', '机器学习入门', 'Git命令', 'Docker使用', 'React框架']
paths = ['config.json', 'README.md', 'src/main.py', 'data.csv']
codes = ['print("hello")', 'import sys; print(sys.version)', 'sum(range(100))']
infos = ['我喜欢编程', '我的名字是小明', '今天天气很好']
exprs = ['2+3', '10*20', '100/3', '2**10']

for template, tool in variant_templates:
    for _ in range(3):  # 每个模板生成3个变体
        task = template
        args = {}

        if '{query}' in task:
            q = random.choice(queries)
            task = task.replace('{query}', q)
            args = {'query': q}
        elif '{path}' in task:
            p = random.choice(paths)
            task = task.replace('{path}', p)
            args = {'path': p}
        elif '{code}' in task:
            c = random.choice(codes)
            task = task.replace('{code}', c)
            args = {'code': c}
        elif '{info}' in task:
            i = random.choice(infos)
            task = task.replace('{info}', i)
            args = {'content': i} if tool == 'remember' else {'query': i}
        elif '{expr}' in task:
            e = random.choice(exprs)
            task = task.replace('{expr}', e)
            args = {'expression': e}

        sample = {
            'messages': [
                {'role': 'system', 'content': '你是态极，一个会使用工具的AI助手。'},
                {'role': 'user', 'content': task},
                {'role': 'assistant', 'content': f'<think>我需要使用{tool}工具。</think><tool_call>{tool}({json.dumps(args, ensure_ascii=False)})</tool_call>'},
                {'role': 'system', 'content': '<tool_result>操作完成</tool_result>'},
                {'role': 'assistant', 'content': '已完成你的请求。'},
            ]
        }
        variants.append(sample)

all_samples = samples + variants

# 保存
with open(output_path, 'w', encoding='utf-8') as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

print(f'Generated {len(samples)} main samples + {len(variants)} variants = {len(all_samples)} total -> {output_path}')
