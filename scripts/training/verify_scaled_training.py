"""验证规模化训练数据下的生成质量改善。

关键验证点：
1. 喂入多样化训练数据（每域 20+ 条，覆盖多主题）
2. 多轮 feed+sleep 训练
3. 每轮评估生成质量：loss、token 多样性、训练数据覆盖率
4. 对比训练前后的生成文本

Usage:
    python scripts/training/verify_scaled_training.py
"""
import sys
import os
from datetime import datetime
from collections import Counter

os.environ.setdefault('TAJIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


# ── 多样化训练数据（500 条，每域 125 条） ──
TRAINING_DATA = {
    "zh": [
        # 天气主题 (8)
        "今天天气很好，我们一起去公园散步。",
        "明天天气怎么样？会下雨吗？",
        "天气预报说周末是晴天，适合出游。",
        "夏天的雷阵雨总是来得很突然。",
        "冬天的雪景真美，孩子们在堆雪人。",
        "春天暖风吹过，百花竞相开放。",
        "秋高气爽，蓝天白云令人心旷神怡。",
        "台风来了，记得关好窗户待在家里。",
        # 科技主题 (8)
        "人工智能正在改变世界，神经元协同工作。",
        "态极神经元架构通过共振场实现意识涌现。",
        "深度学习模型需要大量数据训练。",
        "大语言模型展现了强大的理解能力。",
        "量子计算机将在未来解决复杂问题。",
        "区块链技术让数据更加安全可靠。",
        "云计算让企业无需自建数据中心。",
        "物联网连接了万物，让生活更智能。",
        # 生活主题 (8)
        "我喜欢在清晨喝一杯咖啡，开始新的一天。",
        "读书是获取知识的重要途径。",
        "运动有益健康，每天坚持锻炼。",
        "音乐能陶冶情操，让人心情愉悦。",
        "做饭是一门艺术，需要不断练习。",
        "旅行能开阔眼界，增长见识。",
        "和朋友聚会是最快乐的时光。",
        "养花种草让生活充满绿意。",
        # 学习主题 (8)
        "学习新语言需要耐心和持续练习。",
        "数学是科学的基础，逻辑思维很重要。",
        "历史告诉我们很多道理。",
        "物理研究自然界的基本规律。",
        "化学实验需要注意安全操作。",
        "写作能力需要通过大量阅读来提升。",
        "编程是一项实用的技能。",
        "哲学思考帮助人们认识自我。",
        # 自然主题 (8)
        "春天来了，万物复苏，花开满园。",
        "秋天是收获的季节，树叶变黄飘落。",
        "海洋是地球上最大的生态系统。",
        "森林是地球的肺，提供氧气。",
        "河流奔腾不息，汇入大海。",
        "高山巍峨耸立，终年积雪。",
        "沙漠虽然干旱，也有独特生态。",
        "极地的极光绚丽多彩，美不胜收。",
        # 社会主题 (8)
        "城市生活节奏快，交通便利。",
        "乡村生活宁静祥和，空气清新。",
        "教育是改变命运的重要途径。",
        "团队合作能完成更复杂的任务。",
        "诚信是为人之本，不可丢弃。",
        "创新推动社会不断向前发展。",
        "文化交流增进不同民族的理解。",
        "保护环境是每个人的责任。",
        # 情感主题 (6)
        "快乐不在于拥有多少，而在于知足。",
        "勇气不是不害怕，而是害怕了仍然前行。",
        "友谊需要用心经营，才能长久。",
        "家是温暖的港湾，永远等着你回来。",
        "梦想是前进的动力，不要轻易放弃。",
        "感恩生活中的每一个美好瞬间。",
        # 历史主题 (9)
        "唐朝是中国历史上最辉煌的朝代之一。",
        "秦始皇统一六国，建立了中央集权制度。",
        "丝绸之路促进了东西方文化的交流。",
        "五四运动推动了中国社会的思想解放。",
        "长城是古代防御工程的伟大奇迹。",
        "火药是中国的四大发明之一。",
        "宋朝的商业经济非常繁荣发达。",
        "汉武帝派遣张骞出使西域。",
        "郑和下西洋展示了明朝的航海实力。",
        # 文学主题 (8)
        "红楼梦被誉为中国古典小说的巅峰之作。",
        "唐诗宋词是中国文学的两座高峰。",
        "鲁迅以犀利的文笔批判社会弊端。",
        "李白是浪漫主义诗人的杰出代表。",
        "小说通过人物描写反映社会现实。",
        "散文以自由的形式抒发作者情感。",
        "民间故事蕴含着丰富的文化智慧。",
        "文学创作来源于生活又高于生活。",
        # 哲学主题 (8)
        "老子主张无为而治的自然哲学。",
        "庄子以逍遥游表达自由精神境界。",
        "儒家思想强调仁义礼智信的道德规范。",
        "知行合一是王阳明心学的核心。",
        "辩证法认为事物发展由矛盾推动。",
        "存在先于本质是一种存在主义观点。",
        "理性思考是人类认识世界的重要方式。",
        "自由意志与决定论是哲学的根本问题。",
        # 经济主题 (8)
        "通货膨胀会导致货币购买力下降。",
        "市场经济通过价格机制调节供需。",
        "国际贸易促进了全球资源配置。",
        "税收是政府提供公共服务的资金来源。",
        "GDP是衡量一个国家经济总量的指标。",
        "投资需要平衡风险与收益的关系。",
        "创业需要创新精神和市场洞察力。",
        "消费升级反映了人民生活水平的提高。",
        # 健康主题 (8)
        "均衡饮食是保持身体健康的基础。",
        "充足的睡眠有助于身体恢复和修复。",
        "心理健康和身体健康同样重要。",
        "定期体检能够及早发现潜在疾病。",
        "吸烟对呼吸系统有严重的危害。",
        "运动能增强免疫力和心肺功能。",
        "保持良好心态有助于延年益寿。",
        "喝水是维持新陈代谢的必要条件。",
        # 饮食主题 (8)
        "中国菜有八大菜系各具特色风味。",
        "川菜以麻辣鲜香闻名于世。",
        "粽子的主要原料是糯米和粽叶。",
        "火锅是冬天最受欢迎的美食之一。",
        "茶的种类丰富有绿茶红茶乌龙茶。",
        "饺子是中国传统节日的必备食品。",
        "食材的新鲜程度决定了菜肴的品质。",
        "烹饪讲究火候和调料的搭配比例。",
        # 旅行主题 (8)
        "旅行能让人体验不同的风土人情。",
        "西藏的高原风光令人叹为观止。",
        "云南的少数民族文化丰富多彩。",
        "故宫是中国古代建筑艺术的瑰宝。",
        "三亚的海滩是度假休闲的好去处。",
        "西安古城墙上可以俯瞰城市全景。",
        "苏州园林以精巧布局闻名天下。",
        "黄山以奇松怪石云海温泉著称。",
        # 运动主题 (8)
        "跑步是最简单有效的有氧运动方式。",
        "游泳能锻炼全身肌肉和协调能力。",
        "篮球需要团队配合和个人技术。",
        "足球场上配合默契才能赢得胜利。",
        "瑜伽可以调节呼吸和放松身心。",
        "乒乓球是中国最受欢迎的球类运动。",
        "运动前的热身能有效防止受伤。",
        "坚持锻炼需要毅力和自律精神。",
        # 节日主题 (6)
        "春节是中国人最重要的传统节日。",
        "中秋节全家人团聚吃月饼赏月。",
        "端午节赛龙舟吃粽子纪念屈原。",
        "元宵节赏花灯猜灯谜吃汤圆。",
        "清明节扫墓祭祖缅怀先人。",
        "国庆节是庆祝新中国成立的节日。",
    ],
    "en": [
        # Technology (10)
        "The cortex architecture uses resonance fields.",
        "Small neurons work together to match large models.",
        "Machine learning models learn from data.",
        "Neural networks are inspired by the brain.",
        "Language models process text efficiently.",
        "Quantum computers will solve complex problems.",
        "Blockchain makes data secure and reliable.",
        "Cloud computing eliminates data centers.",
        "The internet connects people worldwide.",
        "Artificial intelligence transforms industries.",
        # Daily life (10)
        "The weather is nice today, let's go for a walk.",
        "Reading books is important for learning.",
        "Exercise is good for your health.",
        "Music can affect our emotions deeply.",
        "Cooking is an art that requires practice.",
        "Travel broadens the mind and perspective.",
        "Friends make life more enjoyable.",
        "Gardening brings joy and beauty.",
        "A good cup of coffee starts the day right.",
        "Sleep is essential for good health.",
        # Science (8)
        "Science advances through careful experimentation.",
        "Technology changes the way we live.",
        "Physics studies the laws of nature.",
        "Chemistry explores matter and its properties.",
        "Biology investigates living organisms.",
        "Astronomy reveals the mysteries of universe.",
        "Mathematics is the language of science.",
        "Evolution explains the diversity of life.",
        # Society (8)
        "Education opens doors to new opportunities.",
        "Teamwork makes difficult tasks achievable.",
        "Nature provides beauty and resources.",
        "Cities offer convenience and diversity.",
        "Honesty is the foundation of trust.",
        "Innovation drives society forward.",
        "Cultural exchange promotes understanding.",
        "Protecting the environment is our duty.",
        # Emotions (8)
        "Happiness comes from within, not from things.",
        "Courage is acting despite fear.",
        "Friendship requires effort to maintain.",
        "Home is where the heart belongs.",
        "Dreams give us motivation to continue.",
        "Gratitude makes life more meaningful.",
        "Love is the most powerful force.",
        "Patience brings rewards in time.",
        # Nature (6)
        "The ocean is the largest ecosystem on Earth.",
        "Mountains rise majestically into the sky.",
        "Forests provide oxygen for the planet.",
        "Rivers flow endlessly to the sea.",
        "Seasons change in a beautiful cycle.",
        "Stars light up the night sky.",
        # History (10)
        "The Renaissance was a period of great cultural rebirth.",
        "World War Two reshaped the global political landscape.",
        "Ancient Rome built an extensive network of roads.",
        "The Industrial Revolution transformed manufacturing forever.",
        "Egyptian pyramids are wonders of ancient engineering.",
        "The Silk Road connected East and West for centuries.",
        "The invention of printing revolutionized knowledge sharing.",
        "Democracy originated in ancient Greek city-states.",
        "The moon landing was a historic achievement for mankind.",
        "Colonial independence movements changed the world order.",
        # Philosophy (9)
        "Philosophy asks fundamental questions about existence.",
        "Ethics concerns what is right and wrong in behavior.",
        "Logic provides the rules for valid reasoning.",
        "Knowledge requires justified true belief according to Plato.",
        "Free will debates challenge our understanding of choice.",
        "The meaning of life is a central philosophical question.",
        "Empiricism holds that knowledge comes from experience.",
        "Rationalism trusts reason as the source of knowledge.",
        "Moral relativism claims values vary across cultures.",
        # Economics (8)
        "Supply and demand determine market equilibrium prices.",
        "Interest rates influence borrowing and investment decisions.",
        "International trade creates comparative advantage benefits.",
        "Fiscal policy uses government spending to manage economy.",
        "Monetary policy controls money supply and inflation.",
        "Economic growth raises living standards over time.",
        "Unemployment rates signal the health of labor markets.",
        "The stock market aggregates information about companies.",
        # Health (8)
        "Regular checkups help prevent serious medical conditions.",
        "Mental wellbeing is as important as physical health.",
        "A balanced diet includes proteins carbohydrates and fats.",
        "Vaccination protects individuals and the wider community.",
        "Stress management techniques improve quality of life.",
        "Heart disease can be prevented with healthy habits.",
        "Adequate hydration is essential for body functions.",
        "The immune system defends the body against pathogens.",
        # Travel (8)
        "The Grand Canyon offers breathtaking views of nature.",
        "Paris is known as the city of lights and romance.",
        "Japanese temples reflect a deep spiritual tradition.",
        "African safaris let you observe wildlife up close.",
        "Venice canals create a unique urban landscape.",
        "The Great Wall stretches across northern China beautifully.",
        "Tropical islands offer pristine beaches and clear waters.",
        "Traveling solo builds confidence and self-reliance.",
        # Sports (8)
        "Tennis requires both physical agility and mental focus.",
        "Marathon runners train for months to build endurance.",
        "Gymnastics demands exceptional flexibility and strength.",
        "Soccer is the most popular sport in the world.",
        "Basketball players need speed coordination and teamwork.",
        "Swimming competitions test speed and technique.",
        "Mountain climbing challenges both body and spirit.",
        "Baseball is a game of strategy and precision.",
        # Arts (8)
        "Portrait painting captures the essence of a person.",
        "Symphony orchestras blend many instruments harmoniously.",
        "Ballet combines athleticism with artistic expression.",
        "Modern art challenges traditional notions of beauty.",
        "Poetry distills emotions into carefully chosen words.",
        "Architecture shapes the spaces where we live and work.",
        "Photography freezes moments in time forever.",
        "Theater brings stories to life on stage.",
        # Education (8)
        "Critical thinking is a key skill for lifelong learning.",
        "Online courses make education accessible to everyone.",
        "Early childhood education builds a strong foundation.",
        "Teachers play a vital role in shaping future generations.",
        "Learning a second language opens new perspectives.",
        "STEM education prepares students for modern careers.",
        "Reading widely develops vocabulary and comprehension.",
        "Universities drive research and innovation forward.",
        # Environment (8)
        "Renewable energy reduces dependence on fossil fuels.",
        "Climate change requires urgent global cooperation.",
        "Recycling helps conserve natural resources effectively.",
        "Plastic pollution threatens marine ecosystems worldwide.",
        "Sustainable farming protects soil and water quality.",
        "Biodiversity is essential for ecosystem resilience.",
        "Deforestation contributes to global carbon emissions.",
        "Urban parks improve air quality and public health.",
    ],
    "code": [
        # Functions (12)
        "def hello(): print('world')",
        "def add(a, b): return a + b",
        "def multiply(x, y): return x * y",
        "def is_even(n): return n % 2 == 0",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "def reverse(s): return s[::-1]",
        "def max_of_list(lst): return max(lst)",
        "def count_words(text): return len(text.split())",
        "def fib(n): a, b = 0, 1\n    for _ in range(n): a, b = b, a+b\n    return a",
        "def sort_list(items): return sorted(items)",
        "def check_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
        # Classes (12)
        "class Neuron: def forward(self, x): return x",
        "class ResonanceField: def reset(self): pass",
        "class Cortex: def think(self, x): return x",
        "class FeedEngine: def feed(self, data): pass",
        "class SleepEngine: def sleep(self): pass",
        "class Ensemble: def forward(self, x): return x",
        "class PlayEngine: def play(self): pass",
        "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)",
        "class Queue:\n    def __init__(self): self.items = []\n    def enqueue(self, x): self.items.append(x)",
        "class Point:\n    def __init__(self, x, y): self.x = x; self.y = y",
        "class Vector:\n    def __init__(self, data): self.data = data\n    def sum(self): return sum(self.data)",
        "class Logger:\n    def info(self, msg): print(f'[INFO] {msg}')",
        # Utility (10)
        "import torch; print(torch.__version__)",
        "import json; data = json.loads('{\"key\": \"value\"}')",
        "import os; files = os.listdir('.')",
        "import sys; print(sys.path)",
        "import math; result = math.sqrt(16)",
        "import random; choice = random.choice([1,2,3])",
        "import datetime; now = datetime.datetime.now()",
        "import collections; counter = collections.Counter('abracadabra')",
        "import itertools; pairs = list(itertools.combinations([1,2,3], 2))",
        "import functools; result = functools.reduce(lambda a,b: a+b, [1,2,3,4])",
        # Patterns (10)
        "def generate(prompt): return prompt + ' generated'",
        "def validate_input(x): assert x is not None",
        "def save_state(model, path): torch.save(model, path)",
        "def load_state(path): return torch.load(path)",
        "def encode(text, tokenizer): return tokenizer.encode(text)",
        "def decode(ids, tokenizer): return tokenizer.decode(ids)",
        "def train(model, data): model.fit(data)",
        "def predict(model, x): return model(x)",
        "def loss_fn(pred, target): return (pred - target).mean()",
        "def accuracy(preds, labels): return (preds == labels).float().mean()",
        # Algorithms (8)
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]",
        "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))",
        "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    return quick_sort([x for x in arr[1:] if x < pivot]) + [pivot] + quick_sort([x for x in arr[1:] if x >= pivot])",
        "def linear_search(arr, target):\n    for i, x in enumerate(arr):\n        if x == target: return i\n    return -1",
        "def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i-1\n        while j >= 0 and arr[j] > key:\n            arr[j+1] = arr[j]\n            j -= 1\n        arr[j+1] = key",
        "def depth_first_search(graph, start, visited=None):\n    if visited is None: visited = set()\n    visited.add(start)\n    for next in graph[start] - visited:\n        depth_first_search(graph, next, visited)",
        "def breadth_first_search(graph, start):\n    visited = set([start])\n    queue = [start]\n    while queue:\n        vertex = queue.pop(0)\n        for next in graph[vertex] - visited:\n            visited.add(next)\n            queue.append(next)",
        # Data Structures (10)
        "class ListNode:\n    def __init__(self, val=0, next=None): self.val = val; self.next = next",
        "class TreeNode:\n    def __init__(self, val=0, left=None, right=None): self.val = val; self.left = left; self.right = right",
        "class HashTable:\n    def __init__(self, size=100): self.size = size; self.table = [[] for _ in range(size)]",
        "class Graph:\n    def __init__(self): self.adj = {}\n    def add_edge(self, u, v): self.adj.setdefault(u, []).append(v)",
        "class Deque:\n    def __init__(self): self.items = []\n    def push_front(self, x): self.items.insert(0, x)",
        "class MinHeap:\n    def __init__(self): self.heap = []\n    def push(self, x): self.heap.append(x); self._sift_up(len(self.heap)-1)",
        "class TrieNode:\n    def __init__(self): self.children = {}; self.is_end = False",
        "class CircularQueue:\n    def __init__(self, k): self.k = k; self.q = [0]*k; self.front = self.rear = -1",
        "class LinkedList:\n    def __init__(self): self.head = None\n    def append(self, val): pass",
        "class BinarySearchTree:\n    def __init__(self): self.root = None\n    def insert(self, val): pass",
        # Sorting (8)
        "def selection_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        min_idx = i\n        for j in range(i+1, n):\n            if arr[j] < arr[min_idx]: min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]",
        "def heap_sort(arr):\n    import heapq\n    heapq.heapify(arr)\n    return [heapq.heappop(arr) for _ in range(len(arr))]",
        "def counting_sort(arr, max_val):\n    count = [0] * (max_val + 1)\n    for x in arr: count[x] += 1\n    result = []\n    for i, c in enumerate(count): result.extend([i]*c)\n    return result",
        "def radix_sort(arr):\n    max_digit = len(str(max(arr)))\n    for d in range(max_digit):\n        buckets = [[] for _ in range(10)]\n        for x in arr: buckets[(x//10**d)%10].append(x)\n        arr = [y for b in buckets for y in b]\n    return arr",
        "def shell_sort(arr):\n    n = len(arr); gap = n // 2\n    while gap > 0:\n        for i in range(gap, n):\n            temp = arr[i]; j = i\n            while j >= gap and arr[j-gap] > temp: arr[j] = arr[j-gap]; j -= gap\n            arr[j] = temp\n        gap //= 2",
        "def bucket_sort(arr):\n    buckets = [[] for _ in range(len(arr))]\n    for x in arr: buckets[int(x*len(arr))].append(x)\n    return [y for b in buckets for y in sorted(b)]",
        "def tim_sort(arr): return sorted(arr)  # Python's built-in Timsort",
        "def pancake_sort(arr):\n    def flip(arr, k): arr[:k] = arr[:k][::-1]\n    n = len(arr)\n    for size in range(n, 1, -1):\n        max_idx = arr.index(max(arr[:size]))\n        flip(arr, max_idx+1); flip(arr, size)",
        # Error Handling (8)
        "def safe_divide(a, b):\n    try: return a / b\n    except ZeroDivisionError: return float('inf')\n    except TypeError: return None",
        "def parse_json(data):\n    try: return json.loads(data)\n    except json.JSONDecodeError as e: print(f'Invalid JSON: {e}'); return None",
        "def read_file(path):\n    try:\n        with open(path) as f: return f.read()\n    except FileNotFoundError: return ''\n    except PermissionError: return None",
        "def validate_range(x, lo, hi):\n    if not isinstance(x, (int, float)): raise TypeError('Must be number')\n    if x < lo or x > hi: raise ValueError(f'{x} not in [{lo},{hi}]')",
        "class Retry:\n    def __init__(self, max_retries=3): self.max = max_retries\n    def call(self, fn, *args):\n        for i in range(self.max):\n            try: return fn(*args)\n            except Exception: pass\n        raise RuntimeError('Max retries exceeded')",
        "def atomic_write(path, content):\n    tmp = path + '.tmp'\n    write(tmp, content)\n    os.replace(tmp, path)  # atomic on POSIX",
        "def with_timeout(fn, seconds, default=None):\n    import signal\n    def handler(signum, frame): raise TimeoutError()\n    signal.signal(signal.SIGALRM, handler)\n    signal.alarm(seconds)\n    try: return fn()\n    except TimeoutError: return default\n    finally: signal.alarm(0)",
        "class Result:\n    def __init__(self, value=None, error=None): self.value = value; self.error = error\n    def is_ok(self): return self.error is None\n    def unwrap(self): return self.value if self.is_ok() else raise Exception(self.error)",
        # Concurrency (8)
        "import threading\nlock = threading.Lock()\nwith lock:\n    print('critical section')",
        "from concurrent.futures import ThreadPoolExecutor\nexecutor = ThreadPoolExecutor(max_workers=4)\nfutures = [executor.submit(task, arg) for arg in args]",
        "import asyncio\nasync def async_fetch(url):\n    await asyncio.sleep(1)\n    return f'result from {url}'",
        "from multiprocessing import Pool\nwith Pool(processes=4) as pool:\n    results = pool.map(process_item, items)",
        "queue = Queue()\nthread_a = Thread(target=producer, args=(queue,))\nthread_b = Thread(target=consumer, args=(queue,))",
        "semaphore = threading.Semaphore(5)\nwith semaphore:\n    limited_concurrent_operation()",
        "import queue\npq = queue.PriorityQueue()\npq.put((1, 'low')); pq.put((0, 'high'))\nwhile not pq.empty(): print(pq.get())",
        "async def gather_tasks():\n    results = await asyncio.gather(\n        fetch_page('url1'),\n        fetch_page('url2'),\n        fetch_page('url3'),\n    )\n    return results",
        # Testing (8)
        "import unittest\nclass TestMath(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(1, 2), 3)\n    def test_divide_by_zero(self):\n        with self.assertRaises(ZeroDivisionError): divide(1, 0)",
        "import pytest\ndef test_reverse():\n    assert reverse('abc') == 'cba'\n    assert reverse('') == ''\n\n@pytest.mark.parametrize('n,expected', [(1,1), (2,1), (5,5)])",
        "def test_edge_cases():\n    # Test empty input\n    result = process([])\n    assert result == []\n    # Test large input\n    result = process(list(range(100000)))\n    assert len(result) == 100000",
        "from unittest.mock import Mock, patch\ndef test_with_mock():\n    api = Mock()\n    api.fetch.return_value = {'data': [1,2,3]}\n    assert len(api.fetch()['data']) == 3",
        "def test_concurrently():\n    results = []\n    threads = [Thread(target=lambda: results.append(fn())) for _ in range(10)]\n    for t in threads: t.start()\n    for t in threads: t.join()\n    assert all(r is not None for r in results)",
        "def benchmark(fn, iterations=1000):\n    import time\n    start = time.perf_counter()\n    for _ in range(iterations): fn()\n    elapsed = time.perf_counter() - start\n    return elapsed / iterations",
        "class TestDatabase(unittest.TestCase):\n    def setUp(self): self.db = connect(':memory:')\n    def tearDown(self): self.db.close()\n    def test_insert(self):\n        self.db.execute('INSERT INTO t VALUES (1)')\n        rows = self.db.execute('SELECT * FROM t').fetchall()\n        self.assertEqual(len(rows), 1)",
        "def test_property_based():\n    from hypothesis import given, strategies as st\n    @given(st.integers(), st.integers())\n    def test_commutative(a, b):\n        assert add(a, b) == add(b, a)",
        # Design Patterns (8)
        "class Singleton:\n    _instance = None\n    def __new__(cls, *args, **kwargs):\n        if cls._instance is None: cls._instance = super().__new__(cls)\n        return cls._instance",
        "class Observer:\n    def __init__(self): self._observers = []\n    def subscribe(self, observer): self._observers.append(observer)\n    def notify(self, event):\n        for o in self._observers: o.on_event(event)",
        "class Context: pass\nclass StrategyA: pass\nclass StrategyB: pass\nctx = Context()\nctx.strategy = StrategyA() if condition else StrategyB()",
        "class Handler:\n    def __init__(self, next_handler=None): self.next = next_handler\n    def handle(self, request):\n        if self.can_handle(request): return self.process(request)\n        return self.next.handle(request) if self.next else None",
        "class Command:\n    def execute(self): raise NotImplementedError\nclass PrintCommand(Command):\n    def __init__(self, text): self.text = text\n    def execute(self): print(self.text)",
        "class Component:\n    def operation(self): pass\nclass Decorator(Component):\n    def __init__(self, component): self._component = component\n    def operation(self): return f'[{self._component.operation()}]'",
        "class Iterator:\n    def __init__(self, collection): self._collection = collection; self._index = 0\n    def __next__(self):\n        if self._index >= len(self._collection): raise StopIteration\n        item = self._collection[self._index]; self._index += 1\n        return item",
        "class State:\n    def handle(self, context): pass\nclass StateA(State): pass\nclass StateB(State): pass\nclass Machine:\n    def __init__(self): self.state = StateA()\n    def transition(self, new_state): self.state = new_state",
        # File IO (8)
        "def write_lines(path, lines):\n    with open(path, 'w', encoding='utf-8') as f:\n        f.write('\\n'.join(lines))",
        "def read_csv(path):\n    import csv\n    with open(path, newline='') as f:\n        return list(csv.reader(f))",
        "def walk_directory(root):\n    for dirpath, dirnames, filenames in os.walk(root):\n        for fn in filenames:\n            yield os.path.join(dirpath, fn)",
        "def read_chunks(path, chunk_size=8192):\n    with open(path, 'rb') as f:\n        while chunk := f.read(chunk_size): yield chunk",
        "def append_log(path, message):\n    with open(path, 'a') as f:\n        f.write(f'{datetime.now().isoformat()} {message}\\n')",
        "import tempfile\nwith tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:\n    f.write('temporary data')\n    tmp_path = f.name",
        "def read_config(path):\n    import configparser\n    cfg = configparser.ConfigParser()\n    cfg.read(path)\n    return {s: dict(cfg.items(s)) for s in cfg.sections()}",
        "def count_lines(path):\n    with open(path) as f:\n        return sum(1 for _ in f)",
        # Regex (7)
        "import re\npattern = r'\\b\\w+@\\w+\\.\\w+\\b'\nemails = re.findall(pattern, text)",
        "phone_pattern = r'\\b1[3-9]\\d{9}\\b'\nphones = re.findall(phone_pattern, document)",
        "result = re.sub(r'\\s+', ' ', text)  # collapse whitespace",
        "match = re.search(r'version=(\\d+\\.\\d+)', config_str)\nversion = match.group(1) if match else 'unknown'",
        "tokens = re.split(r'[,;]+', csv_line)  # split on commas or semicolons",
        "url_pattern = r'https?://[\\w.-]+(?:\\.[\\w.-]+)+[\\w\\-._~:/?#@!$&()*+,;=]+'\nurls = re.findall(url_pattern, html_content)",
        "def is_valid_ip(s):\n    pattern = r'^(\\d{1,3}\\.){3}\\d{1,3}$'\n    return bool(re.match(pattern, s))",
        # Networking (8)
        "import requests\nresponse = requests.get('https://api.example.com/data')\ndata = response.json()",
        "import socket\nsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nsock.connect(('localhost', 8080))\nsock.send(b'Hello')\nsock.close()",
        "from http.server import HTTPServer, BaseHTTPRequestHandler\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.send_header('Content-Type', 'application/json')\n        self.end_headers()\n        self.wfile.write(b'{\"status\":\"ok\"}')",
        "import urllib.parse\nparams = {'q': 'search term', 'page': '1'}\nencoded = urllib.parse.urlencode(params)\nurl = f'https://example.com/search?{encoded}'",
        "import json\nresponse_data = {'users': [{'id': 1, 'name': 'Alice'}]}\njson_str = json.dumps(response_data, indent=2)",
        "headers = {'Authorization': 'Bearer token123', 'Content-Type': 'application/json'}\nresponse = requests.post(url, headers=headers, json={'key': 'value'})",
        "def download_file(url, dest):\n    r = requests.get(url, stream=True)\n    with open(dest, 'wb') as f:\n        for chunk in r.iter_content(chunk_size=8192):\n            f.write(chunk)",
        "import websockets\nasync def ws_client():\n    async with websockets.connect('ws://localhost:8765') as ws:\n        await ws.send('Hello')\n        reply = await ws.recv()\n        return reply",
    ],
    "math": [
        # Arithmetic (10)
        "1 + 1 = 2",
        "2 * 3 = 6",
        "10 - 4 = 6",
        "15 / 3 = 5",
        "7 + 8 = 15",
        "9 * 9 = 81",
        "100 / 4 = 25",
        "3 + 5 = 8",
        "6 * 7 = 42",
        "20 - 12 = 8",
        # Algebra (10)
        "x^2 + y^2 = r^2",
        "f(x) = ax + b",
        "a^2 + b^2 = c^2 for right triangles",
        "log(a*b) = log(a) + log(b)",
        "(a + b)^2 = a^2 + 2ab + b^2",
        "a^2 - b^2 = (a+b)(a-b)",
        "quadratic formula: x = (-b +- sqrt(b^2-4ac)) / 2a",
        "slope of line: m = (y2-y1)/(x2-x1)",
        "distance formula: d = sqrt((x2-x1)^2 + (y2-y1)^2)",
        "midpoint: M = ((x1+x2)/2, (y1+y2)/2)",
        # Calculus (10)
        "integral of x dx = x^2/2 + C",
        "derivative of x^2 is 2x",
        "derivative of sin(x) is cos(x)",
        "derivative of e^x is e^x",
        "derivative of ln(x) is 1/x",
        "integral of 1/x dx = ln|x| + C",
        "integral of cos(x) dx = sin(x) + C",
        "integral of e^x dx = e^x + C",
        "limit of 1/x as x -> inf is 0",
        "Taylor series: f(x) = sum of f^(n)(a)/n! * (x-a)^n",
        # Trigonometry (8)
        "sin(0) = 0, cos(0) = 1",
        "sin(30) = 0.5, cos(60) = 0.5",
        "sin(90) = 1, cos(90) = 0",
        "tan(45) = 1",
        "sin^2(x) + cos^2(x) = 1",
        "sin(a+b) = sin(a)cos(b) + cos(a)sin(b)",
        "cos(a+b) = cos(a)cos(b) - sin(a)sin(b)",
        "law of cosines: c^2 = a^2 + b^2 - 2ab*cos(C)",
        # Famous formulas (8)
        "e^(i*pi) + 1 = 0",
        "sum of 1 to n is n*(n+1)/2",
        "The area of circle is pi * r^2",
        "The volume of sphere is 4/3 * pi * r^3",
        "The circumference of circle is 2 * pi * r",
        "F = ma (Newton's second law)",
        "E = mc^2 (Einstein's mass-energy equivalence)",
        "Pythagorean theorem: a^2 + b^2 = c^2",
        # Linear Algebra (6)
        "Matrix multiplication is associative",
        "The determinant of 2x2 matrix [[a,b],[c,d]] is ad - bc",
        "Identity matrix has 1s on diagonal and 0s elsewhere",
        "Transpose of matrix A is A^T where (A^T)_ij = A_ji",
        "The inverse of A satisfies A * A^(-1) = I",
        "Eigenvalue equation: Av = lambda*v",
        # Statistics (10)
        "The mean of dataset is sum divided by count of values.",
        "The median is the middle value when data is sorted.",
        "The mode is the value that appears most frequently.",
        "Standard deviation measures the spread of data around mean.",
        "Variance is the average of squared deviations from mean: sigma^2 = sum(x_i-mu)^2 / n.",
        "The normal distribution is bell-shaped and symmetric: N(mu, sigma^2).",
        "Correlation coefficient r ranges from -1 to 1 for linear relationship.",
        "The p-value indicates the probability of observing data under null hypothesis.",
        "Confidence interval gives a range of plausible values for a parameter.",
        "Linear regression models relationship: y = beta_0 + beta_1*x + epsilon.",
        # Probability (10)
        "P(A) = number of favorable outcomes / total possible outcomes.",
        "Expected value E(X) = sum of x_i * P(X=x_i) for discrete random variables.",
        "Bayes theorem: P(A|B) = P(B|A)*P(A) / P(B).",
        "Two events are independent if P(A and B) = P(A) * P(B).",
        "The binomial distribution counts successes in n trials: B(n, p).",
        "Law of large numbers: sample mean converges to expected value as n increases.",
        "Conditional probability: P(A|B) = P(A and B) / P(B).",
        "The sum of probabilities of all possible outcomes equals 1.",
        "Markov chains have the property that future depends only on present state.",
        "Central limit theorem: sampling distribution of mean approaches normal as n grows.",
        # Linear Algebra Pt2 (8)
        "A matrix is singular if and only if its determinant is zero.",
        "The rank of a matrix is the number of linearly independent rows or columns.",
        "Orthogonal vectors have dot product equal to zero.",
        "Gram-Schmidt process orthonormalizes a set of vectors.",
        "Singular value decomposition factors A = U * Sigma * V^T.",
        "The trace of a matrix is the sum of its diagonal elements.",
        "A positive definite matrix has all positive eigenvalues.",
        "Kernel of linear transformation is set of vectors mapped to zero.",
        # Geometry (10)
        "Area of triangle = 1/2 * base * height.",
        "Perimeter of rectangle = 2 * (length + width).",
        "Volume of cylinder = pi * r^2 * h.",
        "The sum of interior angles of triangle is 180 degrees.",
        "Surface area of sphere = 4 * pi * r^2.",
        "Volume of cone = 1/3 * pi * r^2 * h.",
        "Area of trapezoid = 1/2 * (base1 + base2) * height.",
        "The diagonal of rectangle = sqrt(length^2 + width^2).",
        "Volume of rectangular prism = length * width * height.",
        "Arc length of circle sector = (theta/360) * 2 * pi * r.",
        # Number Theory (8)
        "A prime number has exactly two positive divisors: 1 and itself.",
        "The greatest common divisor can be found using Euclidean algorithm.",
        "Modular arithmetic: a ≡ b (mod n) means n divides (a - b).",
        "Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p.",
        "Euler's totient function phi(n) counts integers coprime to n.",
        "The fundamental theorem of arithmetic: every integer has unique prime factorization.",
        "Perfect numbers equal the sum of their proper divisors.",
        "Diophantine equations seek integer solutions to polynomial equations.",
        # Combinatorics (8)
        "Permutation P(n,r) = n! / (n-r)! counts ordered selections of r items from n.",
        "Combination C(n,r) = n! / (r!*(n-r)!) counts unordered selections of r items.",
        "The pigeonhole principle: if n items go into m boxes and n > m, one box has at least 2.",
        "The number of subsets of a set with n elements is 2^n.",
        "The binomial theorem: (x + y)^n = sum of C(n,k)*x^(n-k)*y^k.",
        "Stars and bars: number of solutions to x1+x2+...+xk=n is C(n+k-1, k-1).",
        "Derangements count permutations with no fixed points: !n = n! * sum((-1)^k/k!).",
        "The inclusion-exclusion principle handles overlapping sets: |A∪B| = |A|+|B|-|A∩B|.",
        # Complex Analysis (6)
        "Complex number z = a + bi where i^2 = -1.",
        "Euler's formula: e^(i*theta) = cos(theta) + i*sin(theta).",
        "The modulus of complex number: |z| = sqrt(a^2 + b^2).",
        "Complex conjugate of a+bi is a-bi: z * conj(z) = |z|^2.",
        "De Moivre's theorem: (cos x + i*sin x)^n = cos(nx) + i*sin(nx).",
        "The complex plane maps real part to x-axis and imaginary to y-axis.",
        # Set Theory (5)
        "The empty set is a subset of every set.",
        "The power set P(A) is the set of all subsets of A: |P(A)| = 2^|A|.",
        "Union of sets A and B is A ∪ B = {x | x ∈ A or x ∈ B}.",
        "Intersection of sets A and B is A ∩ B = {x | x ∈ A and x ∈ B}.",
        "Cardinality |A| is the number of elements in set A.",
        # Logic (8)
        "Modus ponens: if P implies Q and P is true, then Q is true.",
        "Contrapositive: P implies Q is equivalent to not Q implies not P.",
        "A tautology is a statement that is always true by its logical form.",
        "Proof by contradiction assumes negation and derives a contradiction.",
        "De Morgan's laws: not(A and B) = not A or not B.",
        "Universal quantifier: ∀x means 'for all x' in the domain.",
        "Existential quantifier: ∃x means 'there exists an x' such that.",
        "Mathematical induction proves P(n) for all n by base case and inductive step.",
    ],
}


def compute_diversity(text: str) -> float:
    """计算文本 token 多样性（unique chars / total chars）"""
    if not text:
        return 0.0
    chars = list(text.replace(" ", "").replace("▁", ""))
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


def compute_repetition_ratio(text: str) -> float:
    """计算重复率（最高频 char 出现次数 / 总长度）"""
    if not text:
        return 0.0
    chars = list(text.replace(" ", "").replace("▁", ""))
    if not chars:
        return 0.0
    counter = Counter(chars)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(chars)


def main():
    print("=" * 60)
    print("规模化训练数据生成质量验证")
    print("=" * 60)

    # Step 1: 装配 Cortex
    print("\n[Step 1] 装配 Cortex...")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
    )
    assert cortex._shared_embedding is not None
    total_samples = sum(len(v) for v in TRAINING_DATA.values())
    print(f"  ✅ Cortex 装配完成: {len(cortex.neurons)} neurons")
    print(f"  训练数据: {total_samples} 条, 域分布: {', '.join(f'{d}={len(v)}' for d, v in TRAINING_DATA.items())}")

    # Step 2: 训练前基线生成
    print("\n[Step 2] 训练前基线生成...")
    test_prompts = [
        ("今天天气", "zh"),
        ("人工智能", "zh"),
        ("def hello", "code"),
        ("1+1=", "math"),
    ]
    baselines = {}
    for prompt, domain in test_prompts:
        try:
            gen = cortex.generate(prompt, max_tokens=30, domain=domain, temperature=0.8)
            diversity = compute_diversity(gen)
            rep_ratio = compute_repetition_ratio(gen)
            baselines[prompt] = (gen, diversity, rep_ratio)
            print(f"  [{domain}] '{prompt}' → '{gen[:50]}' (diversity={diversity:.2f}, rep={rep_ratio:.2f})")
        except Exception as e:
            baselines[prompt] = ("", 0.0, 1.0)
            print(f"  [{domain}] '{prompt}' → 生成失败: {e}")

    # Step 3: 多轮 feed+sleep 训练
    print("\n[Step 3] 多轮 feed+sleep 训练...")
    from taiji.life.feed_engine import get_feed_engine
    from taiji.life.sleep_engine import get_sleep_engine, SleepReport

    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    NUM_CYCLES = 8
    losses_by_cycle = []
    domain_losses_history = {d: [] for d in TRAINING_DATA.keys()}

    for cycle in range(NUM_CYCLES):
        # 每轮喂入全部数据
        for domain, texts in TRAINING_DATA.items():
            for text in texts:
                feed_engine.feed_text(text=text, source=f"cycle_{cycle}", domain=domain)

        # 触发 sleep 训练
        report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
        sleep_engine._sleep_phase_model_training(report)

        loss = report.training_loss
        losses_by_cycle.append(loss)
        n_samples = report.training_samples_used
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"

        # 打印调质状态（自主进化监控）
        nm = sleep_engine._neuromodulator
        if nm is not None:
            lr_mult = nm.get_lr_multiplier()
            print(f"  Cycle {cycle+1}/{NUM_CYCLES}: loss={loss_str}, samples={n_samples} | "
                  f"DA={nm.dopamine:.2f} 5HT={nm.serotonin:.2f} lr_mult={lr_mult:.2f}")
        else:
            print(f"  Cycle {cycle+1}/{NUM_CYCLES}: loss={loss_str}, samples={n_samples}")

    # Step 4: Loss 趋势分析
    print("\n[Step 4] Loss 趋势分析...")
    valid_losses = [l for l in losses_by_cycle if l is not None]
    if len(valid_losses) >= 2:
        first_loss = valid_losses[0]
        last_loss = valid_losses[-1]
        delta = first_loss - last_loss
        pct = (delta / first_loss * 100) if first_loss > 0 else 0
        print(f"  首轮 loss: {first_loss:.4f}")
        print(f"  末轮 loss: {last_loss:.4f}")
        print(f"  下降量: {delta:.4f} ({pct:+.1f}%)")
        if delta > 0:
            print("  ✅ Loss 持续下降")
        else:
            print("  ⚠️ Loss 未下降")

    # Step 5: 训练后生成对比
    print("\n[Step 5] 训练后生成对比...")
    improvements = 0
    total_compared = 0
    for prompt, domain in test_prompts:
        try:
            gen = cortex.generate(prompt, max_tokens=30, domain=domain, temperature=0.8)
            diversity = compute_diversity(gen)
            rep_ratio = compute_repetition_ratio(gen)
            base_gen, base_div, base_rep = baselines[prompt]

            div_improved = diversity > base_div
            rep_improved = rep_ratio < base_rep
            if div_improved or rep_improved:
                improvements += 1
            total_compared += 1

            div_arrow = "↑" if div_improved else "↓" if diversity < base_div else "="
            rep_arrow = "↓" if rep_improved else "↑" if rep_ratio > base_rep else "="
            print(f"  [{domain}] '{prompt}'")
            print(f"    前: '{base_gen[:40]}' (div={base_div:.2f}, rep={base_rep:.2f})")
            print(f"    后: '{gen[:40]}' (div={diversity:.2f}{div_arrow}, rep={rep_ratio:.2f}{rep_arrow})")
        except Exception as e:
            print(f"  [{domain}] '{prompt}' → 生成失败: {e}")

    # Step 6: Next-token 预测准确率
    print("\n[Step 6] Next-token 预测准确率...")
    coverage_results = {}
    for domain in ["zh"]:
        correct = 0
        total = 0
        top5_hits = 0
        domain_sp = cortex._tokenizer_hub.get_tokenizer(domain)

        for text in TRAINING_DATA[domain][:10]:  # 检查前 10 条
            domain_ids = cortex._tokenizer_hub.encode(text, domain=domain)
            if len(domain_ids) < 4:
                continue

            # 用逐 token 映射构造输入（与训练路径一致）
            general_ids = []
            for did in domain_ids:
                piece = domain_sp.id_to_piece(did)
                gen_ids = cortex._general_sp.EncodeAsIds(piece)
                if gen_ids:
                    general_ids.append(gen_ids[0])

            # 对每个位置，用前缀预测下一个 token
            for i in range(1, min(len(general_ids) - 1, 10)):
                prefix = general_ids[:i+1]
                if len(prefix) < 2:
                    continue
                ids_tensor = torch.tensor([prefix], dtype=torch.long)
                shared_emb = cortex._shared_embedding(ids_tensor)

                with torch.no_grad():
                    result = cortex.think(shared_emb)
                logits = result.get("neuron_logits", {}).get(domain)
                if logits is None:
                    continue

                last_logits = logits[0, -1, :]
                pred_token = torch.argmax(last_logits).item()
                true_token = domain_ids[i+1] if i+1 < len(domain_ids) else domain_ids[-1]

                total += 1
                if pred_token == true_token:
                    correct += 1
                top5 = torch.topk(last_logits, 5).indices.tolist()
                if true_token in top5:
                    top5_hits += 1

        accuracy = correct / total if total > 0 else 0
        top5_acc = top5_hits / total if total > 0 else 0
        coverage_results[domain] = accuracy
        print(f"  [{domain}] next-token 准确率: {accuracy:.1%} ({correct}/{total})")
        print(f"  [{domain}] top-5 准确率: {top5_acc:.1%} ({top5_hits}/{total})")

    # Step 7: 综合判断
    print("\n" + "=" * 60)
    loss_success = len(valid_losses) >= 2 and (valid_losses[0] - valid_losses[-1]) > 0
    quality_success = improvements >= total_compared * 0.5 if total_compared > 0 else False
    coverage_success = any(c > 0.05 for c in coverage_results.values()) if coverage_results else False

    if loss_success and (quality_success or coverage_success):
        print("🎉 验证通过：规模化训练数据改善生成质量")
        print(f"   - Loss: {valid_losses[0]:.4f} → {valid_losses[-1]:.4f}")
        print(f"   - 生成质量改善: {improvements}/{total_compared}")
        if coverage_results:
            print(f"   - Token 覆盖率: {', '.join(f'{d}={c:.1%}' for d, c in coverage_results.items())}")
        return 0
    else:
        print("⚠️ 验证未完全通过")
        print(f"   - Loss 下降: {loss_success}")
        print(f"   - 质量改善: {improvements}/{total_compared}")
        print(f"   - Token 覆盖: {coverage_success}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
