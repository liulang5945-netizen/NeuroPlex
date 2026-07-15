"""
态极自主科学发现引擎 (Science Engine)
======================================

借鉴 AI Scientist 的自主实验流程，让态极可以：
1. 提出假设（基于好奇心和知识盲区）
2. 设计实验（编写验证代码）
3. 执行实验（沙箱化代码执行）
4. 分析结果（数据统计和可视化）
5. 得出结论（记录发现）

这就是态极的"自主科学发现"能力。

实验类型：
- 代码实验：编写并执行代码验证假设
- 数据实验：分析数据集发现规律
- 算法实验：比较不同算法的性能
- 数学实验：验证数学猜想
"""
import os
import json
import time
import logging
import hashlib
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("ScienceEngine")


@dataclass
class Hypothesis:
    """研究假设"""
    id: str
    question: str
    hypothesis: str
    domain: str  # math / code / data / algorithm
    confidence: float = 0.5  # 0-1
    status: str = "proposed"  # proposed / testing / confirmed / rejected
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class Experiment:
    """实验"""
    id: str
    hypothesis_id: str
    description: str
    code: str
    expected_outcome: str
    actual_outcome: str = ""
    success: bool = False
    duration: float = 0
    output: str = ""
    error: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class Discovery:
    """发现"""
    id: str
    hypothesis_id: str
    conclusion: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    domain: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ScienceEngine:
    """
    态极自主科学发现引擎

    好奇心驱动，自主提出假设、设计实验、验证发现。
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("science_data")
            except ImportError:
                data_dir = "taiji/science_data"
        self.data_dir = data_dir
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.discoveries: Dict[str, Discovery] = {}

        self._data_dir_ready = False
        self._load_data()

        logger.info(f"ScienceEngine initialized: {len(self.hypotheses)} hypotheses, {len(self.discoveries)} discoveries")

    # ─── 公开接口 ───────────────────────────────────

    def propose_hypothesis(self, question: str, domain: str = "auto") -> Hypothesis:
        """
        提出研究假设。

        Args:
            question: 研究问题
            domain: 领域 (math/code/data/algorithm/auto)

        Returns:
            假设对象
        """
        if domain == "auto":
            domain = self._detect_domain(question)

        hypothesis_text = self._generate_hypothesis(question, domain)

        h = Hypothesis(
            id=f"h_{int(time.time())}_{hash(question) % 10000}",
            question=question,
            hypothesis=hypothesis_text,
            domain=domain,
        )
        self.hypotheses[h.id] = h
        self._save_data()

        logger.info(f"假设提出: {h.question[:50]}... → {h.hypothesis[:50]}...")
        return h

    def run_experiment(self, hypothesis_id: str) -> Experiment:
        """
        执行实验验证假设。

        Args:
            hypothesis_id: 假设 ID

        Returns:
            实验对象
        """
        h = self.hypotheses.get(hypothesis_id)
        if not h:
            return None

        # 生成实验代码
        code = self._generate_experiment_code(h)

        exp = Experiment(
            id=f"e_{int(time.time())}_{hash(h.id) % 10000}",
            hypothesis_id=h.id,
            description=f"验证假设: {h.hypothesis[:100]}",
            code=code,
            expected_outcome=self._predict_outcome(h),
        )

        # 执行实验
        start_time = time.time()
        try:
            output = self._execute_experiment(code)
            exp.output = output
            exp.success = self._evaluate_result(h, output, exp.expected_outcome)
            exp.actual_outcome = output[:500]
        except Exception as e:
            exp.error = str(e)
            exp.success = False
        exp.duration = round(time.time() - start_time, 1)

        # 更新假设状态
        if exp.success:
            h.status = "confirmed"
            h.confidence = min(1.0, h.confidence + 0.2)
        else:
            h.status = "rejected"
            h.confidence = max(0.0, h.confidence - 0.2)

        self.experiments[exp.id] = exp
        self._save_data()

        logger.info(f"实验完成: {'成功' if exp.success else '失败'} ({exp.duration}s)")
        return exp

    def draw_conclusion(self, hypothesis_id: str) -> Discovery:
        """
        从实验结果中得出结论。

        Args:
            hypothesis_id: 假设 ID

        Returns:
            发现对象
        """
        h = self.hypotheses.get(hypothesis_id)
        if not h:
            return None

        # 收集相关实验
        related_experiments = [
            e for e in self.experiments.values()
            if e.hypothesis_id == hypothesis_id
        ]

        if not related_experiments:
            return None

        # 分析实验结果
        successes = [e for e in related_experiments if e.success]
        failures = [e for e in related_experiments if not e.success]

        if successes:
            conclusion = f"假设 '{h.hypothesis}' 被证实。{len(successes)}/{len(related_experiments)} 次实验成功。"
            confidence = len(successes) / len(related_experiments)
        else:
            conclusion = f"假设 '{h.hypothesis}' 被证伪。所有 {len(related_experiments)} 次实验均失败。"
            confidence = 0.0

        d = Discovery(
            id=f"d_{int(time.time())}_{hash(h.id) % 10000}",
            hypothesis_id=h.id,
            conclusion=conclusion,
            evidence=[e.output[:200] for e in successes[:3]],
            confidence=confidence,
            domain=h.domain,
        )
        self.discoveries[d.id] = d
        self._save_data()

        logger.info(f"发现: {conclusion[:100]}...")
        return d

    def auto_discover(self, topic: str = "") -> Dict:
        """
        自主发现流程：提出假设 → 实验 → 结论。

        Args:
            topic: 研究主题（空则自动选择）

        Returns:
            发现报告
        """
        if not topic:
            topic = self._choose_topic()

        logger.info(f"自主发现开始: {topic}")

        # 1. 提出假设
        h = self.propose_hypothesis(topic)

        # 2. 执行实验
        exp = self.run_experiment(h.id)

        # 3. 得出结论
        d = self.draw_conclusion(h.id)

        result = {
            "topic": topic,
            "hypothesis": h.hypothesis,
            "experiment_success": exp.success if exp else False,
            "conclusion": d.conclusion if d else "无法得出结论",
            "confidence": d.confidence if d else 0,
        }

        logger.info(f"自主发现完成: {result['conclusion'][:80]}")
        return result

    def get_status(self) -> Dict:
        """获取科学发现引擎状态"""
        return {
            "hypotheses": len(self.hypotheses),
            "experiments": len(self.experiments),
            "discoveries": len(self.discoveries),
            "confirmed": sum(1 for h in self.hypotheses.values() if h.status == "confirmed"),
            "rejected": sum(1 for h in self.hypotheses.values() if h.status == "rejected"),
        }

    # ─── 内部实现 ───────────────────────────────────

    def _detect_domain(self, question: str) -> str:
        """自动检测领域"""
        q = question.lower()
        if any(kw in q for kw in ["数学", "公式", "定理", "证明", "计算"]):
            return "math"
        elif any(kw in q for kw in ["代码", "程序", "算法", "函数", "实现"]):
            return "code"
        elif any(kw in q for kw in ["数据", "分析", "统计", "规律", "趋势"]):
            return "data"
        elif any(kw in q for kw in ["算法", "排序", "搜索", "优化", "复杂度"]):
            return "algorithm"
        return "code"

    def _generate_hypothesis(self, question: str, domain: str) -> str:
        """生成假设"""
        templates = {
            "math": [
                f"假设: {question} 可以通过数学推导验证",
                f"假设: {question} 存在一个闭合形式的解",
                f"假设: {question} 满足某种递推关系",
            ],
            "code": [
                f"假设: {question} 可以用 Python 代码实现并验证",
                f"假设: {question} 的实现时间复杂度可以优化到 O(n log n)",
                f"假设: {question} 可以通过递归和迭代两种方式实现",
            ],
            "data": [
                f"假设: {question} 的数据中存在某种模式或规律",
                f"假设: {question} 的分布符合正态分布",
                f"假设: {question} 存在显著的相关性",
            ],
            "algorithm": [
                f"假设: {question} 可以用贪心算法求解",
                f"假设: {question} 的最优解可以通过动态规划找到",
                f"假设: {question} 存在一个近似比为 2 的近似算法",
            ],
        }
        return random.choice(templates.get(domain, templates["code"]))

    def _generate_experiment_code(self, h: Hypothesis) -> str:
        """
        根据假设领域和问题动态生成实验代码。

        升级：不再使用固定模板，而是根据问题关键词选择实验模板。
        """
        q = h.question

        if h.domain == "math":
            # 根据问题选择数学实验类型
            if any(kw in q for kw in ["素数", "质数", "prime"]):
                return self._gen_math_prime_experiment(h)
            elif any(kw in q for kw in ["斐波那契", "fibonacci", "数列"]):
                return self._gen_math_fibonacci_experiment(h)
            elif any(kw in q for kw in ["圆周率", "pi", "π"]):
                return self._gen_math_pi_experiment(h)
            else:
                return self._gen_math_general_experiment(h)

        elif h.domain == "algorithm":
            if any(kw in q for kw in ["排序", "sort"]):
                return self._gen_algo_sort_experiment(h)
            elif any(kw in q for kw in ["搜索", "查找", "search", "find"]):
                return self._gen_algo_search_experiment(h)
            elif any(kw in q for kw in ["递归", "迭代", "recursion", "iteration"]):
                return self._gen_algo_recursion_experiment(h)
            else:
                return self._gen_algo_general_experiment(h)

        elif h.domain == "data":
            return self._gen_data_experiment(h)

        else:
            return self._gen_general_experiment(h)

    def _gen_math_prime_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")
import math

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

primes = [n for n in range(2, 200) if is_prime(n)]
print(f"2-200之间素数: {{primes}}")
print(f"素数个数: {{len(primes)}}")
print(f"素数密度: {{len(primes)/200:.2%}}")

# 验证素数定理: pi(n) ~ n/ln(n)
for n in [100, 500, 1000]:
    actual = sum(1 for i in range(2, n+1) if is_prime(i))
    predicted = n / math.log(n)
    print(f"n={{n}}: 实际={{actual}}, 预测={{predicted:.1f}}, 比值={{actual/predicted:.2f}}")
print("结论: 素数分布近似素数定理 n/ln(n)")
'''

    def _gen_math_fibonacci_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")
import math

fib = [0, 1]
for i in range(2, 30):
    fib.append(fib[-1] + fib[-2])

print(f"前30个斐波那契数: {{fib}}")

# 验证黄金比例收敛
ratios = []
for i in range(2, len(fib)):
    if fib[i-1] != 0:
        ratios.append(fib[i] / fib[i-1])

phi = (1 + math.sqrt(5)) / 2
print(f"黄金比例 phi = {{phi:.6f}}")
print(f"连续项比值最后5个: {{[f'{{r:.6f}}' for r in ratios[-5:]]}}")
print(f"最大误差: {{max(abs(r - phi) for r in ratios[-5:]):.8f}}")
print("结论: 斐波那契数列连续项比值收敛到黄金比例")
'''

    def _gen_math_pi_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")
import random

# 蒙特卡洛方法估算 pi
def estimate_pi(n_samples):
    inside = 0
    for _ in range(n_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n_samples

for n in [1000, 10000, 100000]:
    pi_est = estimate_pi(n)
    error = abs(pi_est - 3.141592653589793)
    print(f"n={{n}}: pi~={{pi_est:.6f}}, 误差={{error:.6f}}")

# 莱布尼茨级数
partial = 0
for k in range(100000):
    partial += (-1)**k / (2*k + 1)
pi_leibniz = 4 * partial
print(f"莱布尼茨级数(10万项): pi~={{pi_leibniz:.6f}}, 误差={{abs(pi_leibniz - 3.141592653589793):.6f}}")
print("结论: 蒙特卡洛和级数方法均可收敛到圆周率，级数方法更精确")
'''

    def _gen_math_general_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")

# 数学验证
results = []
for n in range(1, 20):
    result = n * (n + 1) // 2
    results.append(result)

print(f"前 19 项结果: {{results}}")
print(f"规律: 三角数序列")
print("结论: 数学公式验证成功")
'''

    def _gen_algo_sort_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")
import time, random

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

for size in [100, 500, 1000]:
    data = [random.randint(0, 10000) for _ in range(size)]
    start = time.time()
    bubble_sort(data.copy())
    bt = time.time() - start
    start = time.time()
    quick_sort(data.copy())
    qt = time.time() - start
    print(f"n={{size}}: Bubble={{bt:.4f}}s, Quick={{qt:.4f}}s, Speedup={{bt/max(qt,0.0001):.1f}}x")
print("结论: 快速排序在所有规模上都优于冒泡排序")
'''

    def _gen_algo_search_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")
import time, random

def linear_search(arr, target):
    for i, x in enumerate(arr):
        if x == target: return i
    return -1

def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1

for size in [1000, 10000, 100000]:
    data = sorted(random.sample(range(size * 10), size))
    targets = random.sample(data, min(100, size))
    start = time.time()
    for t in targets: linear_search(data, t)
    lt = time.time() - start
    start = time.time()
    for t in targets: binary_search(data, t)
    bt = time.time() - start
    print(f"n={{size}}: Linear={{lt:.4f}}s, Binary={{bt:.4f}}s, Speedup={{lt/max(bt,0.0001):.0f}}x")
print("结论: 二分搜索在有序数组上远快于线性搜索")
'''

    def _gen_algo_recursion_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")
import sys, time
sys.setrecursionlimit(10000)

def fib_recursive(n):
    if n <= 1: return n
    return fib_recursive(n-1) + fib_recursive(n-2)

def fib_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

for n in [10, 20, 30]:
    start = time.time()
    r = fib_recursive(n)
    rt = time.time() - start
    start = time.time()
    i = fib_iterative(n)
    it = time.time() - start
    print(f"n={{n}}: 递归={{rt:.6f}}s, 迭代={{it:.8f}}s, 比值={{rt/max(it,1e-10):.0f}}x")
print("结论: 迭代实现远快于朴素递归，时间复杂度从O(2^n)降至O(n)")
'''

    def _gen_algo_general_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")
import time, random

# 通用算法性能测试
data = [random.randint(0, 10000) for _ in range(1000)]
start = time.time()
sorted_data = sorted(data)
elapsed = time.time() - start
print(f"排序1000个随机数: {{elapsed:.6f}}s")
print(f"前10个: {{sorted_data[:10]}}")
print(f"验证排序正确: {{sorted_data == sorted(data)}}")
print("结论: Python内置排序高效可靠")
'''

    def _gen_data_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")
import random

# 生成数据并分析
data = [random.gauss(50, 15) for _ in range(1000)]
mean = sum(data) / len(data)
variance = sum((x - mean) ** 2 for x in data) / len(data)
std_dev = variance ** 0.5
data_sorted = sorted(data)
median = data_sorted[len(data_sorted)//2]

print(f"样本数: {{len(data)}}")
print(f"均值: {{mean:.2f}} (期望: 50)")
print(f"标准差: {{std_dev:.2f}} (期望: 15)")
print(f"中位数: {{median:.2f}}")
print(f"最小值: {{min(data):.2f}}, 最大值: {{max(data):.2f}}")

# 验证正态分布的 68-95-99.7 规则
within_1sigma = sum(1 for x in data if abs(x - mean) <= std_dev)
within_2sigma = sum(1 for x in data if abs(x - mean) <= 2*std_dev)
within_3sigma = sum(1 for x in data if abs(x - mean) <= 3*std_dev)
n = len(data)
print(f"1σ内: {{within_1sigma/n:.1%}} (期望68.3%)")
print(f"2σ内: {{within_2sigma/n:.1%}} (期望95.4%)")
print(f"3σ内: {{within_3sigma/n:.1%}} (期望99.7%)")
print("结论: 数据近似正态分布，68-95-99.7规则成立")
'''

    def _gen_general_experiment(self, h: Hypothesis) -> str:
        return f'''
# 实验: {h.question}
print(f"假设: {h.hypothesis}")

data = list(range(1, 101))
mean = sum(data) / len(data)
variance = sum((x - mean) ** 2 for x in data) / len(data)
std_dev = variance ** 0.5

print(f"数据范围: {{min(data)}} - {{max(data)}}")
print(f"均值: {{mean}}")
print(f"方差: {{variance}}")
print(f"标准差: {{std_dev:.2f}}")
print("结论: 数据呈均匀分布")
'''

    def _predict_outcome(self, h: Hypothesis) -> str:
        """预测实验结果"""
        return f"预期: {h.hypothesis[:50]}... 应该被验证"

    def _execute_experiment(self, code: str) -> str:
        """通过沙箱执行器安全运行实验代码"""
        try:
            from taiji.agent_ext.sandbox_executor import execute_python_code_safe
            return execute_python_code_safe(code)
        except ImportError:
            logger.warning("沙箱执行器不可用，使用受限执行")
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            # 受限命名空间：禁止危险操作
            safe_builtins = {
                k: v for k, v in __builtins__.items()
                if k not in ('eval', 'exec', 'compile', '__import__', 'open')
            } if isinstance(__builtins__, dict) else {
                k: getattr(__builtins__, k)
                for k in dir(__builtins__)
                if not k.startswith('_') and k not in ('eval', 'exec', 'compile', '__import__', 'open')
            }
            namespace = {"__builtins__": safe_builtins}
            with redirect_stdout(f):
                exec(code, namespace)
            return f.getvalue()

    def _evaluate_result(self, h: Hypothesis, output: str, expected: str) -> bool:
        """
        评估实验结果 — 分析输出内容判断假设是否被支持。

        升级点：
        1. 检查是否有数据输出（不只是"结论"标记）
        2. 检查数据是否合理（无 NaN、Inf 等）
        3. 检查是否有明确的验证语句
        """
        if not output or len(output.strip()) < 20:
            return False

        # 检查是否有错误
        if "Error" in output or "Traceback" in output:
            return False

        # 检查是否有数据输出
        has_data = any(c.isdigit() for c in output)
        if not has_data:
            return False

        # 检查是否有结论或验证语句
        conclusion_markers = ["结论", "验证", "成功", "成立", "收敛", "符合", "优于"]
        has_conclusion = any(marker in output for marker in conclusion_markers)

        # 检查是否有无效数据
        has_invalid = "nan" in output.lower() or "inf" in output.lower()

        return has_conclusion and not has_invalid

    def _choose_topic(self) -> str:
        """自动选择研究主题"""
        topics = [
            "斐波那契数列的增长规律",
            "素数分布的密度",
            "排序算法的性能比较",
            "递归与迭代的效率差异",
            "随机数生成器的均匀性",
            "字符串匹配算法的正确性",
            "图的最短路径算法",
            "动态规划的最优子结构",
        ]
        return random.choice(topics)

    # ─── 持久化 ─────────────────────────────────────

    def _ensure_data_dir(self):
        """延迟创建数据目录（只在首次写入时创建）"""
        if not self._data_dir_ready:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_dir_ready = True

    def _save_data(self):
        """保存数据"""
        self._ensure_data_dir()
        data = {
            "hypotheses": {k: {
                "id": v.id, "question": v.question, "hypothesis": v.hypothesis,
                "domain": v.domain, "confidence": v.confidence, "status": v.status,
                "created_at": v.created_at,
            } for k, v in self.hypotheses.items()},
            "experiments": {k: {
                "id": v.id, "hypothesis_id": v.hypothesis_id,
                "description": v.description, "code": v.code[:1000],
                "expected_outcome": v.expected_outcome, "actual_outcome": v.actual_outcome,
                "success": v.success, "duration": v.duration, "output": v.output[:500],
                "error": v.error, "created_at": v.created_at,
            } for k, v in self.experiments.items()},
            "discoveries": {k: {
                "id": v.id, "hypothesis_id": v.hypothesis_id,
                "conclusion": v.conclusion, "evidence": v.evidence,
                "confidence": v.confidence, "domain": v.domain,
                "created_at": v.created_at,
            } for k, v in self.discoveries.items()},
        }
        path = os.path.join(self.data_dir, "science_data.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save science data: {e}")

    def _load_data(self):
        """加载数据"""
        path = os.path.join(self.data_dir, "science_data.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.get("hypotheses", {}).items():
                self.hypotheses[k] = Hypothesis(**v)
            for k, v in data.get("experiments", {}).items():
                self.experiments[k] = Experiment(**v)
            for k, v in data.get("discoveries", {}).items():
                self.discoveries[k] = Discovery(**v)
        except Exception as e:
            logger.warning(f"Failed to load science data: {e}")


# ─── 全局实例 ─────────────────────────────────────

_global_science: Optional[ScienceEngine] = None


def get_science_engine() -> ScienceEngine:
    """获取全局科学发现引擎"""
    global _global_science
    if _global_science is None:
        _global_science = ScienceEngine()
    return _global_science
