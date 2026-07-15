"""Phase 2.5-2.6: Prepare distillation data and extract teacher directions.

Creates domain-specific synthetic datasets for initial distillation testing,
then extracts teacher hidden-state directions from the 1.5B checkpoint.

Domains: zh (Chinese), en (English), code (programming), math, general

Usage:
    python scripts/training/prepare_distill_data.py \
        --checkpoint e:/taiji/checkpoint-400000 \
        --output_dir data/distill \
        --samples_per_domain 500 \
        --seq_len 256
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states
from taiji.resonance import COMPACT, STANDARD, EXPERT, TINY_TEST


# ── Domain-specific synthetic text templates ──

DOMAIN_TEXTS: Dict[str, List[str]] = {
    "zh": [
        "今天天气很好，我们一起去公园散步吧。春天的花开得特别美丽，空气中弥漫着花香。",
        "人工智能正在改变我们的生活方式。从智能手机到自动驾驶汽车，技术正在深入到每个角落。",
        "中国有着悠久的历史和丰富的文化遗产。从长城到故宫，每一处都讲述着动人的故事。",
        "学习编程需要耐心和毅力。每天坚持写代码，慢慢就会看到进步。重要的是不要放弃。",
        "健康饮食对我们的身体非常重要。多吃蔬菜水果，少吃油炸食品，保持良好的作息习惯。",
        "数学是一门非常有趣的学科。它不仅仅是计算，更是逻辑思维和问题解决能力的训练。",
        "阅读是获取知识的重要途径。通过阅读，我们可以了解不同的文化和思想，拓宽视野。",
        "环境保护是每个人的责任。减少塑料使用，节约能源，从小事做起保护我们的地球。",
        "音乐能够治愈心灵。无论是古典音乐还是流行歌曲，都能给人带来不同的情感体验。",
        "团队合作是现代工作的重要方式。良好的沟通和协作能够提高工作效率和质量。",
        "教育是改变命运的关键。良好的教育不仅传授知识，更培养独立思考和创新能力。",
        "科技发展日新月异，5G、物联网、区块链等新技术不断涌现，深刻影响着社会发展。",
        "传统文化需要传承和创新。将传统与现代结合，让古老的文化在新时代焕发新的活力。",
        "旅行可以开阔视野，体验不同地方的风土人情，感受世界的多样性和美好。",
        "面对困难时，保持积极乐观的心态非常重要。每一次挑战都是成长的机会。",
    ],
    "en": [
        "The quick brown fox jumps over the lazy dog. This classic pangram contains every letter of the alphabet.",
        "Artificial intelligence and machine learning are transforming industries worldwide at an unprecedented pace.",
        "The history of computing spans from mechanical calculators to quantum computers in just a few centuries.",
        "Effective communication skills are essential in both personal relationships and professional environments.",
        "Climate change is one of the most pressing challenges facing humanity in the twenty-first century.",
        "The scientific method involves observation, hypothesis formation, experimentation, and drawing conclusions.",
        "Reading books expands our understanding of the world and exposes us to diverse perspectives and ideas.",
        "Regular physical exercise combined with a balanced diet contributes significantly to overall health and wellbeing.",
        "The Renaissance period marked a profound transformation in European art, science, and philosophical thought.",
        "Democratic societies rely on informed citizens participating in the political process through voting and civic engagement.",
        "Space exploration has led to numerous technological innovations that benefit everyday life on Earth.",
        "Understanding basic economics helps people make better financial decisions and navigate complex markets.",
        "The development of the internet has fundamentally changed how we communicate, work, and access information.",
        "Critical thinking skills enable us to evaluate arguments, identify biases, and make reasoned judgments.",
        "Music theory provides a framework for understanding how melodies, harmonies, and rhythms work together.",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n    def insert(self, value):\n        self.root = self._insert_recursive(self.root, value)",
        "async function fetchData(url) {\n    const response = await fetch(url);\n    const data = await response.json();\n    return data;\n}",
        "import numpy as np\ndef gradient_descent(X, y, theta, alpha, iterations):\n    m = len(y)\n    for _ in range(iterations):\n        h = X @ theta\n        theta -= (alpha/m) * X.T @ (h - y)\n    return theta",
        "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nGROUP BY u.id, u.name\nHAVING COUNT(o.id) > 5\nORDER BY order_count DESC;",
        "fn merge_sort<T: Ord + Clone>(arr: &mut [T]) {\n    if arr.len() <= 1 { return; }\n    let mid = arr.len() / 2;\n    merge_sort(&mut arr[..mid]);\n    merge_sort(&mut arr[mid..]);\n}",
        "docker build -t myapp:latest . && docker run -p 8080:8080 -e DATABASE_URL=postgres://localhost/mydb myapp:latest",
        "const useDebounce = (value, delay) => {\n    const [debouncedValue, setDebouncedValue] = useState(value);\n    useEffect(() => {\n        const handler = setTimeout(() => setDebouncedValue(value), delay);\n        return () => clearTimeout(handler);\n    }, [value, delay]);\n    return debouncedValue;\n};",
        "package main\nimport \"fmt\"\ntype Stack struct {\n    items []int\n}\nfunc (s *Stack) Push(item int) { s.items = append(s.items, item) }\nfunc (s *Stack) Pop() int {\n    item := s.items[len(s.items)-1]\n    s.items = s.items[:len(s.items)-1]\n    return item\n}",
        "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
        "const express = require('express');\nconst app = express();\napp.get('/api/users/:id', async (req, res) => {\n    try {\n        const user = await User.findById(req.params.id);\n        res.json(user);\n    } catch (err) {\n        res.status(500).json({ error: err.message });\n    }\n});",
        "public class LinkedList<T> {\n    private Node<T> head;\n    public void add(T data) {\n        Node<T> node = new Node<>(data);\n        if (head == null) { head = node; return; }\n        Node<T> current = head;\n        while (current.next != null) current = current.next;\n        current.next = node;\n    }\n}",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
        "# Kubernetes Deployment\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: nginx-deployment\nspec:\n  replicas: 3\n  selector:\n    matchLabels:\n      app: nginx\n  template:\n    metadata:\n      labels:\n        app: nginx\n    spec:\n      containers:\n      - name: nginx\n        image: nginx:1.19",
        "type Result<T> = Ok<T> | Err<Error>;\nfunction match<T, R>(result: Result<T>, handlers: { ok: (val: T) => R; err: (e: Error) => R }): R {\n    if (result.type === 'Ok') return handlers.ok(result.value);\n    return handlers.err(result.error);\n}",
    ],
    "math": [
        "The derivative of f(x) = x^3 is f'(x) = 3x^2. This follows from the power rule: d/dx(x^n) = nx^(n-1).",
        "In a right triangle with sides a and b and hypotenuse c, the Pythagorean theorem states that a^2 + b^2 = c^2.",
        "The integral of e^x from 0 to 1 equals e - 1, approximately 1.71828. This is a fundamental result in calculus.",
        "The probability of rolling a sum of 7 with two fair dice is 6/36 = 1/6, as there are 6 favorable outcomes out of 36 total.",
        "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.",
        "The quadratic formula x = (-b ± sqrt(b^2 - 4ac)) / (2a) gives the roots of ax^2 + bx + c = 0.",
        "The sum of the first n natural numbers is n(n+1)/2. For n=100, this equals 5050.",
        "In linear algebra, a matrix A is invertible if and only if its determinant det(A) is non-zero.",
        "The Fibonacci sequence is defined by F(0)=0, F(1)=1, and F(n)=F(n-1)+F(n-2) for n >= 2.",
        "The area of a circle with radius r is pi * r^2, and its circumference is 2 * pi * r.",
        "Euler's identity e^(i*pi) + 1 = 0 is considered one of the most beautiful equations in mathematics.",
        "The binomial theorem states that (x + y)^n = sum(k=0 to n) C(n,k) * x^(n-k) * y^k.",
        "A function f is continuous at point a if lim(x->a) f(x) = f(a).",
        "The Law of Large Numbers states that as sample size grows, the sample mean approaches the expected value.",
        "In set theory, the power set of a set S with n elements has 2^n elements.",
    ],
    "general": [
        "Technology has become an integral part of modern life, influencing everything from communication to transportation and healthcare delivery systems worldwide.",
        "环境保护和可持续发展是当今全球面临的重要课题，需要各国政府、企业和个人的共同努力。",
        "def solve_problem(input_data):\n    result = process_data(input_data)\n    return validate_and_format(result)",
        "The concept of infinity has fascinated mathematicians and philosophers for millennia, from Zeno's paradoxes to modern set theory.",
        "学习方法因人而异，找到适合自己的学习方式能够事半功倍。有些人喜欢视觉学习，有些人则偏好动手实践。",
        "RESTful API design principles include statelessness, resource-based URLs, and proper use of HTTP methods like GET, POST, PUT, and DELETE.",
        "The Earth's climate system is complex, involving interactions between the atmosphere, oceans, land surface, and cryosphere.",
        "设计模式是软件工程中常用的解决方案，如单例模式、工厂模式和观察者模式等，帮助开发者写出可维护的代码。",
        "A balanced diet should include proteins, carbohydrates, healthy fats, vitamins, and minerals from a variety of food sources.",
        "量子计算利用量子力学原理进行信息处理，在某些特定问题上可能远超经典计算机的计算能力。",
        "const pipeline = [filterValid, transformData, aggregateResults].reduce((acc, fn) => fn(acc), rawInput);",
        "The Renaissance was a period of great cultural and intellectual achievement in Europe, spanning roughly from the 14th to the 17th century.",
        "团队管理的关键在于有效的沟通和明确的目标设定。每个人都需要清楚自己的职责和团队的整体方向。",
        "Machine learning models require careful feature engineering, hyperparameter tuning, and validation to achieve optimal performance.",
        "The human brain contains approximately 86 billion neurons, forming an incredibly complex network that enables consciousness and cognition.",
    ],
}


def load_tokenizer(checkpoint_dir: str):
    """Load the SentencePiece tokenizer from the checkpoint directory."""
    import sentencepiece as spm
    sp_path = os.path.join(checkpoint_dir, "sentencepiece.model")
    if not os.path.exists(sp_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(os.path.dirname(checkpoint_dir), "sentencepiece.model"),
            "e:/taiji/sentencepiece.model",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                sp_path = p
                break

    if not os.path.exists(sp_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {sp_path}. "
            f"Checked: {checkpoint_dir}/sentencepiece.model"
        )

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    print(f"  Loaded tokenizer: vocab_size={sp.GetPieceSize()}")
    return sp


def create_domain_dataset(
    tokenizer,
    domain: str,
    texts: List[str],
    num_samples: int,
    seq_len: int,
) -> torch.Tensor:
    """Create tokenized dataset for a domain by repeating and shuffling texts.

    Returns token_ids tensor of shape [num_samples, seq_len].
    """
    all_tokens = []
    for text in texts:
        encoded = tokenizer.EncodeAsIds(text)
        if len(encoded) >= seq_len:
            # Truncate to seq_len
            all_tokens.append(encoded[:seq_len])
        else:
            # Pad with repeats of same text
            padded = encoded.copy()
            while len(padded) < seq_len:
                padded = padded + encoded
            all_tokens.append(padded[:seq_len])

    # Repeat to reach num_samples
    while len(all_tokens) < num_samples:
        all_tokens = all_tokens + all_tokens
    all_tokens = all_tokens[:num_samples]

    tensor = torch.tensor(all_tokens, dtype=torch.long)
    return tensor


def extract_teacher_direction(
    teacher_model,
    domain_dataset: torch.Tensor,
    device: str = "cpu",
    max_batches: int = 50,
) -> torch.Tensor:
    """Extract average teacher hidden-state direction for a domain.

    Returns [hidden_dim] normalized direction vector.
    """
    teacher_model.eval()
    all_hidden = []

    with torch.no_grad():
        for i in range(0, min(len(domain_dataset), max_batches)):
            input_ids = domain_dataset[i:i+1].to(device)  # [1, L]
            hidden = extract_hidden_states(teacher_model, input_ids)
            all_hidden.append(hidden.mean(dim=1))  # [1, hidden_dim]

    if not all_hidden:
        return torch.zeros(1)

    direction = torch.cat(all_hidden, dim=0).mean(dim=0)  # [hidden_dim]
    return direction / (direction.norm() + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Prepare distillation data and extract teacher directions")
    parser.add_argument("--checkpoint", default="e:/taiji/checkpoint-400000",
                        help="Path to first-gen checkpoint directory")
    parser.add_argument("--output_dir", default="data/distill",
                        help="Output directory for prepared data")
    parser.add_argument("--samples_per_domain", type=int, default=500,
                        help="Number of samples per domain")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--device", default="cpu",
                        help="Device for computation")
    parser.add_argument("--skip_teacher", action="store_true",
                        help="Skip teacher direction extraction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 2.5: Prepare Distillation Data")
    print("=" * 60)

    # ── Load tokenizer ──
    sp = load_tokenizer(args.checkpoint)

    # ── Create domain datasets ──
    datasets = {}
    for domain in ["zh", "en", "code", "math", "general"]:
        texts = DOMAIN_TEXTS[domain]
        data = create_domain_dataset(sp, domain, texts, args.samples_per_domain, args.seq_len)
        datasets[domain] = data
        print(f"  {domain}: {data.shape} tokens, range=[{data.min().item()}, {data.max().item()}]")

    # ── Save datasets ──
    data_path = os.path.join(args.output_dir, "domain_datasets.pt")
    torch.save(datasets, data_path)
    print(f"\n  Saved datasets to {data_path}")

    # ── Print dataloader info ──
    print("\n  Dataloader usage:")
    for domain, data in datasets.items():
        dl = DataLoader(TensorDataset(data), batch_size=4, shuffle=True)
        batch = next(iter(dl))
        print(f"    {domain}: batch shape={batch[0].shape}, loader size={len(dl)}")

    # ── Phase 2.6: Extract teacher directions ──
    if not args.skip_teacher:
        print("\n" + "=" * 60)
        print("Phase 2.6: Extract Teacher Directions")
        print("=" * 60)

        teacher, embedding = load_teacher_model(args.checkpoint, device=args.device)
        print(f"  Teacher model loaded: {sum(p.numel() for p in teacher.parameters())/1e9:.2f}B params")

        teacher_dirs = {}
        for domain, data in datasets.items():
            direction = extract_teacher_direction(teacher, data, device=args.device)
            teacher_dirs[domain] = direction
            print(f"  {domain}: direction norm={direction.norm().item():.4f}, dim={direction.shape}")

        # Save teacher directions
        dir_path = os.path.join(args.output_dir, "teacher_directions.pt")
        torch.save(teacher_dirs, dir_path)
        print(f"\n  Saved teacher directions to {dir_path}")

    print("\n" + "=" * 60)
    print("DONE: Data preparation complete")
    print("=" * 60)
    print(f"\nNext: Phase 2.7 — Run distillation training for each neuron.")
    print(f"  python scripts/training/distill_neurons.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
