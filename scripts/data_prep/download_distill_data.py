"""Download real text data via raw URLs for neuron distillation.

Avoids datasets library (Python 3.14 incompatibility).
Downloads raw text from simple sources, tokenizes 256K general tokenizer.

Usage:
    python scripts/data_prep/download_distill_data.py --output_dir data/real --samples 2000
"""
from __future__ import annotations
import argparse, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import requests

# Raw text URLs - simple, no auth needed, small enough for quick download
TEXT_SOURCES = {
    "zh": {
        "urls": [
            "https://raw.githubusercontent.com/nicklauszhu/chinese-text-dataset/master/Chinese_news.csv",
        ],
        "desc": "Chinese news texts",
        "parser": "csv",
    },
    "en": {
        "urls": [
            "https://raw.githubusercontent.com/niderhoff/nlp-datasets/master/wikitext/wikitext-2/train.txt",
        ],
        "desc": "WikiText-2 English",
        "parser": "lines",
    },
    "code": {
        "urls": [],
        "desc": "Code samples (synthetic)",
        "parser": "synthetic",
    },
    "math": {
        "urls": [],
        "desc": "Math problems (synthetic)",
        "parser": "synthetic",
    },
    "general": {
        "urls": [],
        "desc": "General mixed (synthetic)",
        "parser": "synthetic",
    },
}

# Fallback synthetic data with MUCH more variety than 15 templates
def generate_synthetic_code(n=2000):
    """Generate diverse code snippets."""
    import random
    funcs = ["process", "compute", "validate", "transform", "parse", "encode", "load", "save", "init", "filter"]
    args = ["x", "y", "data", "items", "config", "query", "name", "idx"]
    bodies = [
        "result = x + y", "data.sort()", "if not items: return None",
        "config.update(params)", "with open(path) as f: return f.read()",
        "return sum(items) / max(len(items), 1)", "self.cache[id] = value",
        "for item in items: process(item)", "logger.info(f'Processing {name}')",
    ]
    texts = []
    for _ in range(n):
        a = ", ".join(random.sample(args, random.randint(1,3)))
        b = random.choice(bodies)
        f = random.choice(funcs)
        n = random.choice(["MyClass", "Processor", "Handler", "Manager", "Controller"])
        # Use template strings without .format() conflicts
        patterns = [
            f"def {f}({a}):\n    {b}\n    return result",
            f"class {n}:\n    def __init__(self, {a}):\n        self.data = {{}}\n    def {f}(self, {a}):\n        {b}",
            f"for item in items:\n    {b}",
            f"if len(items) > 0:\n    {b}\nelse:\n    return None",
            f"try:\n    {b}\nexcept Exception as e:\n    print(e)",
            f"with open('data.txt', 'r') as f:\n    content = f.read()\n    {b}",
            f"result = [{f}(x) for x in items if x > 0]",
            f"sorted_items = sorted(items, key=lambda x: x.id, reverse=True)",
            f"import json\nimport os\n\n{b}",
            f"const handle{random.choice(['Click','Submit','Load','Change'])} = ({a}) => {{\n    {b}\n}}",
            f"SELECT * FROM {random.choice(['users','orders','products'])} WHERE id = {random.randint(1,100)} ORDER BY created_at DESC LIMIT 10;",
            f"INSERT INTO {random.choice(['users','orders'])} (name, email) VALUES ('test', 'user@example.com');",
            f"docker run -d -p 8080:80 {random.choice(['nginx','redis','python'])}:latest",
            f"git clone https://github.com/user/repo.git && cd repo && npm install",
            f"public class {n} {{\n    public static void main(String[] args) {{\n        System.out.println(\"Hello\");\n    }}\n}}",
        ]
        texts.append(random.choice(patterns)[:500])
    return texts


def generate_synthetic_math(n=2000):
    """Generate diverse math problems using f-strings."""
    import random
    rd = random
    texts = []
    for _ in range(n):
        a, b, c = rd.randint(1,20), rd.randint(1,20), rd.randint(1,50)
        d, e = rd.randint(1,10), rd.randint(1,10)
        n_val = rd.randint(1,100)
        patterns = [
            f"Solve for x: {a}x + {b} = {c}.",
            f"Find the derivative of f(x) = {a}x^{rd.randint(2,5)} + {b}x^{rd.randint(1,3)} + {c}.",
            f"Compute the integral of {a}x^{rd.randint(1,4)} dx from 0 to {b}.",
            f"What is the probability of rolling a sum of {rd.randint(2,12)} with two fair dice?",
            f"Find the area of a circle with radius {rd.randint(1,20)}.",
            f"If f(x) = {a}x^2 + {b}x + {c}, find f({d}).",
            f"Solve the system: {rd.randint(1,10)}x + {rd.randint(1,10)}y = {rd.randint(1,50)}, {rd.randint(1,10)}x + {rd.randint(1,10)}y = {rd.randint(1,50)}.",
            f"What is the {n_val}th term of the arithmetic sequence starting at {a} with difference {d}?",
            f"Compute the determinant of matrix [[{a},{b}],[{c},{d}]].",
            f"What is the value of sin({rd.choice([30,45,60,90])} degrees)?",
            f"Solve: log base {rd.randint(2,10)} of x = {rd.randint(1,5)}. Find x.",
            f"A triangle has sides {a}, {b}, {c}. Find its area using Heron's formula.",
            f"What is {a} + {b} * {c} - {d} / {e}? Follow order of operations.",
            f"Convert {rd.randint(10,500)} from base {rd.randint(2,8)} to base {rd.randint(2,8)}.",
            f"Find the greatest common divisor of {a*rd.randint(1,10)} and {b*rd.randint(1,10)}.",
            f"Is {n_val} a prime number? Explain your reasoning step by step.",
            f"Find the limit as x approaches {a} of (x^2 - {b})/(x - {c}).",
            f"Compute the dot product of vectors ({rd.randint(1,5)},{rd.randint(1,5)},{rd.randint(1,5)}) and ({rd.randint(1,5)},{rd.randint(1,5)},{rd.randint(1,5)}).",
            f"The sum of the first {n_val} natural numbers equals n(n+1)/2 = {n_val*(n_val+1)//2}. Verify this.",
            f"In a right triangle with sides {a} and {b}, the hypotenuse equals sqrt({a*a + b*b}) = {int((a*a + b*b)**0.5*100)/100:.2f} approximately.",
        ]
        texts.append(rd.choice(patterns)[:500])
    return texts


def generate_synthetic_text(lang="zh", n=2000):
    """Generate diverse text in given language."""
    import random

    zh_templates = [
        "根据最新研究报告显示，{topic}领域在近{num}年来取得了显著进展。研究人员发现，{finding}这一现象对于理解{field}具有重要价值。",
        "在{field}的发展历程中，{year}年是一个重要的转折点。那一年，{event}的发生彻底改变了人们对{topic}的认识。",
        "随着{tech}技术的不断进步，{industry}行业正在经历前所未有的变革。专家预测，在未来{num}年内，{prediction}将成为现实。",
        "从{place1}到{place2}的旅程让我深刻体会到了{topic}的魅力。沿途的风景和人文气息令人难忘。",
        "关于{topic}的讨论已经持续了很长时间。支持者认为{arg1}，而反对者则强调{arg2}。这场争论反映了人们对{field}的不同理解。",
        "在日常生活中，我们经常会遇到{problem}的问题。解决这个问题的关键在于{key}。通过{method}的方法，可以有效地改善现状。",
        "{person}曾经说过：'{quote}'。这句话在今天看来仍然具有深刻的启示意义，尤其是在{topic}方面。",
        "对比{option1}和{option2}，我们可以看到两者各有优劣。{option1}的优势在于{pro1}，而{option2}则更适合{pro2}的场景。",
        "学习{skill}需要掌握{fundamental}等基础知识，然后通过{method}来逐步提高。最重要的是保持{attitude}的心态。",
        "在{event}的影响下，{field}领域出现了新的发展趋势。越来越多的{group}开始关注{topic}，这预示着未来的发展方向。",
    ]
    topics = ["人工智能", "气候变化", "教育改革", "医疗健康", "经济发展", "文化传承", "科技创新", "社会治理"]
    fields = ["计算机科学", "环境科学", "教育学", "医学", "经济学", "社会学", "物理学", "生物学"]
    techs = ["5G", "人工智能", "区块链", "量子计算", "物联网", "大数据", "云计算"]
    industries = ["制造业", "金融", "医疗", "教育", "农业", "零售", "物流"]
    places = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京"]
    args = ["技术进步带来便利", "传统方法更可靠", "创新是必要的", "稳定更重要"]
    quotes = ["学而不思则罔，思而不学则殆", "知行合一", "千里之行始于足下"]

    en_templates = [
        "Recent studies in {field} have shown that {topic} plays a crucial role in {aspect}. Researchers at {university} found that {finding} after analyzing data from over {num} participants.",
        "The development of {tech} has transformed the {industry} landscape. Companies that embrace {approach} are more likely to succeed in the coming {period}.",
        "When discussing {topic}, it is important to consider both {perspective1} and {perspective2}. Different stakeholders may have varying opinions on the matter.",
        "The history of {field} dates back to {year}, when {pioneer} first proposed the concept of {concept}. Since then, the field has evolved significantly.",
        "One of the most challenging aspects of {problem} is {challenge}. To address this, experts recommend {solution} as the most effective approach.",
        "A comparative analysis of {method1} versus {method2} reveals that each has distinct advantages. {method1} excels at {strength1}, while {method2} is better for {strength2}.",
        "The relationship between {factor1} and {factor2} has been studied extensively. Current evidence suggests a {relationship} correlation between these variables.",
        "In practice, implementing {strategy} requires careful consideration of {factor1}, {factor2}, and {factor3}. Failure to account for any of these can lead to {consequence}.",
        "The future of {domain} will likely be shaped by emerging trends such as {trend1}, {trend2}, and {trend3}. Organizations must adapt to remain competitive.",
        "Understanding {concept} is essential for anyone working in {field}. The fundamental principles include {principle1}, {principle2}, and {principle3}.",
    ]
    universities = ["MIT", "Stanford", "Oxford", "Cambridge", "Harvard", "ETH Zurich"]
    approaches = ["agile methodology", "data-driven decision making", "continuous integration", "user-centered design"]
    methods = ["machine learning", "statistical analysis", "qualitative research", "experimental design"]

    texts = []
    templates_use = zh_templates if lang == "zh" else en_templates
    for _ in range(n):
        t = random.choice(templates_use)
        try:
            text = t.format(
                topic=random.choice(topics) if lang == "zh" else random.choice(["artificial intelligence", "climate change", "education reform", "healthcare", "economic development"]),
                num=random.randint(5, 50), field=random.choice(fields) if lang == "zh" else random.choice(["computer science", "biology", "economics", "physics", "sociology"]),
                finding=random.choice(["这一现象", "这种趋势", "这个结果"]) if lang == "zh" else random.choice(["this phenomenon", "the observed pattern", "the correlation"]),
                year=random.randint(1990, 2024), event=random.choice(["重大突破", "政策变化", "技术革新"]) if lang == "zh" else "a major breakthrough",
                tech=random.choice(techs) if lang == "zh" else random.choice(["AI", "blockchain", "quantum computing", "IoT"]),
                industry=random.choice(industries) if lang == "zh" else random.choice(["manufacturing", "finance", "healthcare"]),
                prediction="这将成为现实" if lang == "zh" else "this will become mainstream",
                place1=random.choice(places), place2=random.choice(places),
                arg1=random.choice(args), arg2=random.choice(args),
                problem="效率低下" if lang == "zh" else "inefficiency",
                key="系统优化" if lang == "zh" else "systematic optimization",
                method="科学" if lang == "zh" else "scientific approaches",
                person="孔子" if lang == "zh" else "Einstein",
                quote=random.choice(quotes) if lang == "zh" else "Imagination is more important than knowledge",
                option1="方案A", option2="方案B", pro1="效率高", pro2="成本低",
                skill="编程" if lang == "zh" else "programming",
                fundamental="基本概念" if lang == "zh" else "basic concepts",
                attitude="积极" if lang == "zh" else "persistent",
                group="年轻人" if lang == "zh" else "young professionals",
                aspect="productivity" if lang == "en" else "",
                university=random.choice(universities) if lang == "en" else "",
                approach=random.choice(approaches) if lang == "en" else "",
                period="decade" if lang == "en" else "",
                perspective1="economic factors" if lang == "en" else "",
                perspective2="social implications" if lang == "en" else "",
                pioneer="Turing" if lang == "en" else "",
                concept="machine learning" if lang == "en" else "",
                challenge="scalability" if lang == "en" else "",
                solution="distributed systems" if lang == "en" else "",
                method1="supervised learning", method2="unsupervised learning",
                strength1="accuracy", strength2="discovery",
                factor1="education", factor2="income level", factor3="geographic location",
                relationship="positive" if lang == "en" else "",
                strategy="digital transformation",
                consequence="project failure" if lang == "en" else "",
                domain="technology", trend1="edge computing", trend2="serverless", trend3="WebAssembly",
                principle1="abstraction", principle2="composition", principle3="interface design",
            )
            texts.append(text[:500])
        except KeyError:
            continue
    return texts


def download_text(url, parser="lines"):
    """Download raw text from URL."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    text = r.text
    if parser == "lines":
        lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 50]
        return lines
    elif parser == "csv":
        # Simple CSV parser
        lines = []
        for line in text.split("\n")[1:]:  # skip header
            parts = line.split(",")
            if len(parts) >= 2:
                content = ",".join(parts[1:]).strip('"').strip()
                if len(content) > 50:
                    lines.append(content)
        return lines
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/real")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--tokenizer", default="e:/taiji/checkpoint-400000/sentencepiece.model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"Tokenizer: {sp.GetPieceSize()} tokens")

    # P0-1c 修复:v2 contract 要求 text token ID = sentencepiece ID + text_offset
    # 见 tokenizer_contract.json: text_offset=13388, text range [13388, 256000)
    # 不加偏移会导致文本 token 落在 image/audio/control 区间,模型看到错位 token
    TEXT_OFFSET = 13388
    print(f"TEXT_OFFSET: {TEXT_OFFSET} (v2 contract: text range [{TEXT_OFFSET}, 256000))")

    all_data = {}

    for domain in ["zh", "en", "code", "math", "general"]:
        cfg = TEXT_SOURCES[domain]
        print(f"\n--- {domain}: {cfg['desc']} ---")

        texts = []

        if cfg["parser"] == "synthetic":
            if domain == "code":
                texts = generate_synthetic_code(args.samples * 2)
            elif domain == "math":
                texts = generate_synthetic_math(args.samples)
            else:
                texts = generate_synthetic_text("en", args.samples)
        else:
            for url in cfg["urls"]:
                try:
                    lines = download_text(url, cfg["parser"])
                    texts.extend(lines)
                    print(f"  Downloaded {len(lines)} lines from {url[:60]}...")
                except Exception as e:
                    print(f"  Download failed: {e}")

        if not texts:
            print(f"  No data, using synthetic fallback")
            texts = generate_synthetic_text("en" if domain != "zh" else "zh", args.samples)

        # Tokenize
        # P0-1c 修复:加 text_offset 把 sentencepiece ID 转成 v2 contract token ID
        # sp.EncodeAsIds 返回 raw sentencepiece ID (范围 [0, 242612))
        # v2 contract 要求 text token ID = sentencepiece ID + text_offset (13388)
        # 这样 text token 落在 [13388, 256000) 的 text range 内
        # control tokens (0-3: pad/unk/bos/eos) 不通过 sentencepiece 生成,
        # 而是由后续 collate 单独添加,所以这里不需要保留 0-3 的特殊处理
        tokens_list = []
        for text in texts:
            text = text.strip().replace("\n", " ")
            if len(text) < 50:
                continue
            encoded = [tid + TEXT_OFFSET for tid in sp.EncodeAsIds(text)]
            if len(encoded) >= args.seq_len:
                tokens_list.append(encoded[:args.seq_len])
            elif len(encoded) >= 20:
                padded = encoded.copy()
                while len(padded) < args.seq_len:
                    padded = padded + encoded
                tokens_list.append(padded[:args.seq_len])
            if len(tokens_list) >= args.samples:
                break

        t = torch.tensor(tokens_list, dtype=torch.long)
        all_data[domain] = t
        torch.save(t, os.path.join(args.output_dir, f"{domain}.pt"))
        print(f"  {t.shape}, range=[{t.min().item()}, {t.max().item()}]")

    torch.save(all_data, os.path.join(args.output_dir, "domain_datasets.pt"))
    print(f"\nSaved: {args.output_dir}/domain_datasets.pt")
    for d, t in all_data.items():
        print(f"  {d}: {t.shape}")
    print("\nNext: python scripts/training/distill_neurons.py --data_dir data/real")


if __name__ == "__main__":
    main()
