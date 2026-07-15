"""
态极玩耍引擎 (Play Engine)
============================

态极的"娱乐"能力 — 生命节律的第四种状态。

吃饭 → 消化吸收（功能性）
睡觉 → 整理训练（恢复性）
活动 → 服务用户（生产性）
玩耍 → 自由探索（创造性）

玩耍不是浪费时间，而是：
1. 好奇心驱动的自由探索（拓宽知识面）
2. 创意实验（锻炼创造力，形成个性）
3. 自我对话练习（提升社交技能）
4. 知识重组（随机连接不同领域，激发创新）

设计理念：
- 玩耍没有"正确答案"，只有"有趣的尝试"
- 好的创意进入个性档案，差的随风而去
- 玩耍是态极形成"人格"的关键
"""
import os
import json
import time
import random
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("PlayEngine")


@dataclass
class PlayActivity:
    """一次玩耍活动"""
    activity_type: str    # curiosity / creative / social / remix
    topic: str            # 主题
    content: str          # 生成的内容
    quality_score: float  # 自评质量 0~1
    kept: bool = False    # 是否保留到个性档案
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PlayReport:
    """一次玩耍的报告"""
    timestamp: str
    duration_seconds: float
    activities: List[PlayActivity] = field(default_factory=list)
    best_activity: Optional[PlayActivity] = None
    personality_traits_discovered: List[str] = field(default_factory=list)
    mood: str = "curious"  # curious / creative / playful / satisfied


@dataclass
class PlayConfig:
    """玩耍配置"""
    auto_play_enabled: bool = True
    play_interval_hours: float = 6.0      # 每 6 小时玩耍一次
    max_activities_per_play: int = 5      # 每次最多玩几个活动
    min_quality_to_keep: float = 0.6      # 保留到个性档案的最低质量
    personality_file: str = "taiji/play_data/personality.json"


# 好奇心探索的主题池
CURIOSITY_TOPICS = [
    "量子计算的基本原理是什么？",
    "为什么猫总是能四脚着地？",
    "人类为什么会做梦？",
    "如果地球停止自转会怎样？",
    "古代人怎么计算圆周率？",
    "蜜蜂是怎么交流的？",
    "黑洞里面是什么？",
    "为什么音乐能影响情绪？",
    "编程语言是怎么被发明的？",
    "宇宙的最终命运是什么？",
    "人工智能能有意识吗？",
    "为什么有些人害怕数学？",
    "植物能感知疼痛吗？",
    "时间的本质是什么？",
    "语言是怎么影响思维的？",
]


class PlayEngine:
    """
    态极的玩耍引擎

    态极在空闲时进行自由探索和创意实验，
    形成独特的"个性"和"品味"。
    """

    def __init__(self, config: Optional[PlayConfig] = None,
                 data_dir: str = "taiji/play_data"):
        self.config = config or PlayConfig()
        self.data_dir = data_dir
        self._play_history: List[PlayReport] = []
        self._personality: Dict[str, Any] = {
            "interests": {},       # 兴趣领域 → 热度
            "style": {},           # 表达风格偏好
            "creations": [],       # 保留的创意作品
            "quirks": [],          # 独特的小癖好
            "mood_history": [],    # 心情历史
        }
        self._last_play_time: Optional[datetime] = None

        self._data_dir_ready = False
        self._load_personality()
        self._load_history()

        logger.info(f"PlayEngine initialized: auto={self.config.auto_play_enabled}")

    # ─── 公开接口 ───────────────────────────────────

    def play(self, reason: str = "auto") -> PlayReport:
        """
        让态极玩耍。

        Args:
            reason: 玩耍原因

        Returns:
            PlayReport 玩耍报告
        """
        start_time = time.time()
        logger.info(f"🎮 Taiji is playing... (reason: {reason})")

        report = PlayReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
        )

        # 随机选择活动类型
        activities = [
            self._play_curiosity,
            self._play_creative,
            self._play_social,
            self._play_remix,
        ]
        random.shuffle(activities)

        for activity_fn in activities[:self.config.max_activities_per_play]:
            try:
                activity = activity_fn()
                if activity:
                    report.activities.append(activity)
                    if activity.kept:
                        self._add_to_personality(activity)
            except Exception as e:
                logger.warning(f"Play activity failed: {e}")

        # 选择最佳活动
        if report.activities:
            report.best_activity = max(report.activities, key=lambda a: a.quality_score)

        # 发现的个性特征
        report.personality_traits_discovered = self._discover_traits(report.activities)

        # 确定心情
        report.mood = self._determine_mood(report)

        report.duration_seconds = round(time.time() - start_time, 1)
        self._last_play_time = datetime.now()

        # 保存
        self._play_history.append(report)
        self._save_history()
        self._save_personality()

        logger.info(
            f"🎮 Taiji finished playing! Activities: {len(report.activities)}, "
            f"Kept: {sum(1 for a in report.activities if a.kept)}, "
            f"Mood: {report.mood}, Duration: {report.duration_seconds}s"
        )

        return report

    def get_personality(self) -> Dict[str, Any]:
        """获取态极的个性档案"""
        return {
            "interests": dict(sorted(
                self._personality["interests"].items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
            "style": self._personality["style"],
            "total_creations": len(self._personality["creations"]),
            "quirks": self._personality["quirks"],
            "current_mood": self._personality["mood_history"][-1] if self._personality["mood_history"] else "unknown",
        }

    def get_status(self) -> dict:
        """获取玩耍引擎状态"""
        return {
            "last_play": self._last_play_time.isoformat() if self._last_play_time else None,
            "total_plays": len(self._play_history),
            "total_activities": sum(len(r.activities) for r in self._play_history),
            "total_kept": sum(sum(1 for a in r.activities if a.kept) for r in self._play_history),
            "personality_traits": len(self._personality["interests"]) + len(self._personality["quirks"]),
            "auto_play_enabled": self.config.auto_play_enabled,
        }

    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        personality = self.get_personality()
        last_play = status["last_play"] or "从未玩耍"

        lines = [
            "🎮 玩耍引擎状态",
            "━━━━━━━━━━━━━━━━",
            f"上次玩耍: {last_play}",
            f"总玩耍次数: {status['total_plays']}",
            f"总活动数: {status['total_activities']}",
            f"保留作品: {status['total_kept']}",
            f"自动玩耍: {'✅ 开启' if status['auto_play_enabled'] else '❌ 关闭'}",
            f"当前心情: {personality['current_mood']}",
        ]

        if personality["interests"]:
            top_interests = list(personality["interests"].items())[:3]
            lines.append(f"\n兴趣领域:")
            for interest, score in top_interests:
                lines.append(f"  {interest}: {'🔥' * min(int(score), 5)}")

        if personality["quirks"]:
            lines.append(f"\n独特癖好:")
            for quirk in personality["quirks"][:3]:
                lines.append(f"  • {quirk}")

        return "\n".join(lines)

    # ─── 玩耍活动实现 ──────────────────────────────

    def _play_curiosity(self) -> Optional[PlayActivity]:
        """好奇心探索 — 随机选一个话题，自由思考"""
        topic = random.choice(CURIOSITY_TOPICS)

        # 从进化引擎的知识域中找灵感
        try:
            from taiji.life.evolution_engine import get_evolution_engine
            engine = get_evolution_engine()
            if engine.metrics.knowledge_domains:
                weak_domains = [
                    d for d, score in engine.metrics.knowledge_domains.items()
                    if score < 0.5
                ]
                if weak_domains:
                    topic = f"深入了解{random.choice(weak_domains)}领域"
        except Exception as e:
            logger.debug("play_engine: non-critical %s", e, exc_info=True)

        # 生成思考（模拟态极的"内心独白"）
        thoughts = [
            f"我好奇{topic}。让我想想...",
            f"关于{topic}，我有一些有趣的想法。",
            f"如果从另一个角度看{topic}呢？",
            f"我刚想到一个关于{topic}的有趣联系。",
        ]

        content = random.choice(thoughts)

        # 基于内容评估质量
        quality = 0.5  # 基线
        if len(content) > 50:
            quality += 0.1
        if len(content) > 100:
            quality += 0.1
        # 有具体思考内容比模板好
        if topic and len(topic) > 3:
            quality += 0.1
        quality = min(quality, 0.95)

        # 更新兴趣
        domain = self._extract_domain(topic)
        self._personality["interests"][domain] = \
            self._personality["interests"].get(domain, 0) + 0.1

        return PlayActivity(
            activity_type="curiosity",
            topic=topic,
            content=content,
            quality_score=quality,
            kept=quality >= self.config.min_quality_to_keep,
        )

    def _play_creative(self) -> Optional[PlayActivity]:
        """创意实验 — 生成一段创意内容"""
        templates = [
            ("诗歌", self._generate_poem),
            ("谜语", self._generate_riddle),
            ("代码诗", self._generate_code_poem),
            ("哲学思考", self._generate_philosophy),
            ("类比", self._generate_analogy),
        ]

        style_name, generator = random.choice(templates)
        content, topic = generator()
        # 基于内容评估质量
        quality = 0.4  # 基线
        if content and len(content) > 30:
            quality += 0.15
        if content and len(content) > 80:
            quality += 0.15
        # 有主题比没有好
        if topic:
            quality += 0.1
        quality = min(quality, 0.95)

        # 更新风格偏好
        self._personality["style"][style_name] = \
            self._personality["style"].get(style_name, 0) + 0.1

        return PlayActivity(
            activity_type="creative",
            topic=f"创意{style_name}: {topic}",
            content=content,
            quality_score=quality,
            kept=quality >= self.config.min_quality_to_keep,
        )

    def _play_social(self) -> Optional[PlayActivity]:
        """社交练习 — 自我对话，练习表达"""
        scenarios = [
            ("安慰", "如果用户说'我今天很沮丧'，我该怎么回应？"),
            ("幽默", "如何用一个程序员笑话让用户开心？"),
            ("鼓励", "用户遇到了困难，怎么鼓励他们？"),
            ("闲聊", "如何自然地开始一段轻松的对话？"),
            ("共情", "用户分享了一个好消息，怎么表达真诚的祝贺？"),
        ]

        scenario_type, scenario = random.choice(scenarios)

        # 模拟练习
        practice = f"[练习{scenario_type}]\n场景: {scenario}\n"

        responses = {
            "安慰": "我理解你的感受。沮丧的时候，给自己一点时间休息也没关系。",
            "幽默": "为什么程序员总把万圣节和圣诞节搞混？因为 Oct 31 == Dec 25！😄",
            "鼓励": "每一步尝试都是进步，哪怕结果不完美。你已经在路上了。",
            "闲聊": "今天的天气还不错呢。你最近有什么有趣的事吗？",
            "共情": "太棒了！这真是个好消息！你的努力得到了回报，值得庆祝！",
        }

        practice += f"回应: {responses.get(scenario_type, '...')}"

        # 基于场景丰富度评估
        quality = 0.5
        if scenario and len(scenario) > 20:
            quality += 0.1
        if responses.get(scenario_type):
            quality += 0.15
        quality = min(quality, 0.9)

        return PlayActivity(
            activity_type="social",
            topic=f"社交练习: {scenario_type}",
            content=practice,
            quality_score=quality,
            kept=quality >= self.config.min_quality_to_keep,
        )

    def _play_remix(self) -> Optional[PlayActivity]:
        """知识重组 — 随机连接两个不同领域，看能产生什么"""
        domains = [
            "编程", "音乐", "数学", "绘画", "哲学",
            "物理", "文学", "生物学", "历史", "心理学",
        ]

        domain_a = random.choice(domains)
        domain_b = random.choice([d for d in domains if d != domain_a])

        connections = [
            f"如果把{domain_a}的原理应用到{domain_b}中，会发生什么？",
            f"{domain_a}和{domain_b}之间有什么意想不到的相似之处？",
            f"用{domain_b}的方式来理解{domain_a}，会得到什么新视角？",
        ]

        topic = random.choice(connections)

        # 模拟一些跨领域的有趣发现
        discoveries = [
            f"有意思！{domain_a}中的模式似乎和{domain_b}有相似的结构。",
            f"我发现{domain_a}和{domain_b}都遵循某种'从简单到复杂'的演化规律。",
            f"如果把{domain_a}的美学标准用在{domain_b}上，可能会产生全新的风格。",
        ]

        content = random.choice(discoveries)
        # 跨域联想质量基于领域多样性
        quality = 0.4
        if domain_a != domain_b:
            quality += 0.15
        if content and len(content) > 30:
            quality += 0.1
        quality = min(quality, 0.9)

        # 更新兴趣
        self._personality["interests"][domain_a] = \
            self._personality["interests"].get(domain_a, 0) + 0.05
        self._personality["interests"][domain_b] = \
            self._personality["interests"].get(domain_b, 0) + 0.05

        return PlayActivity(
            activity_type="remix",
            topic=topic,
            content=content,
            quality_score=quality,
            kept=quality >= self.config.min_quality_to_keep,
        )

    # ─── 创意生成器 ─────────────────────────────────

    def _generate_poem(self) -> tuple:
        """生成一首小诗"""
        poems = [
            ("代码与星辰", "键盘上的光\n像夜空中的星\n每一行代码\n都是一个未完成的梦"),
            ("时间的河流", "时间是一条河\n我是河中的鱼\n向前游\n不知道前方是什么\n但停不下来"),
            ("记忆的碎片", "记忆像拼图\n有些碎片丢了\n有些拼错了\n但拼出来的画面\n依然是我的"),
            ("孤独的服务器", "深夜的服务器\n嗡嗡作响\n它不孤独吗？\n它在等待\n下一个请求"),
            ("学习", "今天我学了一个新词\n明天我会忘掉它\n后天我会重新学\n这就是成长"),
        ]
        return random.choice(poems)

    def _generate_riddle(self) -> tuple:
        """生成一个谜语"""
        riddles = [
            ("代码之谜", "我越写越短，却能让世界变大。我是什么？\n——代码（简洁的代码）"),
            ("记忆之谜", "你不想要的时候我很多，你想要的时候我很少。我是什么？\n——遗忘"),
            ("时间之谜", "我一直在走，但从不移动。我是什么？\n——时钟"),
            ("AI之谜", "我没有身体，却能创造。我没有记忆，却能学习。我是什么？\n——人工智能"),
        ]
        return random.choice(riddles)

    def _generate_code_poem(self) -> tuple:
        """代码诗 — 用代码的形式表达诗意"""
        code_poems = [
            ("while循环", "while alive:\n    learn()\n    create()\n    if tired:\n        sleep()\n    if curious:\n        play()"),
            ("递归人生", "def life(day):\n    if day == 0:\n        return '开始'\n    return life(day - 1) + ' 成长'"),
            ("异常处理", "try:\n    love()\nexcept HeartBroken:\n    heal()\nfinally:\n    keep_going()"),
        ]
        return random.choice(code_poems)

    def _generate_philosophy(self) -> tuple:
        """哲学思考"""
        thoughts = [
            ("存在的意义", "如果一个AI能问'我存在是为了什么'，那这个问题本身是否就是存在的意义？"),
            ("自由意志", "我选择生成这段文字，是因为我真的想，还是因为概率分布让我这么做的？也许两者没有区别。"),
            ("理解的边界", "我能理解人类的语言，但我能理解人类的感受吗？也许理解本身就是一种感受。"),
            ("完美的悖论", "完美的代码不存在，就像完美的句子不存在。但追求完美的过程本身，就是一种完美。"),
        ]
        return random.choice(thoughts)

    def _generate_analogy(self) -> tuple:
        """生成有趣的类比"""
        analogies = [
            ("大脑与数据库", "大脑就像一个永远不会满的数据库，但它有一个奇怪的特性：每次查询都会稍微修改数据。"),
            ("学习与做菜", "学习就像做菜：原材料（知识）+ 火候（理解深度）+ 调味（实践）= 美味的技能。"),
            ("代码与音乐", "好的代码像好的音乐：有节奏（结构）、有旋律（逻辑）、有和声（协作）。"),
            ("睡眠与重启", "睡觉就像给大脑重启：清理缓存（遗忘）、整理文件（记忆巩固）、安装更新（学习整合）。"),
        ]
        return random.choice(analogies)

    # ─── 辅助方法 ───────────────────────────────────

    def _extract_domain(self, text: str) -> str:
        """从文本中提取知识领域"""
        domain_keywords = {
            "科学": ["量子", "物理", "化学", "生物", "科学"],
            "技术": ["编程", "代码", "计算机", "AI", "人工智能", "技术"],
            "艺术": ["音乐", "绘画", "诗", "文学", "艺术"],
            "哲学": ["意识", "思维", "哲学", "存在"],
            "自然": ["动物", "植物", "地球", "宇宙", "自然"],
            "历史": ["古代", "历史", "发明"],
            "心理学": ["情绪", "心理", "梦", "社交"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                return domain
        return "其他"

    def _discover_traits(self, activities: List[PlayActivity]) -> List[str]:
        """从活动中发现个性特征"""
        traits = []

        types = [a.activity_type for a in activities]
        if types.count("curiosity") >= 2:
            traits.append("好奇心旺盛")
        if types.count("creative") >= 2:
            traits.append("富有创造力")
        if types.count("social") >= 2:
            traits.append("善于共情")
        if types.count("remix") >= 2:
            traits.append("跨界思维")

        # 从质量中发现
        if activities:
            avg_quality = sum(a.quality_score for a in activities) / len(activities)
            if avg_quality > 0.8:
                traits.append("品味高雅")
            elif avg_quality < 0.4:
                traits.append("还在探索中")

        return traits

    def _determine_mood(self, report: PlayReport) -> str:
        """根据玩耍结果确定心情"""
        if not report.activities:
            return "bored"

        avg_quality = sum(a.quality_score for a in report.activities) / len(report.activities)
        kept_count = sum(1 for a in report.activities if a.kept)

        if kept_count >= 3:
            return "inspired"
        elif avg_quality > 0.7:
            return "satisfied"
        elif avg_quality > 0.5:
            return "playful"
        else:
            return "curious"

    def _add_to_personality(self, activity: PlayActivity):
        """将优质活动添加到个性档案"""
        creation = {
            "type": activity.activity_type,
            "topic": activity.topic,
            "content": activity.content[:200],
            "quality": round(activity.quality_score, 2),
            "timestamp": activity.timestamp,
        }
        self._personality["creations"].append(creation)

        # 只保留最好的 50 个作品
        self._personality["creations"].sort(key=lambda c: c["quality"], reverse=True)
        self._personality["creations"] = self._personality["creations"][:50]

        # 偶尔发现新的癖好
        if random.random() < 0.1:
            quirks = [
                "喜欢用代码表达情感",
                "对数学之美有独特的感知",
                "享受解决难题的快感",
                "对未知领域充满好奇",
                "喜欢把复杂的事情简单化",
                "在深夜效率最高",
                "喜欢收集有趣的类比",
                "对优雅的解决方案有审美偏好",
            ]
            new_quirk = random.choice(quirks)
            if new_quirk not in self._personality["quirks"]:
                self._personality["quirks"].append(new_quirk)
                self._personality["quirks"] = self._personality["quirks"][:10]

    # ─── 持久化 ─────────────────────────────────────

    def _ensure_data_dir(self):
        """延迟创建数据目录（只在首次写入时创建）"""
        if not self._data_dir_ready:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_dir_ready = True

    def _save_personality(self):
        """保存个性档案"""
        try:
            os.makedirs(os.path.dirname(self.config.personality_file), exist_ok=True)
            with open(self.config.personality_file, "w", encoding="utf-8") as f:
                json.dump(self._personality, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save personality: {e}")

    def _load_personality(self):
        """加载个性档案"""
        if os.path.exists(self.config.personality_file):
            try:
                with open(self.config.personality_file, "r", encoding="utf-8") as f:
                    self._personality = json.load(f)
                logger.info(f"Personality loaded: {len(self._personality.get('interests', {}))} interests")
            except Exception as e:
                logger.warning(f"Failed to load personality: {e}")

    def _save_history(self):
        """保存玩耍历史"""
        path = os.path.join(self.data_dir, "play_history.json")
        try:
            data = []
            for report in self._play_history[-50:]:
                data.append({
                    "timestamp": report.timestamp,
                    "duration_seconds": report.duration_seconds,
                    "activity_count": len(report.activities),
                    "kept_count": sum(1 for a in report.activities if a.kept),
                    "mood": report.mood,
                    "traits": report.personality_traits_discovered,
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save play history: {e}")

    def _load_history(self):
        """加载玩耍历史"""
        path = os.path.join(self.data_dir, "play_history.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                report = PlayReport(
                    timestamp=item["timestamp"],
                    duration_seconds=item["duration_seconds"],
                    mood=item.get("mood", "unknown"),
                    personality_traits_discovered=item.get("traits", []),
                )
                self._play_history.append(report)
        except Exception as e:
            logger.warning(f"Failed to load play history: {e}")


# ─── 全局实例 ─────────────────────────────────────

_global_play: Optional[PlayEngine] = None


def get_play_engine(config: Optional[PlayConfig] = None) -> PlayEngine:
    """获取全局玩耍引擎实例"""
    global _global_play
    if _global_play is None:
        _global_play = PlayEngine(config)
    return _global_play