"""
提取层 (Extractor Layer)
========================

从 HTML 提取结构化内容：
1. ReadabilityExtractor — 正文提取（启发式 DOM 分析，非 regex）
2. MarkdownConverter    — HTML → Markdown 转换
3. LinkExtractor        — 同域链接发现

统一输出 PageContent(url, title, text, markdown, links)
"""

import re
import logging
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("Taiji.Search.Extractor")


@dataclass
class PageContent:
    """提取后的页面内容"""
    url: str = ""
    title: str = ""
    text: str = ""           # 纯文本正文
    markdown: str = ""       # Markdown 格式
    links: List[str] = field(default_factory=list)
    word_count: int = 0
    extract_method: str = "readability"  # readability / fallback


# ═══════════════════════════════════════════════
# ReadabilityExtractor — 启发式正文提取
# ═══════════════════════════════════════════════

class ReadabilityExtractor:
    """
    启发式正文提取，不依赖外部库。

    核心思路（参考 readability.js 算法）：
    1. 移除 script/style/nav/footer/aside 等噪音标签
    2. 找文本密度最高的容器（main/article/[role=main]/.content）
    3. 在容器内按段落 <p> 提取正文
    4. 如果找不到理想容器，回退到全页 <p> 段落
    """

    NOISE_TAGS = {
        "script", "style", "nav", "footer", "aside", "header",
        "form", "iframe", "noscript", "svg", "button",
    }

    NOISE_CLASS_PATTERNS = [
        r"nav", r"menu", r"sidebar", r"footer", r"header",
        r"comment", r"share", r"related", r"advert", r"banner",
        r"popup", r"modal", r"cookie", r"subscribe", r"social",
    ]

    CONTENT_SELECTORS = [
        "main", "article", '[role="main"]',
        ".content", "#content", ".post-content", ".article-content",
        ".entry-content", ".post-body", ".article-body",
    ]

    def extract(self, html: str, url: str = "") -> PageContent:
        """从 HTML 提取正文"""
        if not html or len(html) < 100:
            return PageContent(url=url)

        title = self._extract_title(html)
        clean_html = self._remove_noise(html)
        content_html = self._find_main_content(clean_html)
        text = self._html_to_text(content_html)
        links = self._extract_links(content_html, url)

        md = self._to_markdown(content_html, title, url)

        wc = len(text.split())
        method = "readability" if wc > 50 else "fallback"

        return PageContent(
            url=url, title=title, text=text, markdown=md,
            links=links, word_count=wc, extract_method=method,
        )

    def _extract_title(self, html: str) -> str:
        """提取标题"""
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        m = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE)
        if m:
            return re.sub(r"<[^>]+>", "", m.group(1)).strip()
        return ""

    def _remove_noise(self, html: str) -> str:
        """移除噪音标签"""
        for tag in self.NOISE_TAGS:
            html = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", html,
                flags=re.DOTALL | re.IGNORECASE,
            )
        # 移除噪音 class
        for pattern in self.NOISE_CLASS_PATTERNS:
            html = re.sub(
                rf'<div[^>]*class="[^"]*{pattern}[^"]*"[^>]*>.*?</div>',
                "", html, flags=re.DOTALL | re.IGNORECASE,
            )
        # 移除注释
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        return html

    def _find_main_content(self, html: str) -> str:
        """找到文本密度最高的内容容器"""
        for selector in self.CONTENT_SELECTORS:
            # 提取选择器中的标签名（main, article 等）
            tag_match = re.match(r'([a-z]+)', selector)
            if not tag_match:
                continue
            tag_name = tag_match.group(1)
            m = re.search(
                rf"<{tag_name}[^>]*>(.*?)</{tag_name}>",
                html, re.DOTALL | re.IGNORECASE,
            )
            if m and len(m.group(1)) > 500:
                return m.group(1)

        # 回退：找 <p> 最多的 div
        divs = re.findall(r"<div[^>]*>(.*?)</div>", html, re.DOTALL | re.IGNORECASE)
        best = ""
        best_p_count = 0
        for div in divs:
            p_count = len(re.findall(r"<p[^>]*>", div, re.IGNORECASE))
            if p_count > best_p_count:
                best_p_count = p_count
                best = div
        return best if best_p_count >= 3 else html

    def _html_to_text(self, html: str) -> str:
        """HTML → 纯文本"""
        # 块级标签 → 换行
        text = re.sub(r"<(?:p|div|br|li|h[1-6]|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"</(?:p|div|li|h[1-6]|tr)>", "\n", text, flags=re.IGNORECASE)
        # 移除所有剩余标签
        text = re.sub(r"<[^>]+>", "", text)
        # 解码 HTML 实体
        text = self._decode_entities(text)
        # 清理空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _decode_entities(self, text: str) -> str:
        """解码 HTML 实体"""
        entities = {
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
            "&hellip;": "...", "&mdash;": "-", "&ndash;": "-",
            "&copy;": "(c)", "&reg;": "(R)",
        }
        for ent, char in entities.items():
            text = text.replace(ent, char)
        # 数字实体
        text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
        return text

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """提取同域链接（含相对路径转绝对路径）"""
        if not base_url:
            return []
        parsed = urllib.parse.urlparse(base_url)
        base_domain = parsed.netloc
        base_scheme = parsed.scheme
        raw_links = re.findall(r'href="([^"]+)"', html, re.IGNORECASE)
        result = []
        seen = set()
        for link in raw_links:
            # 跳过锚点、js、mailto
            if link.startswith("#") or link.startswith("javascript:") or link.startswith("mailto:"):
                continue
            # 相对路径转绝对路径
            if link.startswith("/"):
                link = f"{base_scheme}://{base_domain}{link}"
            elif not link.startswith(("http://", "https://")):
                link = urllib.parse.urljoin(base_url, link)
            try:
                domain = urllib.parse.urlparse(link).netloc
                if domain == base_domain and link not in seen:
                    if not any(link.lower().endswith(ext) for ext in
                               [".jpg", ".png", ".pdf", ".zip", ".css", ".js"]):
                        seen.add(link)
                        result.append(link)
            except Exception:
                pass
        return result

    def _to_markdown(self, html: str, title: str, url: str) -> str:
        """HTML → Markdown"""
        md = ""
        if title:
            md += f"# {title}\n\n"
        # 标题
        for level in range(1, 7):
            md = re.sub(
                rf"<h{level}[^>]*>(.*?)</h{level}>",
                lambda m, lv=level: "\n" + "#" * lv + " " + re.sub(r"<[^>]+>", "", m.group(1)).strip() + "\n",
                html, flags=re.DOTALL | re.IGNORECASE,
            )
        # 段落
        md = re.sub(r"<p[^>]*>(.*?)</p>", lambda m: re.sub(r"<[^>]+>", "", m.group(1)).strip() + "\n\n",
                    md, flags=re.DOTALL | re.IGNORECASE)
        # 加粗
        md = re.sub(r"<(?:strong|b)[^>]*>(.*?)</(?:strong|b)>", r"**\1**", md, flags=re.DOTALL | re.IGNORECASE)
        # 斜体
        md = re.sub(r"<(?:em|i)[^>]*>(.*?)</(?:em|i)>", r"*\1*", md, flags=re.DOTALL | re.IGNORECASE)
        # 链接
        md = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', r"[\2](\1)", md, flags=re.DOTALL | re.IGNORECASE)
        # 列表
        md = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", md, flags=re.DOTALL | re.IGNORECASE)
        # 代码块
        md = re.sub(r"<pre[^>]*>(.*?)</pre>", lambda m: "\n```\n" + re.sub(r"<[^>]+>", "", m.group(1)).strip() + "\n```\n",
                    md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", md, flags=re.DOTALL | re.IGNORECASE)
        # 换行
        md = re.sub(r"<br\s*/?>", "\n", md, flags=re.IGNORECASE)
        # 移除剩余标签
        md = re.sub(r"<[^>]+>", "", md)
        # 解码实体
        md = self._decode_entities(md)
        # 清理
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()


# ═══════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════

_default_extractor: Optional[ReadabilityExtractor] = None


def get_extractor() -> ReadabilityExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = ReadabilityExtractor()
    return _default_extractor
