"""
态极浏览器自动化 (Browser Automation)
======================================

让态极真正打开浏览器，像人一样操作网页。

能力：
1. 打开网页（支持 JavaScript 渲染）
2. 点击按钮/链接
3. 填写表单
4. 滚动页面
5. 截图
6. 提取文本/链接/图片
7. 等待元素出现
8. 执行 JavaScript

使用方式：
    from taiji.tools.browser import Browser
    async with Browser() as b:
        await b.goto("https://example.com")
        text = await b.get_text()
        await b.click("button.submit")
        await b.screenshot("page.png")

依赖：
    pip install playwright
    playwright install chromium
"""
import os
import json
import time
import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger("Taiji.Browser")

# 截图保存目录
SCREENSHOT_DIR = os.path.join("agent_workspace", "screenshots")


class Browser:
    """
    态极浏览器

    基于 Playwright 的无头浏览器自动化。
    """

    def __init__(self, headless: bool = True, timeout: int = 30000):
        self.headless = headless
        self.timeout = timeout
        self._browser = None
        self._page = None
        self._context = None
        self._playwright = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def start(self):
        """启动浏览器"""
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36 TaijiBot/1.0",
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.timeout)
            logger.info("浏览器启动成功")
        except ImportError:
            logger.error("需要安装 playwright: pip install playwright && playwright install chromium")
            raise
        except Exception as e:
            logger.error(f"浏览器启动失败: {e}")
            raise

    async def close(self):
        """关闭浏览器"""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            logger.info("浏览器已关闭")
        except Exception:
            pass

    # ─── 导航 ───────────────────────────────────────

    async def goto(self, url: str, wait_until: str = "domcontentloaded") -> str:
        """
        导航到指定 URL。

        Args:
            url: 网页地址
            wait_until: 等待条件 (load/domcontentloaded/networkidle/commit)

        Returns:
            页面标题
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        await self._page.goto(url, wait_until=wait_until)
        title = await self._page.title()
        logger.info(f"已打开: {title} ({url})")
        return title

    async def back(self):
        """后退"""
        await self._page.go_back()

    async def forward(self):
        """前进"""
        await self._page.go_forward()

    async def reload(self):
        """刷新"""
        await self._page.reload()

    # ─── 交互 ───────────────────────────────────────

    async def click(self, selector: str) -> str:
        """
        点击元素。

        Args:
            selector: CSS 选择器或文本内容

        Returns:
            操作结果
        """
        try:
            await self._page.click(selector)
            return f"已点击: {selector}"
        except Exception as e:
            return f"点击失败: {e}"

    async def type_text(self, selector: str, text: str) -> str:
        """
        在输入框中输入文本。

        Args:
            selector: 输入框选择器
            text: 要输入的文本

        Returns:
            操作结果
        """
        try:
            await self._page.fill(selector, text)
            return f"已输入: {text[:50]}..."
        except Exception as e:
            return f"输入失败: {e}"

    async def press(self, key: str) -> str:
        """
        按键。

        Args:
            key: 键名 (Enter/Tab/Escape/ArrowDown/...)

        Returns:
            操作结果
        """
        await self._page.keyboard.press(key)
        return f"已按键: {key}"

    async def scroll(self, direction: str = "down", amount: int = 500) -> str:
        """
        滚动页面。

        Args:
            direction: 方向 (up/down/left/right)
            amount: 滚动像素

        Returns:
            操作结果
        """
        if direction == "down":
            await self._page.mouse.wheel(0, amount)
        elif direction == "up":
            await self._page.mouse.wheel(0, -amount)
        elif direction == "right":
            await self._page.mouse.wheel(amount, 0)
        elif direction == "left":
            await self._page.mouse.wheel(-amount, 0)
        return f"已滚动: {direction} {amount}px"

    async def wait_for(self, selector: str, timeout: int = 10000) -> str:
        """
        等待元素出现。

        Args:
            selector: CSS 选择器
            timeout: 超时时间（毫秒）

        Returns:
            操作结果
        """
        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            return f"元素已出现: {selector}"
        except Exception as e:
            return f"等待超时: {e}"

    # ─── 内容提取 ───────────────────────────────────

    async def get_text(self, selector: str = None) -> str:
        """
        获取页面/元素文本。

        Args:
            selector: CSS 选择器（None 则获取整个页面）

        Returns:
            文本内容
        """
        if selector:
            elem = await self._page.query_selector(selector)
            if elem:
                return await elem.inner_text()
            return f"未找到元素: {selector}"
        return await self._page.inner_text("body")

    async def get_html(self, selector: str = None) -> str:
        """
        获取页面/元素 HTML。

        Args:
            selector: CSS 选择器（None 则获取整个页面）

        Returns:
            HTML 内容
        """
        if selector:
            elem = await self._page.query_selector(selector)
            if elem:
                return await elem.inner_html()
            return f"未找到元素: {selector}"
        return await self._page.content()

    async def get_links(self) -> List[Dict[str, str]]:
        """获取页面所有链接"""
        links = await self._page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                text: a.innerText.trim(),
                href: a.href,
            })).filter(l => l.text && l.href)
        """)
        return links

    async def get_images(self) -> List[Dict[str, str]]:
        """获取页面所有图片"""
        images = await self._page.evaluate("""
            () => Array.from(document.querySelectorAll('img[src]')).map(img => ({
                alt: img.alt || '',
                src: img.src,
                width: img.naturalWidth,
                height: img.naturalHeight,
            }))
        """)
        return images

    async def screenshot(self, filename: str = None, full_page: bool = False) -> str:
        """
        截图。

        Args:
            filename: 保存文件名（None 则自动生成）
            full_page: 是否截取整个页面

        Returns:
            截图文件路径
        """
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        if not filename:
            filename = f"screenshot_{int(time.time())}.png"
        path = os.path.join(SCREENSHOT_DIR, filename)
        await self._page.screenshot(path=path, full_page=full_page)
        logger.info(f"截图已保存: {path}")
        return path

    async def execute_js(self, code: str) -> Any:
        """
        执行 JavaScript 代码。

        Args:
            code: JavaScript 代码

        Returns:
            执行结果
        """
        return await self._page.evaluate(code)

    # ─── 高级操作 ───────────────────────────────────

    async def search_and_extract(self, query: str, search_url: str = None) -> str:
        """
        搜索并提取结果。

        Args:
            query: 搜索关键词
            search_url: 搜索引擎 URL（None 则用 Google）

        Returns:
            搜索结果文本
        """
        if not search_url:
            search_url = f"https://www.google.com/search?q={query}"

        await self.goto(search_url, wait_until="domcontentloaded")
        await asyncio.sleep(1)  # 等待 JS 渲染

        # 提取搜索结果
        results = await self._page.evaluate("""
            () => {
                const results = [];
                // Google 搜索结果
                document.querySelectorAll('.g').forEach(el => {
                    const title = el.querySelector('h3')?.innerText || '';
                    const link = el.querySelector('a')?.href || '';
                    const snippet = el.querySelector('.VwiC3b')?.innerText || '';
                    if (title && link) {
                        results.push({ title, link, snippet });
                    }
                });
                return results.slice(0, 10);
            }
        """)

        if not results:
            # 降级：提取所有文本
            text = await self.get_text()
            return f"搜索结果（文本提取）:\n{text[:2000]}"

        lines = [f"搜索 '{query}' 找到 {len(results)} 条结果:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            if r['snippet']:
                lines.append(f"   {r['snippet'][:200]}")
            lines.append(f"   URL: {r['link']}")
            lines.append("")

        return "\n".join(lines)

    async def fill_form(self, form_data: Dict[str, str]) -> str:
        """
        填写表单。

        Args:
            form_data: {选择器: 值} 字典

        Returns:
            操作结果
        """
        results = []
        for selector, value in form_data.items():
            try:
                await self._page.fill(selector, value)
                results.append(f"已填写 {selector}: {value[:30]}")
            except Exception as e:
                results.append(f"填写失败 {selector}: {e}")
        return "\n".join(results)

    async def get_page_info(self) -> Dict:
        """获取页面信息"""
        title = await self._page.title()
        url = self._page.url
        text_length = len(await self._page.inner_text("body"))
        links = await self.get_links()
        images = await self.get_images()

        return {
            "title": title,
            "url": url,
            "text_length": text_length,
            "links_count": len(links),
            "images_count": len(images),
        }


# ═══════════════════════════════════════════════
# 同步封装（供工具调用）
# ═══════════════════════════════════════════════

def _run_async(coro):
    """在同步上下文中运行异步代码"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已在异步上下文中，使用线程池
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result(timeout=60)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def browse_open(url: str) -> str:
    """打开网页（供工具调用）"""
    async def _open():
        async with Browser() as b:
            title = await b.goto(url)
            info = await b.get_page_info()
            return f"已打开: {title}\nURL: {info['url']}\n文本长度: {info['text_length']}\n链接数: {info['links_count']}\n图片数: {info['images_count']}"
    return _run_async(_open())


def browse_read(url: str) -> str:
    """读取网页内容（供工具调用）"""
    async def _read():
        async with Browser() as b:
            await b.goto(url, wait_until="networkidle")
            text = await b.get_text()
            return f"网页内容 ({url}):\n\n{text[:8000]}"
    return _run_async(_read())


def browse_click(url: str, selector: str) -> str:
    """打开网页并点击元素（供工具调用）"""
    async def _click():
        async with Browser() as b:
            await b.goto(url)
            result = await b.click(selector)
            await asyncio.sleep(1)
            text = await b.get_text()
            return f"{result}\n\n点击后页面内容:\n{text[:3000]}"
    return _run_async(_click())


def browse_search(query: str) -> str:
    """浏览器搜索（供工具调用）"""
    async def _search():
        async with Browser() as b:
            result = await b.search_and_extract(query)
            return result
    return _run_async(_search())


def browse_screenshot(url: str) -> str:
    """截取网页截图（供工具调用）"""
    async def _screenshot():
        async with Browser() as b:
            await b.goto(url)
            path = await b.screenshot()
            return f"截图已保存: {path}"
    return _run_async(_screenshot())
