/**
 * Markdown 渲染 composable
 */
import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import hljs from 'highlight.js';
import 'highlight.js/styles/atom-one-dark.css';
import DOMPurify from 'dompurify';

const marked = new Marked(
  markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code, lang) {
      const language = hljs.getLanguage(lang) ? lang : 'plaintext';
      return hljs.highlight(code, { language }).value;
    }
  })
);

marked.use({
  renderer: {
    code(code, lang) {
      const language = lang || 'text';
      return `<div class="code-block-wrapper">
  <div class="code-header">
    <span class="code-lang">${language}</span>
    <button class="code-copy-btn" data-action="copy-code">📋 复制</button>
  </div>
  <pre><code class="hljs language-${language}">${code}</code></pre>
  </div>`;
    }
  }
});

// 安全复制
const safeCopyText = async (text) => {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (e) { /* ignore and fallback */ }
  }
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.top = "-999999px";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  try { document.execCommand('copy'); } catch (err) {}
  document.body.removeChild(textArea);
};

// 事件委托：处理代码块复制按钮点击
if (typeof document !== 'undefined' && !window.__taijiMarkdownCopyHandler) {
  window.__taijiMarkdownCopyHandler = true;
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-action="copy-code"]');
    if (!btn) return;
    try {
      const pre = btn.parentElement.nextElementSibling;
      await safeCopyText(pre.innerText);
      const oldText = btn.innerText;
      btn.innerText = '✅ 成功';
      setTimeout(() => { btn.innerText = oldText; }, 2000);
    } catch (err) {
      console.error('复制失败', err);
    }
  });
}

/**
 * 解析消息内容，分离思考过程和最终回答
 * 支持格式：
 *   1. 或 <think></think> 标签（大小写不敏感）
 *   2. 思考过程\n\n---\n\n回答内容
 *   3. 思考过程\n\n回答内容（当文本中出现"最终答案："等标记时）
 *   4. 中文"思考过程："等模式（含前导空白）
 * @param {string} text - 原始消息文本
 * @returns {{ reasoning: string, content: string }}
 */
function parseMessageContent(text) {
  if (!text) return { reasoning: '', content: '' };

  let reasoning = '';
  let content = text;

  // 1. 匹配 <think> 或 <推理> 标签（大小写不敏感，支持已闭合和流式未闭合）
  const thinkOpenMatch = text.match(/<(?:think|THINK|推理)>/i);
  if (thinkOpenMatch) {
    const openTag = thinkOpenMatch[0];
    const openIdx = thinkOpenMatch.index;
    const afterOpen = text.slice(openIdx + openTag.length);

    // 查找对应的闭合标签
    const thinkCloseMatch = afterOpen.match(/<\/(?:think|THINK|推理)>/i);
    if (thinkCloseMatch) {
      // 已闭合：正常拆分
      reasoning = afterOpen.slice(0, thinkCloseMatch.index).trim();
      const beforeThink = text.slice(0, openIdx);
      const afterThink = afterOpen.slice(thinkCloseMatch.index + thinkCloseMatch[0].length);
      content = (beforeThink + afterThink).trim();
      if (reasoning) return { reasoning, content };
    } else {
      // 流式未闭合：`<think>` 已打开但 `</think>` 尚未到达
      // 将思考内容暂存，正文置空（等待后续 chunk 填充闭合后正文）
      reasoning = afterOpen.trim();
      content = text.slice(0, openIdx).trim();
      return { reasoning, content: content || '' };
    }
  }

  // 3. 匹配 思考(?:过程)?：... 然后换行分隔的回答
  //    例如 "思考过程：...\n最终答案：..."（放宽行首锚点，允许前缀空白）
  const thoughtPatterns = [
    /思考[：:]\s*([\s\S]*?)(?=\n(?:最终)?(?:回答|答案)[：:]|\n---|\n\n\n|$)/,
    /思考过程[：:]\s*([\s\S]*?)(?=\n(?:最终)?(?:回答|答案)[：:]|\n---|\n\n\n|$)/,
    /^\s*[Tt]hought[：:]\s*([\s\S]*?)(?=\n\s*(?:[Aa]nswer|[Ff]inal)[：:]|\n---|\n\n\n|$)/m,
    /^\s*[Rr]easoning[：:]\s*([\s\S]*?)(?=\n\s*(?:[Aa]nswer|[Ff]inal)[：:]|\n---|\n\n\n|$)/m,
  ];

  for (const pattern of thoughtPatterns) {
    const match = text.match(pattern);
    if (match && match[1] && match[1].trim().length > 5) {
      reasoning = match[1].trim();
      const after = text.slice(match.index + match[0].length);
      content = after.replace(/^(?:最终)?(?:回答|答案)[：:]\s*/, '').replace(/^---\s*\n?/, '').trim();
      if (reasoning) return { reasoning, content };
    }
  }

  // 4. 匹配 "Reasoning: ... Answer: ..." 格式（英文模型常见输出）
  const engMatch = text.match(/^Reasoning:\s*([\s\S]*?)\nAnswer:\s*([\s\S]*)$/im);
  if (engMatch) {
    return { reasoning: engMatch[1].trim(), content: engMatch[2].trim() };
  }

  // 5. 匹配 "### 思考过程" 或 "## 思考" 等 markdown 标题分隔
  const headerMatch = text.match(/^#{1,3}\s*思考(?:过程)?\s*\n([\s\S]*?)(?=\n#{1,3}\s*(?:最终)?(?:回答|答案)|\n---|$)/m);
  if (headerMatch) {
    reasoning = headerMatch[1].trim();
    content = text.slice(headerMatch.index + headerMatch[0].length).replace(/^#{1,3}\s*(?:最终)?(?:回答|答案)[：:]?\s*/m, '').trim();
    if (reasoning) return { reasoning, content };
  }

  // 6. 没有找到分隔标记，将文本按第一个 \n---\n 或 \n\n\n 分割尝试
  const sepMatch = text.match(/^([\s\S]*?)\n---\n([\s\S]*)$/);
  if (sepMatch) {
    const first = sepMatch[1].trim();
    const second = sepMatch[2].trim();
    // 只有当第一部分看起来像思考过程时才拆分
    if (first.length < second.length * 1.5 && first.length > 20) {
      return { reasoning: first, content: second };
    }
  }

  // 默认：无思考过程
  return { reasoning: '', content: text };
}

export function useMarkdown() {
  // DOMPurify 配置：允许 img 标签及常见图片属性
  const purifyConfig = {
    ADD_TAGS: ['img'],
    ADD_ATTR: ['src', 'alt', 'width', 'height', 'loading', 'decoding', 'style', 'title'],
    ALLOW_DATA_ATTR: false,
  };

  const renderMarkdown = (text) => {
    if (!text) return '';
    const raw = marked.parse(text);
    return typeof raw === 'string' ? DOMPurify.sanitize(raw, purifyConfig) : raw;
  };

  const formatDuration = (seconds) => {
    if (!seconds || seconds <= 0) return '-';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return h + 'h ' + m + 'm';
    if (m > 0) return m + 'm ' + s + 's';
    return s + 's';
  };

  return { renderMarkdown, formatDuration, parseMessageContent };
}
