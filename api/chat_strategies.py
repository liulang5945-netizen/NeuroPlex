"""
态极推理策略
============

态极是独立生命体，用自己的大脑推理。
不依赖外部云端 API，不套壳。

统一模式：
  用户提问 → ReAct 引擎判断：
    - 能直接回答 → 直接给出 final_answer（1步完成，快速对话）
    - 需要搜索/工具 → 自动调用工具 → 整合结果回答

态极不需要区分"思维"和"行动"，它是一个统一的生命体。
"""
import asyncio
import json
import logging
import threading
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("ApiServer.Chat.Strategies")

# 中国时区
_CST = timezone(timedelta(hours=8))


def _get_current_time_str() -> str:
    """获取当前时间的格式化字符串"""
    now = datetime.now(_CST)
    date_str = now.strftime("%Y年%m月%d日")
    time_str = now.strftime("%H:%M")
    weekday_map = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekday_map[now.weekday()]
    return f"{date_str} {weekday} {time_str}"


def _inject_datetime(system_prompt: str) -> str:
    """注入当前日期时间到系统提示（显著位置，确保小模型也能识别）"""
    time_str = _get_current_time_str()
    dt_block = f"[重要系统信息] 当前时间：{time_str}。今天的日期是{time_str.split()[0]}。\n\n"
    if system_prompt:
        return dt_block + system_prompt
    return dt_block


def _apply_rag(prompt, app_state):
    """从知识库检索相关上下文，注入 prompt"""
    if app_state.rag_kb and app_state.rag_kb.chunks:
        context = app_state.rag_kb.search_with_fallback(prompt)
        if context:
            context_str = "\n---\n".join(context)
            return (f"基于以下参考资料回答问题。\n\n"
                    f"【参考资料】\n{context_str}\n\n【问题】\n{prompt}")
    return prompt


def _get_life_state():
    """读取态极生命状态（仅读取，不记录交互）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        life = get_life_scheduler()
        return life.needs.to_dict()
    except Exception:
        return {}


def _record_life_interaction(success: bool = True, topic: str = "",
                             reasoning_steps: int = 0, used_tools: bool = False,
                             had_search_results: bool = False):
    """记录交互到生命系统（带真实指标）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        life = get_life_scheduler()
        life.record_interaction(
            success=success,
            topic=topic,
            reasoning_steps=reasoning_steps,
            used_tools=used_tools,
            had_search_results=had_search_results,
        )
    except Exception:
        pass


def _get_memory_context():
    """读取近期记忆，注入推理上下文（使用统一上下文管理器）"""
    try:
        from taiji.agent.context_manager import get_context_manager
        ctx = get_context_manager()
        wm_context = ctx._get_working_memory_context(500)
        if wm_context:
            return f"【近期记忆】\n{wm_context}\n\n"
    except Exception:
        pass

    # 回退到旧方式
    try:
        from taiji.agent.working_memory import get_working_memory
        wm = get_working_memory()
        all_memories = wm.export_all()
        if all_memories:
            lines = [f"- {str(v)[:100]}" for k, v in list(all_memories.items())[:5]]
            if lines:
                return "【近期记忆】\n" + "\n".join(lines) + "\n\n"
    except Exception:
        pass
    return ""


def _build_history(request):
    """构建对话历史"""
    history = []
    if request.history:
        for u, a in request.history:
            if u:
                history.append({"role": "user", "content": u})
            if a:
                history.append({"role": "assistant", "content": a})
    return history


def _record_evolution(prompt, result_text, success):
    """记录到进化引擎"""
    try:
        from taiji.life.evolution_engine import get_evolution_engine
        evo = get_evolution_engine()
        if success:
            evo.record_task_success(
                task=prompt[:200],
                steps=[{"action": "chat", "tool": "react"}],
                final_answer=result_text[:200],
            )
        else:
            evo.record_task_failure(
                task=prompt[:200],
                error=result_text[:200] if result_text else "empty",
            )
    except Exception:
        pass


def _has_react_engine() -> bool:
    """检查 ReAct 引擎是否可用"""
    try:
        from taiji.agent_ext.react_engine import ReActEngine
    except ImportError:
        return False

    from taiji.core.app_state import app_state

    # 有 trainer（模型已加载）就支持 ReAct
    # _call_local_model 会自动根据 tokenizer 类型选择 prompt 格式
    if app_state.trainer is not None:
        return True

    return False


async def _stream_unified(request, prompt, app_state, stop_event, collector):
    """
    态极统一推理模式。

    始终使用 ReAct 引擎：
    - 简单问题 → 1步直接回答（和纯文本生成一样快）
    - 需要工具 → 自动调用搜索/工具 → 整合结果回答

    产出结构化 SSE 事件（前端统一解析）：
      {"type":"life","data":{"needs":{...}}}
      {"type":"thought","data":{"step":1,"content":"..."}}
      {"type":"action","data":{"step":1,"tool":"search","args":{...}}}
      {"type":"observation","data":{"step":1,"tool":"search","result":"..."}}
      {"type":"final","data":{"answer":"...","step":2}}
    """
    life_needs = _get_life_state()
    fatigue = life_needs.get("fatigue", 0)
    curiosity = life_needs.get("curiosity", 50)

    # 生命状态影响推理深度
    max_steps = 6
    if fatigue > 80:
        max_steps = 3
    elif curiosity > 80:
        max_steps = 10

    # 使用统一上下文管理器构建上下文
    system_prompt = _inject_datetime(request.system_prompt or "")
    enriched_prompt = prompt
    history = _build_history(request)

    try:
        from taiji.agent.context_manager import get_context_manager
        ctx = get_context_manager()

        # 注入对话历史到上下文管理器
        if request.history:
            for u, a in request.history:
                if u:
                    ctx.add_message("user", u)
                if a:
                    ctx.add_message("assistant", a)

        # 构建带记忆的 prompt
        enriched_prompt = ctx.build_context(
            task=prompt,
            system_prompt=system_prompt,
            include_history=True,
            include_memory=True,
        )

        # 构建消息列表（用于 ReAct 引擎）
        history = ctx._get_recent_history_messages()
    except Exception:
        # 回退到旧方式
        memory_context = _get_memory_context()
        enriched_prompt = (memory_context + prompt) if memory_context else prompt

    # 发送生命状态
    if life_needs:
        yield f'data: {json.dumps({"type": "life", "data": {"needs": life_needs}}, ensure_ascii=False)}\n\n'

    full_text = ""
    # 跟踪推理真实指标（用于生命系统）
    reasoning_steps = 0
    used_tools = False
    had_search_results = False

    # 尝试 ReAct 引擎（统一推理）
    if _has_react_engine():
        try:
            from taiji.agent_ext.react_engine import ReActEngine
            engine = ReActEngine(max_steps=max_steps)

            for event in engine.run_stream(
                task=enriched_prompt,
                system_prompt=system_prompt,
                history=history,
            ):
                if stop_event.is_set():
                    logger.info("推理被用户停止")
                    break

                event_type = event.get("type", "")
                event_data = event.get("data", {})

                if event_type == "final":
                    full_text = event_data.get("answer", "")
                elif event_type == "thought":
                    full_text += event_data.get("content", "")
                    reasoning_steps += 1
                elif event_type == "action":
                    used_tools = True
                    reasoning_steps += 1
                elif event_type == "observation":
                    # 检查是否获得了搜索结果
                    tool_name = event_data.get("tool", "")
                    result_text = event_data.get("result", "")
                    if tool_name and ("search" in tool_name.lower() or "fetch" in tool_name.lower() or "browse" in tool_name.lower()):
                        if result_text and len(str(result_text).strip()) > 50:
                            had_search_results = True

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"ReAct engine error: {e}, falling back to direct generation")
            full_text = ""
            # 如果 ReAct 引擎出错，回退到直接生成
            async for chunk in _stream_fallback(enriched_prompt, system_prompt, app_state, stop_event):
                yield chunk
                full_text += chunk
    else:
        # ReAct 引擎不可用，回退到直接文本生成
        logger.info("ReAct 引擎不可用，使用直接生成模式")
        async for chunk in _stream_fallback(enriched_prompt, system_prompt, app_state, stop_event):
            yield chunk
            full_text += chunk

    success = bool(full_text and not full_text.startswith("["))
    _record_evolution(request.prompt, full_text, success)

    # 记录交互到生命系统（带真实指标）
    _record_life_interaction(
        success=success,
        topic=request.prompt[:50],
        reasoning_steps=reasoning_steps,
        used_tools=used_tools,
        had_search_results=had_search_results,
    )

    try:
        if collector and full_text:
            collector.collect_conversation(request.prompt, full_text)
            collector.flush()
    except Exception:
        pass
    yield "data: [DONE]\n\n"


async def _stream_fallback(prompt, system_prompt, app_state, stop_event):
    """
    回退生成模式 — 当 ReAct 引擎不可用时使用。
    使用直接文本生成，产出兼容前端的结构化事件。
    """
    full_text = ""

    # 根据模型类型选择 prompt 格式
    taiji = app_state.get_taiji_engine()
    tokenizer = app_state.get_tokenizer()
    is_native = app_state.is_taiji()

    if is_native:
        # ModelSelf 原生模型：使用 [系统]/[用户]/[助手] 格式
        formatted = f"[系统] {system_prompt}\n[用户] {prompt}\n[助手]"
    elif tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        # HF 模型：优先使用 tokenizer 的 chat_template
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = f"{system_prompt}\n\n{prompt}"
    else:
        formatted = f"{system_prompt}\n\n{prompt}"

    if taiji and tokenizer:
        try:
            for chunk in taiji.generate_stream(formatted, tokenizer, max_new_tokens=512, stop_event=stop_event):
                if stop_event.is_set():
                    break
                full_text += chunk
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.warning(f"态极推理失败: {e}")
            yield f"data: {json.dumps(f'[推理失败: {e}]', ensure_ascii=False)}\n\n"
    else:
        try:
            trainer = app_state.get_trainer()
            if trainer and hasattr(trainer, 'generate_stream'):
                for chunk in trainer.generate_stream(formatted, tokenizer, max_new_tokens=512, stop_event=stop_event):
                    if stop_event.is_set():
                        break
                    full_text += chunk
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)
            elif trainer and hasattr(trainer, 'generate'):
                full_text = trainer.generate(formatted, tokenizer, max_new_tokens=512)
                if "### Response:\n" in full_text:
                    full_text = full_text.split("### Response:\n", 1)[-1].strip()
                yield f"data: {json.dumps(full_text, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps('[模型未加载]', ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(f'[推理失败: {e}]', ensure_ascii=False)}\n\n"


def create_event_generator(request, app_state, collector_factory):
    """
    统一推理入口。

    态极是一个统一的生命体，不需要区分思维和行动。
    所有对话都通过统一推理流程：
    - 能直接回答 → 1步完成（快速）
    - 需要工具 → 自动调用搜索/工具
    """
    async def event_generator():
        stop_event = threading.Event()
        try:
            collector = None
            try:
                collector = collector_factory()
            except Exception:
                pass

            prompt = _apply_rag(request.prompt, app_state)

            # 统一推理：始终使用 ReAct 引擎（自动判断是否需要工具）
            async for chunk in _stream_unified(request, prompt, app_state, stop_event, collector):
                yield chunk

        except (GeneratorExit, RuntimeError, asyncio.CancelledError):
            logger.info("推理客户端已断开连接，正在停止生成...")
            stop_event.set()
        except Exception as e:
            logger.error(f"推理生成出错: {e}")
            yield f"data: {json.dumps(f'生成出错: {e}', ensure_ascii=False)}\n\n"
        finally:
            stop_event.set()

    return event_generator