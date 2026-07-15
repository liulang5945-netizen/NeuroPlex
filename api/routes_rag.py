"""
RAG 知识库 API 路由
"""
import json
import logging
import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks

from taiji.core.app_state import app_state
from taiji.core.utils import get_external_path
from taiji.tools.rag import RAGKnowledgeBase, RAGConfig

from .models import RAGSearchRequest

logger = logging.getLogger("ApiServer.RAG")
router = APIRouter()


def _process_rag_file_background(file_path: str):
    """后台处理：嵌入模型向量化"""
    try:
        app_state.rag_kb.add_file(file_path)
        app_state.rag_kb.rebuild_index()
        logger.info(f"✅ 后台 RAG 向量化建库完成: {file_path}")
    except Exception as e:
        logger.error(f"❌ 后台 RAG 向量化失败: {e}")


@router.post("/api/rag/upload")
def upload_rag_document(file: UploadFile = File(...), bg_tasks: BackgroundTasks = BackgroundTasks()):
    """接收前端上传的文档，加入 RAG 知识库"""
    try:
        doc_dir = get_external_path("docs")
        os.makedirs(doc_dir, exist_ok=True)
        file_path = os.path.join(doc_dir, os.path.basename(file.filename))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        bg_tasks.add_task(_process_rag_file_background, file_path)
        return {"status": "success", "message": f"文件 {file.filename} 已上传，正在后台向量化建库，请稍后查看！"}
    except Exception as e:
        logger.error(f"RAG 添加文件失败: {e}")
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.delete("/api/rag/clear")
async def clear_rag_documents():
    """清空 RAG 知识库及本地文档"""
    try:
        doc_dir = get_external_path("docs")
        if os.path.exists(doc_dir):
            shutil.rmtree(doc_dir, ignore_errors=True)
        app_state.update_rag_kb(RAGKnowledgeBase(persist_dir=get_external_path("rag_data")))
        return {"status": "success", "message": "知识库已清空！"}
    except Exception as e:
        logger.error(f"RAG 清空失败: {e}")
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.get("/api/rag/files")
def list_rag_files():
    """获取已挂载的 RAG 文件列表"""
    return {"files": app_state.rag_kb.get_doc_names()}


@router.delete("/api/rag/file/{filename:path}")
def delete_rag_file(filename: str):
    """删除指定的 RAG 文件"""
    try:
        doc_dir = os.path.abspath(get_external_path("docs"))
        doc_path = os.path.abspath(os.path.join(doc_dir, filename))
        if not (doc_path == doc_dir or doc_path.startswith(doc_dir + os.sep)):
            raise HTTPException(status_code=403, detail="路径不安全")
        if os.path.exists(doc_path):
            os.remove(doc_path)
        app_state.rag_kb.remove_file(filename)
        app_state.rag_kb.rebuild_index()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


# ======================== RAG 检索策略配置 API ========================

@router.get("/api/rag/config")
async def get_rag_config():
    """获取当前 RAG 检索策略配置"""
    try:
        config = RAGConfig()
        return {"status": "success", "config": config.to_dict()}
    except Exception as e:
        logger.error(f"获取 RAG 配置失败: {e}")
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.put("/api/rag/config")
async def update_rag_config(updates: dict):
    """更新 RAG 检索策略配置

    可更新字段:
    - enable_hybrid: bool  是否启用混合检索 (Dense + BM25)
    - enable_reranker: bool  是否启用 Cross-Encoder 重排序
    - enable_query_rewrite: bool  是否启用查询改写
    - candidate_k: int  混合检索候选数
    - reranker_model: str  重排序模型名称
    """
    try:
        config = RAGConfig()
        valid_keys = set(RAGConfig.DEFAULTS.keys())
        filtered = {k: v for k, v in updates.items() if k in valid_keys}
        if not filtered:
            raise HTTPException(status_code=400, detail="无有效的配置字段")
        config.update(filtered)
        return {"status": "success", "updated": list(filtered.keys()), "config": config.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新 RAG 配置失败: {e}")
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.get("/api/rag/status")
async def get_rag_status():
    """获取 RAG 知识库状态信息"""
    try:
        kb = app_state.rag_kb
        if not kb:
            return {"status": "not_initialized"}
        return {
            "status": "ok",
            "doc_count": len(kb.documents),
            "chunk_count": len(kb.chunks),
            "has_embeddings": kb.embeddings is not None,
            "has_bm25": kb._bm25_index is not None,
            "bm25_doc_count": kb._bm25_index.doc_count if kb._bm25_index else 0,
            "embed_dim": kb._embed_dim,
        }
    except Exception as e:
        logger.error(f"获取 RAG 状态失败: {e}")
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.post("/api/rag/search")
async def rag_search(req: RAGSearchRequest):
    """在知识库中进行语义搜索"""
    try:
        if not app_state.rag_kb or not app_state.rag_kb.chunks:
            return {"results": []}
        results = app_state.rag_kb.search(req.query, top_k=req.top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"RAG 搜索失败: {e}")
        return {"results": []}


@router.get("/api/rag/preview/{filename:path}")
def rag_preview(filename: str):
    """预览 RAG 文档内容"""
    try:
        doc_dir = os.path.abspath(get_external_path("docs"))
        doc_path = os.path.abspath(os.path.join(doc_dir, filename))
        if not (doc_path == doc_dir or doc_path.startswith(doc_dir + os.sep)):
            raise HTTPException(status_code=403, detail="路径不安全")
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(10000)
            return {"content": content}
        return {"content": "(文件不存在)"}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")