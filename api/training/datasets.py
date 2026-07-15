"""
数据集管理 API 路由
上传、列表、删除、预览微调数据集
"""
import json
import logging
import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File

from taiji.core.utils import get_external_path

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/train/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """接收前端上传的微调数据集"""
    try:
        data_dir = get_external_path("data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, os.path.basename(file.filename))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "status": "success",
            "path": f"data/{file.filename}",
            "message": f"数据集 `{file.filename}` 已成功上传并选中！"
        }
    except Exception as e:
        logger.error(f"数据集上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_all_data_dirs() -> list:
    """获取所有可能的数据目录（解决打包/开发环境路径不一致问题）"""
    dirs = []
    primary = get_external_path("data")
    dirs.append(primary)
    # 回退：项目根目录下的 data/
    import sys
    if getattr(sys, 'frozen', False):
        project_root = os.path.dirname(sys.executable)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback = os.path.join(project_root, "data")
    if fallback not in dirs:
        dirs.append(fallback)
    return dirs


@router.get("/api/train/files")
def list_train_files():
    """获取 data 目录下的数据集文件列表"""
    all_files = set()
    for data_dir in _get_all_data_dirs():
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if os.path.isfile(os.path.join(data_dir, f)):
                    all_files.add(f)
    # 确保主目录存在
    os.makedirs(get_external_path("data"), exist_ok=True)
    return {"files": sorted(all_files)}


@router.delete("/api/train/file/{filename:path}")
def delete_train_file(filename: str):
    """删除指定的数据集文件"""
    try:
        for data_dir in _get_all_data_dirs():
            data_dir = os.path.abspath(data_dir)
            file_path = os.path.abspath(os.path.join(data_dir, filename))
            if not (file_path == data_dir or file_path.startswith(data_dir + os.sep)):
                continue
            if os.path.exists(file_path):
                os.remove(file_path)
                return {"status": "success"}
        return {"status": "error", "message": "文件不存在"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/train/preview/{filename:path}")
def train_preview(filename: str):
    """预览数据集内容"""
    try:
        # 在所有数据目录中查找文件
        data_path = None
        for data_dir in _get_all_data_dirs():
            data_dir = os.path.abspath(data_dir)
            candidate = os.path.abspath(os.path.join(data_dir, filename))
            if candidate.startswith(data_dir + os.sep) and os.path.exists(candidate):
                data_path = candidate
                break
        if not data_path:
            return {"samples": [], "count": 0}

        import jsonlines
        samples = []
        count = 0
        with jsonlines.open(data_path) as reader:
            for item in reader:
                if count < 5:
                    instruction = item.get("instruction", item.get("question", item.get("input", "")))
                    output = item.get("output", item.get("answer", item.get("response", "")))
                    samples.append({"instruction": instruction[:200], "output": output[:300]})
                count += 1
        return {"samples": samples, "count": count}
    except Exception as e:
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                samples = []
                for i, item in enumerate(data[:5]):
                    instruction = item.get("instruction", item.get("question", ""))
                    output = item.get("output", item.get("answer", ""))
                    samples.append({"instruction": str(instruction)[:200], "output": str(output)[:300]})
                return {"samples": samples, "count": len(data)}
        except Exception as fallback_e:
            logger.debug(f"JSON 回退读取也失败: {fallback_e}")
        raise HTTPException(status_code=500, detail=str(e))
