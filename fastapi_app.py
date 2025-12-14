import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psutil
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入RAG系统模块
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag_afsim_system import AFSIMRAGSystem
from utils import ConfigManager

# Pydantic模型
class InitRequest(BaseModel):
    model_path: Optional[str] = None
    embedding_model: Optional[str] = None
    chroma_db_path: Optional[str] = None

class LoadDocsRequest(BaseModel):
    file_list_path: str
    base_dir: Optional[str] = "."

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ClearRequest(BaseModel):
    conversation_id: Optional[str] = None

# 全局状态
system: Optional[AFSIMRAGSystem] = None
is_loading = False
conversations: Dict[str, List[tuple]] = {}  # 存储对话历史

app = FastAPI(
    title="AFSIM RAG代码生成系统API",
    description="基于Qwen3 + BGE嵌入 + Chroma的AFSIM智能助手API",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = os.path.join(ROOT_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# API路由
@app.post("/api/init")
async def initialize_system(request: Optional[InitRequest] = None):
    """初始化RAG系统"""
    global system, is_loading
    
    if is_loading:
        raise HTTPException(status_code=503, detail="系统正在加载中，请稍候...")
    
    is_loading = True
    try:
        config = ConfigManager()
        
        # 使用请求参数或配置
        model_path = request.model_path if request else config.get('model.path')
        embed_model = request.embedding_model if request else config.get('embedding.model_name')
        chroma_path = request.chroma_db_path if request else config.get('database.chroma_path')
        
        logger.info(f"初始化系统，模型路径: {model_path}")
        logger.info(f"嵌入模型: {embed_model}")
        
        system = AFSIMRAGSystem(
            model_path=model_path,
            embedding_model=embed_model,
            chroma_db_path=chroma_path
        )
        
        doc_count = system.collection.count()
        return {
            "success": True,
            "message": "系统初始化成功",
            "model": os.path.basename(model_path) if model_path else "未知",
            "embed_model": os.path.basename(embed_model) if embed_model else "未知",
            "doc_count": doc_count,
            "device": str(system.model.device) if hasattr(system, 'model') and hasattr(system.model, 'device') else '未知'
        }
        
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")
    finally:
        is_loading = False

@app.post("/api/load-docs")
async def load_documents(request: LoadDocsRequest):
    """加载文档"""
    global system, is_loading
    
    if is_loading:
        raise HTTPException(status_code=503, detail="系统正在加载中，请稍候...")
    
    if system is None:
        raise HTTPException(status_code=400, detail="请先初始化系统")
    
    is_loading = True
    try:
        if not os.path.exists(request.file_list_path):
            raise HTTPException(status_code=404, detail=f"路径不存在: {request.file_list_path}")
        
        # 判断是文件夹还是文件
        if os.path.isdir(request.file_list_path):
            success = system.load_documents_from_folder(request.file_list_path)
            method = "文件夹"
        elif os.path.isfile(request.file_list_path):
            success = system.load_documents_from_list(request.file_list_path, request.base_dir)
            method = "文件列表"
        else:
            raise HTTPException(status_code=400, detail=f"路径无效: {request.file_list_path}")
        
        if success:
            doc_count = system.collection.count()
            return {
                "success": True,
                "message": "文档加载完成",
                "method": method,
                "total_chunks": doc_count,
                "load_time": datetime.now().strftime('%H:%M:%S')
            }
        else:
            raise HTTPException(status_code=500, detail="文档加载失败")
            
    except Exception as e:
        logger.error(f"文档加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档加载失败: {str(e)}")
    finally:
        is_loading = False

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """处理聊天查询"""
    global system
    
    if system is None:
        raise HTTPException(status_code=400, detail="请先初始化系统")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="请输入问题")
    
    try:
        logger.info(f"处理查询: {request.query[:100]}...")
        
        # 获取或创建对话历史
        conv_id = request.conversation_id or "default"
        if conv_id not in conversations:
            conversations[conv_id] = []
        
        # 添加用户消息
        conversations[conv_id].append((request.query, "正在思考..."))
        
        # 生成回答
        result = system.generate_response(request.query)
        
        # 更新最后一条记录
        if conversations[conv_id]:
            conversations[conv_id][-1] = (request.query, result["response"])
        
        return {
            "success": True,
            "query": request.query,
            "response": result["response"],
            "sources": result["sources"],
            "conversation_id": conv_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"生成回答时出错: {e}")
        raise HTTPException(status_code=500, detail=f"生成回答时出错: {str(e)}")

@app.post("/api/clear")
async def clear_chat(request: ClearRequest):
    """清空对话历史"""
    conv_id = request.conversation_id if request else "default"
    if conv_id in conversations:
        conversations[conv_id] = []
    
    return {
        "success": True,
        "message": "对话已清空",
        "conversation_id": conv_id
    }

@app.get("/api/export")
async def export_chat(conversation_id: Optional[str] = None):
    """导出对话历史"""
    conv_id = conversation_id or "default"
    history = conversations.get(conv_id, [])
    
    if not history:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "没有对话历史可导出"}
        )
    
    try:
        export_data = {
            "export_time": datetime.now().isoformat(),
            "conversation_id": conv_id,
            "total_conversations": len(history),
            "conversations": [
                {
                    "question": q,
                    "answer": a[:1000] + "..." if len(a) > 1000 else a,
                    "answer_length": len(a),
                    "timestamp": datetime.now().isoformat()
                }
                for q, a in history
            ]
        }
        
        # 保存到文件
        export_dir = os.path.join(ROOT_DIR, "exports")
        os.makedirs(export_dir, exist_ok=True)
        filename = f"afsim_chat_history_{conv_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "message": f"对话历史已导出到: {filepath}",
            "file_path": filepath,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"导出失败: {e}")
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")

@app.get("/api/system-info")
async def get_system_info():
    """获取系统信息"""
    global system
    
    try:
        config = ConfigManager()
        
        info = {
            "success": True,
            "model": {
                "path": config.get('model.path', '未设置'),
                "embed_model": config.get('embedding.model_name', '未设置'),
                "vector_db": config.get('database.chroma_path', './chroma_db')
            },
            "document": {
                "count": system.collection.count() if system else 0
            },
            "system": {
                "web_port": config.get_int('web.port', 7860),
                "debug_mode": config.get_bool('web.debug', True),
                "log_level": config.get('logging.level', 'INFO'),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return info
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")

@app.get("/api/debug-info")
async def get_debug_info():
    """获取调试信息"""
    global system
    
    debug_info = {
        "success": True,
        "system_initialized": system is not None,
        "collection": {
            "name": system.collection.name if system and hasattr(system, 'collection') else None,
            "count": system.collection.count() if system and hasattr(system, 'collection') else 0
        },
        "model": {
            "device": str(system.model.device) if system and hasattr(system, 'model') and hasattr(system.model, 'device') else None
        }
    }
    
    return debug_info

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """主页"""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read(), status_code=200)
    else:
        return HTMLResponse(content="错误: index.html 文件不存在", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)