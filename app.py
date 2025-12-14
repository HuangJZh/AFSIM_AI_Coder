import gradio as gr
import json
import os
import logging
from datetime import datetime
from pathlib import Path
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 延迟导入以避免循环依赖
system = None
is_loading = False

def initialize_system():
    """初始化RAG系统"""
    global system, is_loading
    if is_loading:
        return "系统正在加载中，请稍候...", "系统正在加载中..."
    
    is_loading = True
    try:
        from rag_afsim_system import AFSIMRAGSystem
        from utils import ConfigManager
        
        config = ConfigManager()
        
        # 显示配置信息
        model_path = config.get('model.path', '未设置')
        embed_model = config.get('embedding.model_name', '未设置')
        
        logger.info(f"初始化系统，模型路径: {model_path}")
        logger.info(f"嵌入模型: {embed_model}")
        
        system = AFSIMRAGSystem()
        
        # 获取系统信息
        doc_count = system.collection.count()
        info = f"✅ 系统初始化成功！\n"
        info += f"• 模型: {os.path.basename(model_path)}\n"
        info += f"• 嵌入模型: {os.path.basename(embed_model)}\n"
        info += f"• 文档数量: {doc_count}\n"
        info += f"• 设备: {system.model.device if hasattr(system, 'model') else '未知'}"
        
        is_loading = False
        return info, info
        
    except Exception as e:
        is_loading = False
        error_msg = f"❌ 初始化失败: {str(e)}"
        logger.error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return error_msg, error_msg

def load_documents_from_folder(file_list_path):
    """加载文档"""
    global system, is_loading
    if is_loading:
        return "系统正在加载中，请稍候...", "系统正在加载中..."
    
    if system is None:
        return "请先初始化系统", "系统未初始化"
    
    is_loading = True
    try:
        # 检查路径是否存在
        if not os.path.exists(file_list_path):
            is_loading = False
            return f"❌ 路径不存在: {file_list_path}", f"路径不存在: {file_list_path}"
        
        # 显示加载信息
        info = f"正在从 {file_list_path} 加载文档...\n"
        
        # 检查路径是文件还是文件夹
        if os.path.isdir(file_list_path):
            # 如果是文件夹，使用 load_documents_from_folder
            logger.info(f"从文件夹加载文档: {file_list_path}")
            success = system.load_documents_from_folder(file_list_path)
            method = "文件夹"
        elif os.path.isfile(file_list_path):
            # 如果是文件，使用 load_documents_from_list
            logger.info(f"从文件列表加载文档: {file_list_path}")
            success = system.load_documents_from_list(file_list_path, base_dir=".")
            method = "文件列表"
        else:
            is_loading = False
            return f"❌ 路径无效: {file_list_path}", f"路径无效: {file_list_path}"
        
        if success:
            doc_count = system.collection.count()
            result = f"✅ 文档加载完成！\n"
            result += f"• 加载方式: {method}\n"
            result += f"• 文档块总数: {doc_count}\n"
            result += f"• 加载时间: {datetime.now().strftime('%H:%M:%S')}"
            is_loading = False
            return result, result
        else:
            error_msg = "❌ 文档加载失败，请检查日志"
            is_loading = False
            return error_msg, error_msg
            
    except Exception as e:
        is_loading = False
        error_msg = f"❌ 文档加载失败: {str(e)}"
        logger.error(f"文档加载失败: {e}")
        import traceback
        traceback.print_exc()
        return error_msg, error_msg

def query_afsim(query, history=None):
    """处理查询"""
    global system, is_loading
    if is_loading:
        return "系统正在加载中，请稍候...", history or []
    
    if system is None:
        return "请先初始化系统", []
    
    if not query.strip():
        return "请输入问题", history or []
    
    try:
        logger.info(f"处理查询: {query[:100]}...")
        
        # 显示处理中状态
        if history is None:
            history = []
        
        # 添加用户消息到历史
        history.append((query, "正在思考..."))
        
        # 生成回答
        result = system.generate_response(query)
        
        # 格式化显示
        response = result["response"]
        
        # 添加来源信息
        if result['sources']:
            response += "\n\n**📚 参考来源:**\n"
            for i, source in enumerate(result['sources'][:5], 1):
                response += f"{i}. {source}\n"
            if len(result['sources']) > 5:
                response += f"... 还有 {len(result['sources']) - 5} 个来源\n"
        else:
            response += "\n\n**⚠ 注意:** 未找到相关参考文档，回答基于模型知识生成。"
        
        # 更新最后一条历史记录
        if history and history[-1][0] == query:
            history[-1] = (query, response)
        else:
            history.append((query, response))
        
        logger.info(f"查询完成，响应长度: {len(response)} 字符")
        
        return "", history
        
    except Exception as e:
        error_msg = f"生成回答时出错: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        
        # 更新历史记录
        if history and history[-1][0] == query:
            history[-1] = (query, f"❌ {error_msg}")
        else:
            history.append((query, f"❌ {error_msg}"))
        
        return "", history

def clear_chat():
    """清空聊天"""
    return [], "", "对话已清空"

def export_chat(history):
    """导出对话历史为JSON"""
    if not history:
        return "没有对话历史可导出"
    
    try:
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_conversations": len(history),
            "conversations": []
        }
        
        for q, a in history:
            export_data["conversations"].append({
                "question": q,
                "answer": a[:1000] + "..." if len(a) > 1000 else a,
                "answer_length": len(a),
                "timestamp": datetime.now().isoformat()
            })
        
        # 保存到文件
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        filename = f"afsim_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return f"✅ 对话历史已导出到: {filepath}"
        
    except Exception as e:
        logger.error(f"导出失败: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ 导出失败: {str(e)}"

def get_system_info():
    """获取系统信息"""
    global system
    try:
        from utils import ConfigManager
        config = ConfigManager()
        
        info = "## 系统信息\n\n"
        
        # 模型信息
        model_path = config.get('model.path', '未设置')
        info += f"**模型配置:**\n"
        info += f"- 主模型: {os.path.basename(model_path)}\n"
        info += f"- 嵌入模型: {config.get('embedding.model_name', '未设置')}\n"
        info += f"- 向量数据库: {config.get('database.chroma_path', './chroma_db')}\n\n"
        
        # 文档信息
        if system is not None:
            doc_count = system.collection.count()
            info += f"**文档状态:**\n"
            info += f"- 文档块数量: {doc_count}\n"
        else:
            info += "**文档状态:** 系统未初始化\n\n"
        
        # 系统配置
        info += f"**系统配置:**\n"
        info += f"- Web端口: {config.get_int('web.port', 7860)}\n"
        info += f"- 调试模式: {'开启' if config.get_bool('web.debug', True) else '关闭'}\n"
        info += f"- 日志级别: {config.get('logging.level', 'INFO')}\n"
        
        return info
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        return f"获取系统信息失败: {str(e)}"

# 创建Gradio界面
with gr.Blocks(title="AFSIM RAG代码生成系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 AFSIM RAG增强代码生成系统")
    gr.Markdown("基于Qwen3 + BGE嵌入 + Chroma的AFSIM智能助手")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ 系统控制")
            
            init_btn = gr.Button("🚀 初始化系统", variant="primary", size="sm")
            init_status = gr.Markdown("等待初始化...")
            
            # 文档加载部分
            with gr.Group():
                gr.Markdown("### 📁 文档加载")
                default_docs_path = "tutorials"
                file_input = gr.Textbox(
                    label="文档路径",
                    value=default_docs_path,
                    placeholder="输入文档文件夹路径或文件列表路径"
                )
                load_btn = gr.Button("📂 加载文档", variant="secondary", size="sm")
                load_status = gr.Markdown("")
            
            # 示例查询
            gr.Markdown("### 💡 示例查询")
            examples = [
                "请定义一个蓝方的坦克平台类型",
                "编写一段代码，仅用于设置仿真的结束时间为1200秒",
                "生成一个武器系统控制的示例代码",
                "如何可视化仿真结果？",
                "定义一个蓝方导弹发射车"
            ]
            
            example_selector = gr.Examples(
                examples=examples,
                inputs=[gr.Textbox(visible=False)],
                label="点击示例快速提问"
            )
            
            # 系统信息显示
            gr.Markdown("### 📊 系统信息")
            info_display = gr.Markdown(get_system_info())
            
            # 刷新系统信息按钮
            refresh_btn = gr.Button("🔄 刷新信息", variant="secondary", size="sm")
            
        with gr.Column(scale=3):
            gr.Markdown("## 💬 AFSIM助手")
            
            chatbot = gr.Chatbot(
                label="对话历史",
                height=500,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入你的AFSIM相关问题",
                    placeholder="例如：如何创建AFSIM移动平台？",
                    scale=4,
                    lines=2,
                    max_lines=5
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("清空对话", variant="secondary", scale=1)
                export_btn = gr.Button("导出历史", variant="secondary", scale=1)
                debug_btn = gr.Button("调试信息", variant="secondary", scale=1)
    
    # 事件绑定
    def on_init():
        return initialize_system()
    
    init_btn.click(
        fn=on_init,
        outputs=[init_status, info_display]
    )
    
    def on_load(file_path):
        return load_documents_from_folder(file_path)
    
    load_btn.click(
        fn=on_load,
        inputs=file_input,
        outputs=[load_status, info_display]
    )
    
    def on_clear():
        return clear_chat()
    
    clear_btn.click(
        fn=on_clear,
        outputs=[chatbot, msg, info_display]
    )
    
    def on_export(history):
        return export_chat(history)
    
    export_btn.click(
        fn=on_export,
        inputs=chatbot,
        outputs=info_display
    )
    
    def on_refresh():
        return get_system_info()
    
    refresh_btn.click(
        fn=on_refresh,
        outputs=info_display
    )
    
    def on_debug():
        """显示调试信息"""
        global system
        debug_info = "## 调试信息\n\n"
        
        if system is None:
            debug_info += "系统未初始化\n"
        else:
            debug_info += f"系统已初始化\n"
            debug_info += f"- 集合名称: {system.collection.name if hasattr(system, 'collection') else 'N/A'}\n"
            debug_info += f"- 文档数量: {system.collection.count() if hasattr(system, 'collection') else 'N/A'}\n"
            debug_info += f"- 模型设备: {system.model.device if hasattr(system, 'model') else 'N/A'}\n"
            
        return debug_info
    
    debug_btn.click(
        fn=on_debug,
        outputs=info_display
    )
    
    # 提交查询
    def process_query(message, history):
        if not message.strip():
            return "", history, "请输入问题"
        
        # 清空输入框
        new_history = history.copy() if history else []
        
        # 处理查询
        _, updated_history = query_afsim(message, new_history)
        
        # 获取最新回答
        latest_response = updated_history[-1][1] if updated_history else ""
        status = f"已回答: {message[:30]}..." if len(message) > 30 else f"已回答: {message}"
        
        return "", updated_history, status
    
    submit_btn.click(
        fn=process_query,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, info_display]
    )
    
    # 回车提交
    msg.submit(
        fn=process_query,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, info_display]
    )

def launch_app(share=None, port=None):
    """启动函数"""
    from utils import ConfigManager
    
    config = ConfigManager()
    
    # 获取配置
    if share is None:
        share = config.get_bool('web.share', False)
    if port is None:
        port = config.get_int('web.port', 7860)
    
    debug = config.get_bool('web.debug', True)
    
    try:
        logger.info(f"启动Web界面，端口: {port}, 分享: {share}")
        
        # 启用队列以提高性能
        demo.queue(max_size=20)
        
        # 启动应用
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False,
            show_api=False
        )
        
    except KeyboardInterrupt:
        logger.info("服务器已正常关闭")
        return
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    launch_app()