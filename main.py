#!/usr/bin/env python3
"""
AFSIM RAG系统主启动脚本（兼容Gradio 3.x）
"""
import argparse
import sys
import os
import signal
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AFSIM RAG代码生成系统")
    parser.add_argument("--mode", choices=["cli", "web", "index", "test", "interactive"], 
                       default="web", help="运行模式")
    parser.add_argument("--query", type=str, help="CLI模式下的查询")
    parser.add_argument("--docs", type=str, default=None,
                       help="文档列表文件路径或文件夹路径")
    parser.add_argument("--db", type=str, default=None,
                       help="Chroma数据库路径")
    parser.add_argument("--port", type=int, default=None,
                       help="Web服务器端口")
    parser.add_argument("--share", action="store_true",
                       help="是否生成公网分享链接")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 设置环境变量以便ConfigManager使用
    if args.config and os.path.exists(args.config):
        os.environ['AFSIM_CONFIG_PATH'] = args.config
    
    # 导入必要的模块
    try:
        from utils import ConfigManager
        from rag_afsim_system import AFSIMRAGSystem
    except ImportError as e:
        print(f"导入模块失败: {e}")
        print("请确保所有依赖模块在同一个目录下")
        sys.exit(1)
    
    # 获取配置
    config = ConfigManager()
    
    # 使用配置或命令行参数
    docs_path = args.docs or config.get('paths.tutorials_folder', 'tutorials')
    db_path = args.db or config.get('database.chroma_path', './chroma_db')
    port = args.port or config.get_int('web.port', 7860)
    
    if args.mode == "index":
        print("开始索引文档...")
        try:
            system = AFSIMRAGSystem(chroma_db_path=db_path)
            success = system.load_documents_from_folder(docs_path)
            if success:
                print("✅ 文档索引完成！")
                sys.exit(0)
            else:
                print("❌ 文档索引失败")
                sys.exit(1)
        except Exception as e:
            print(f"❌ 索引过程中出错: {e}")
            sys.exit(1)
    
    elif args.mode == "cli":
        try:
            system = AFSIMRAGSystem(chroma_db_path=db_path)
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            sys.exit(1)
        
        if args.query:
            print(f"查询: {args.query}")
            result = system.generate_response(args.query)
            print("\n" + "="*60)
            print("回答：")
            print(result["response"])
            print("\n来源：")
            for source in result["sources"][:5]:
                print(f"  - {source}")
            if len(result["sources"]) > 5:
                print(f"  - ... 还有 {len(result['sources']) - 5} 个来源")
            print("="*60)
        else:
            print("交互式CLI模式（输入'exit'退出）")
            print("注意：首次运行需要先加载文档，输入 'load' 加载文档")
            
            # 检查文档是否已加载
            doc_count = system.collection.count()
            if doc_count == 0:
                print(f"⚠ 数据库中有 {doc_count} 个文档，建议先加载文档")
            
            while True:
                try:
                    query = input("\n请输入问题: ").strip()
                    if query.lower() in ["exit", "quit", "q"]:
                        print("再见！")
                        break
                    elif query.lower() == "load":
                        print("正在加载文档...")
                        success = system.load_documents_from_folder(docs_path)
                        if success:
                            print("✅ 文档加载完成！")
                        else:
                            print("❌ 文档加载失败")
                        continue
                    elif not query:
                        continue
                    
                    result = system.generate_response(query)
                    print("\n" + "="*60)
                    print(result["response"])
                    print("-"*40)
                    if result["sources"]:
                        print("参考来源:")
                        for source in result["sources"][:5]:
                            print(f"  • {source}")
                        if len(result["sources"]) > 5:
                            print(f"  • ... 还有 {len(result['sources']) - 5} 个来源")
                    else:
                        print("⚠ 未找到相关参考来源")
                    print("="*60)
                    
                except KeyboardInterrupt:
                    print("\n程序已中断")
                    break
                except Exception as e:
                    print(f"错误: {e}")
    
    elif args.mode == "interactive":
        try:
            system = AFSIMRAGSystem(chroma_db_path=db_path)
            system.interactive_chat()
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            sys.exit(1)
    
    elif args.mode == "web":
        print(f"启动Web界面... 访问 http://localhost:{port}")
        print("按 Ctrl+C 停止服务器")
        
        try:
            from app import launch_app
            
            def signal_handler(sig, frame):
                print("\n正在关闭服务器...")
                sys.exit(0)
            
            # 设置信号处理器
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, signal_handler)
            
            # 启动应用
            launch_app(share=args.share, port=port)
            
        except KeyboardInterrupt:
            print("\n服务器已关闭")
            sys.exit(0)
        except Exception as e:
            print(f"启动失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.mode == "test":
        print("运行系统测试...")
        test_system()

def test_system():
    """测试系统功能"""
    from rag_afsim_system import AFSIMRAGSystem
    from utils import ConfigManager
    
    config = ConfigManager()
    
    print("1. 初始化系统...")
    try:
        system = AFSIMRAGSystem(
            chroma_db_path="./chroma_db_test"
        )
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    print("2. 加载文档...")
    docs_path = config.get('paths.tutorials_folder', 'tutorials')
    if os.path.exists(docs_path):
        success = system.load_documents_from_folder(docs_path)
        if not success:
            print("⚠ 文档加载失败，继续测试...")
    else:
        print(f"⚠ 文档路径不存在: {docs_path}")
    
    print("3. 测试查询...")
    test_queries = [
        "什么是AFSIM?",
        "如何创建移动平台?",
        "生成一个简单的仿真示例"
    ]
    
    for query in test_queries:
        print(f"\n测试查询: {query}")
        try:
            result = system.generate_response(query)
            print(f"响应长度: {len(result['response'])} 字符")
            print(f"参考来源数量: {len(result['sources'])}")
            print("-"*50)
        except Exception as e:
            print(f"❌ 查询失败: {e}")
    
    print("测试完成！")

if __name__ == "__main__":
    main()