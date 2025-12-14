#!/usr/bin/env python3
import argparse
import uvicorn
import os

def main():
    parser = argparse.ArgumentParser(description="启动AFSIM RAG FastAPI服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    print(f"启动FastAPI服务器: http://{args.host}:{args.port}")
    print("API文档: http://localhost:8000/docs")
    print("Web界面: http://localhost:8000")
    
    uvicorn.run(
        "fastapi_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )

if __name__ == "__main__":
    main()