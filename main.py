import os
import json
import re
import requests
from typing import List, Dict
from dotenv import load_dotenv
from rag_utils import RagUtils

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "./tutorials")

def llm_call(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """通用 LLM 调用函数"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_tokens": 4000
    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[API Error] {response.text}")
            return ""
    except Exception as e:
        print(f"[Exception] {e}")
        return ""

def format_history(history: List[Dict[str, str]]) -> str:
    """
    将对话历史列表转换为字符串。
    【修改】为了支持代码修改，这里不再截断内容，保留完整代码和换行。
    """
    if not history:
        return "无历史对话。"
    
    formatted = []
    for msg in history:
        role = "用户" if msg['role'] == 'user' else "助手(已生成的代码)"
        # 直接使用完整内容，不截断
        formatted.append(f"=== {role} ===\n{msg['content']}\n")
    return "\n".join(formatted)

def step1_generate_plan(user_query: str, history: List[Dict[str, str]]) -> List[str]:
    """Step 1: 结合历史，生成文件列表计划"""
    print(">>> [Step 1] 正在规划项目结构...")
    
    retrieve_query = f"AFSIM project structure file organization include_once {user_query}"
    chunks = RagUtils.retrieve(retrieve_query, top_k=5)
    reranked = RagUtils.rerank(retrieve_query, chunks, top_k=3)
    context = "\n\n".join(reranked)
    
    history_str = format_history(history)

    prompt = f"""你是一个 AFSIM 仿真架构师。
基于用户最新需求和对话历史，列出构建该仿真所需的标准文件列表。

【对话历史】
{history_str}

【当前用户需求】
{user_query}

【参考规范】
{context}

请仅返回一个纯 JSON 字符串列表，不要包含 Markdown 格式或其他文字。
例如: ["floridistan.txt", "setup.txt", "scenarios/blue_laydown.txt", "scenarios/red_laydown.txt"]
"""
    response = llm_call([{"role": "user", "content": prompt}], temperature=0.1)
    
    try:
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end != -1:
            json_str = response[start:end]
            file_list = json.loads(json_str)
            print(f"    计划生成文件: {file_list}")
            return file_list
    except:
        print("    [Warn] 解析计划失败，使用默认结构")
    
    return ["floridistan.txt", "setup.txt", "scenarios/blue_laydown.txt", "scenarios/red_laydown.txt"]

def step2_generate_file_content(filename: str, user_query: str, file_list: List[str], global_context: str, history: List[Dict[str, str]]) -> str:
    """Step 2: 结合历史，生成具体文件内容"""
    print(f">>> [Generating] 正在生成 {filename} ...")

    search_keyword = ""
    if "setup" in filename:
        search_keyword = "platform_type definitions sensor weapon mover processor"
    elif "laydown" in filename:
        search_keyword = "platform instance position route heading track"
    else: 
        search_keyword = "event_pipe log_file include_once end_time"

    full_query = f"{user_query} {search_keyword} syntax for {filename}"
    
    chunks = RagUtils.retrieve(full_query, top_k=15) 
    best_chunks = RagUtils.rerank(full_query, chunks, top_k=5) 
    doc_context = "\n\n".join(best_chunks)
    
    history_str = format_history(history)

    system_prompt = """你是一个精通 AFSIM 代码生成专家。
你的任务是根据用户的需求，参考提供的知识库规则，生成正确、规范的 AFSIM 代码脚本。
请遵循以下规则：
0. **只能使用知识库中提供的语法和模式**。
1. 在每个文件的开头用注释说明文件名。
2. 保持代码简洁，逻辑清晰。
3. 不需要解释代码，只需提供代码本身。
"""

    user_prompt = f"""
【对话历史（包含之前生成的完整代码）】
{history_str}

【当前任务目标】
编写或修改文件: {filename}
(如果历史中包含该文件的代码且需要修改，请基于历史代码进行更新；如果是新文件则重新生成)

【用户最新需求】
{user_query}

【项目文件结构】
{json.dumps(file_list)}

【本轮生成上下文】
{global_context}

【参考语法/文档】
{doc_context}

请编写 {filename} 的完整代码：
"""
    return llm_call([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

def main():
    if not API_KEY:
        print("请配置 API_KEY")
        return

    # 初始化索引
    RagUtils.load_and_index_directory(KNOWLEDGE_DIR)

    print("\n" + "="*50)
    print("AFSIM 代码生成助手")
    print("输入 'quit' 退出, 'reset' 清空历史")
    print("="*50 + "\n")

    # 对话历史列表
    conversation_history = []

    while True:
        try:
            print(f"\n需求 (历史 {len(conversation_history)//2} 轮) > (输入 //end 结束)")
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == "//end": break
                    lines.append(line)
                except EOFError: break
            
            user_input = "\n".join(lines).strip()
            
            # --- 命令处理 ---
            if user_input.lower() in ["quit", "exit"]: break
            if user_input.lower() in ["clear", "reset"]:
                conversation_history = []
                print("[系统] 历史已清空")
                continue
            if not user_input: continue

            # --- 阶段 1: 规划 ---
            file_plan = step1_generate_plan(user_input, conversation_history)
            
            # --- 阶段 2: 逐个生成 ---
            global_context_summary = "" 
            full_code_log = "代码如下：\n\n" # 用于存入历史

            # 排序
            sorted_files = sorted(file_plan, key=lambda x: (
                0 if "setup" in x else 
                1 if "laydown" in x else 
                2
            ))

            for filename in sorted_files:
                code_content = step2_generate_file_content(filename, user_input, file_plan, global_context_summary, conversation_history)
                
                # 正则修复
                code_content = re.sub(r'^```\w*\n', '', code_content)
                code_content = re.sub(r'\n```$', '', code_content)
                
                print(f"\n--- FILE: {filename} ---")
                print(code_content)
                print("-" * 30)

                # 收集上下文
                if "setup" in filename:
                    types = re.findall(r'platform_type\s+(\w+)', code_content)
                    global_context_summary += f"\n在 {filename} 中定义了平台类型: {', '.join(types)}"
                
                # 【修改】将完整代码追加到记录字符串中
                full_code_log += f"File: {filename}\n```afsim\n{code_content}\n```\n\n"

            print("\n=== 生成完成 ===")
            
            # --- 更新历史 ---
            # 1. 记录用户输入
            conversation_history.append({"role": "user", "content": user_input})
            
            # 2. 【修改】记录全量代码回复
            conversation_history.append({"role": "assistant", "content": full_code_log})

            # 【修改】Token 保护策略：因为代码很长，我们只保留最近 6 条消息（即最近 3 轮对话）
            # 这样保证 AI 至少能看到上一轮生成的完整代码来进行修改
            if len(conversation_history) > 6:
                conversation_history = conversation_history[-6:]

        except KeyboardInterrupt:
            print("\n程序已停止。")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()