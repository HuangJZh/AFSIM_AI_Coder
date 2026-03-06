import os
from pathlib import Path
from typing import TypedDict, List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
import json


# ==========================================
# 0. 环境准备与双模型初始化 
# ==========================================
load_dotenv()

# 【关键新增】强制绕过系统代理，防止请求被 Clash/VPN 拦截导致 502
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# 1. 智慧中枢：DeepSeek 模型 (用于解析、架构规划、校验、修复)
deepseek_llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com/v1",
    max_tokens=1024,
    temperature=0.1
)

# 2. 基础干活节点：本地微调小模型
local_llm = ChatOpenAI(
    model="Qwen2.5-AFSIM-Base-SFT", 
    api_key="LOCAL_DUMMY_KEY",     
    # 【关键修改】将 localhost 换成 127.0.0.1
    base_url="http://127.0.0.1:8000/v1", 
    max_tokens=1024,
    temperature=0.1
)

# ==========================================
# 1. 本地知识库加载 (仅保留给高级节点使用)
# ==========================================
TUTORIALS_DIR = Path("tutorials")
def load_tutorial(filename: str) -> str:
    filepath = TUTORIALS_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# 小模型由于注意力机制限制，不再加载 RAG 教程。
# 仅为 DeepSeek 保留错误排查知识库
KNOWLEDGE_ERRORS = load_tutorial("afsim_error_collection.md")

# ==========================================
# 2. 动态数据结构定义 (Pydantic Schema)
# ==========================================
class RoutePoint(BaseModel):
    latitude: str = Field(description="纬度，例如：24:26:35.300n")
    longitude: str = Field(description="经度，例如：120:09:56.232e")
    altitude: Optional[str] = Field(None, description="高度")

class Platform(BaseModel):
    name: str = Field(description="平台标识")
    platform_type: str = Field(description="平台类型")
    initial_location: str = Field(description="初始位置描述")
    speed: Optional[str] = Field(None, description="速度描述")
    route: Optional[List[RoutePoint]] = Field(None, description="航行路线")
    other_attributes: Optional[Dict[str, Any]] = Field(None, description="其他属性")

class Faction(BaseModel):
    faction_name: str = Field(description="阵营名称的英文缩写")
    platforms: List[Platform] = Field(description="该阵营平台实体")

class ScenarioStructure(BaseModel):
    factions: List[Faction] = Field(description="场景阵营列表")
    weapons: List[Dict[str, Any]] = Field(description="武器及效能参数")
    simulation_events: List[Dict[str, Any]] = Field(description="交战事件")

class AfsimProjectFile(BaseModel):
    filepath: str = Field(description="相对路径，如 'setup.txt', 'platforms/x.txt', 'main.txt'")
    content: str = Field(description="AFSIM 脚本内容")

class AfsimProjectFiles(BaseModel):
    files: List[AfsimProjectFile] = Field(description="项目文件列表")

# ==========================================
# 3. 定义全局状态 (State)
# ==========================================
class AgentState(TypedDict):
    original_prompt: str
    scenario_json: str
    platform_scripts: str
    weapon_scripts: str
    control_scripts: str
    project_files: List[dict]
    errors: List[str]
    revision_count: int

# ==========================================
# 4. 工作流节点定义 (模型路由与分发)
# ==========================================

def scenario_parser(state: AgentState):
    """节点A：需求解析 (使用 DeepSeek)"""
    print("--- [DeepSeek] 提取场景结构 ---")
    parser = PydanticOutputParser(pydantic_object=ScenarioStructure)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 AFSIM 数据提取专家。提取阵营、平台航线等。\n{format_instructions}"),
        ("user", "{prompt}")
    ])
    chain = prompt | deepseek_llm | parser
    try:
        parsed_scenario = chain.invoke({
            "prompt": state["original_prompt"],
            "format_instructions": parser.get_format_instructions()
        })
        return {"scenario_json": parsed_scenario.model_dump_json()}
    except Exception as e:
        state["errors"].append(f"解析失败: {str(e)}")
        return {"scenario_json": "{}"}

def weapon_builder(state: AgentState):
    """节点B1：武器生成 (任务拆解版)"""
    print(f"--- [本地模型] 开始逐条生成武器代码 ---")
    
    # 1. 将大 JSON 解析回 Python 字典
    try:
        scenario_dict = json.loads(state["scenario_json"])
    except json.JSONDecodeError:
        return {"weapon_scripts": "// 武器 JSON 解析失败"}
        
    weapons_code = []
    
    # 2. 定义专门针对“单条任务”的极简提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 AFSIM 脚本助手。请根据下方提供的单一武器参数，生成该武器的 AFSIM 定义代码。\n【严格要求】只输出纯代码，绝对不要任何解释，不要使用 ``` 这种 markdown 代码块符号。"),
        ("user", "武器参数：\n{single_item}")
    ])
    chain = prompt | local_llm
    
    # 3. 循环遍历，每次只让小模型写一个武器
    weapons = scenario_dict.get("weapons", [])
    if not weapons:
        print("  -> 未检测到武器需求。")
        return {"weapon_scripts": ""}
        
    for i, weapon in enumerate(weapons):
        print(f"  -> 正在生成第 {i+1}/{len(weapons)} 个武器...")
        item_str = json.dumps(weapon, ensure_ascii=False)
        response = chain.invoke({"single_item": item_str})
        weapons_code.append(response.content.strip())
        
    return {"weapon_scripts": "\n\n".join(weapons_code)}

def platform_builder(state: AgentState):
    """节点B2：平台生成 (任务拆解版)"""
    print("--- [本地模型] 开始逐条生成平台与航线代码 ---")
    try:
        scenario_dict = json.loads(state["scenario_json"])
    except json.JSONDecodeError:
        return {"platform_scripts": "// 平台 JSON 解析失败"}
        
    platforms_code = []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 AFSIM 脚本助手。请根据下方提供的单一平台及航线参数，生成对应的 AFSIM 实体和路线定义代码。\n【严格要求】只输出纯代码，绝对不要任何解释，不要使用 ``` 这种 markdown 代码块符号。"),
        ("user", "所属阵营：{faction_name}\n平台参数：\n{single_item}")
    ])
    chain = prompt | local_llm
    
    # 嵌套循环：遍历阵营 -> 遍历该阵营的平台
    for faction in scenario_dict.get("factions", []):
        faction_name = faction.get("faction_name", "Unknown")
        platforms = faction.get("platforms", [])
        
        for i, platform in enumerate(platforms):
            print(f"  -> 正在生成 [{faction_name}] 阵营的第 {i+1}/{len(platforms)} 个平台...")
            item_str = json.dumps(platform, ensure_ascii=False)
            response = chain.invoke({
                "faction_name": faction_name, 
                "single_item": item_str
            })
            platforms_code.append(response.content.strip())
            
    return {"platform_scripts": "\n\n".join(platforms_code)}

def event_builder(state: AgentState):
    """节点B3：事件生成 (使用 本地小模型, 移除 RAG)"""
    print("--- [本地模型] 生成控制事件代码 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 AFSIM 脚本助手。根据 JSON 生成 execute 块和时间控制。只需输出代码。"),
        ("user", "{json_data}")
    ])
    chain = prompt | local_llm
    response = chain.invoke({"json_data": state["scenario_json"]})
    return {"control_scripts": response.content}

def project_architect(state: AgentState):
    """节点C：项目多文件汇编 (使用 DeepSeek)"""
    print("--- [DeepSeek] 统筹项目文件架构 ---")
    parser = PydanticOutputParser(pydantic_object=AfsimProjectFiles)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是 AFSIM 架构师。重组代码片段为多文件项目(setup.txt, scenarios/laydown.txt, main.txt)。
        要求主文件按顺序 include_once 子文件。
        {format_instructions}"""),
        ("user", "武器:\n{weapons}\n\n平台:\n{platforms}\n\n控制:\n{controls}")
    ])
    chain = prompt | deepseek_llm | parser
    try:
        project_obj = chain.invoke({
            "weapons": state.get('weapon_scripts', ''),
            "platforms": state.get('platform_scripts', ''),
            "controls": state.get('control_scripts', ''),
            "format_instructions": parser.get_format_instructions()
        })
        return {"project_files": [f.model_dump() for f in project_obj.files]}
    except Exception as e:
        return {"project_files": []}

def syntax_validator(state: AgentState):
    """节点D：跨文件校验 (使用 DeepSeek + RAG)"""
    print("--- [DeepSeek] 执行多文件语法校验 ---")
    project_text = "\n".join([f"==== {f['filepath']} ====\n{f['content']}\n" for f in state.get("project_files", [])])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 AFSIM 语法审计员。检查以下代码。\n【常见错误】\n{knowledge}\n完全无误回复 PASS，有错指出错误。"),
        ("user", "{code}")
    ])
    response = deepseek_llm.invoke(prompt.format_messages(code=project_text, knowledge=KNOWLEDGE_ERRORS))
    feedback = response.content.strip()
    
    errors = [line for line in feedback.split("\n") if line.strip()] if "PASS" not in feedback.upper() else []
    return {"errors": errors, "revision_count": state.get("revision_count", 0) + 1}

def code_corrector(state: AgentState):
    """节点E：代码自愈 (使用 DeepSeek)"""
    print("--- [DeepSeek] 自动修复代码 ---")
    project_text = "\n".join([f"==== {f['filepath']} ====\n{f['content']}\n" for f in state.get("project_files", [])])
    parser = PydanticOutputParser(pydantic_object=AfsimProjectFiles)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是高级修复专家。根据错误报告修复项目，重新输出。只修复报错处。\n{format_instructions}"),
        ("user", "【项目】\n{code}\n\n【错误】\n{errors}")
    ])
    chain = prompt | deepseek_llm | parser
    try:
        project_obj = chain.invoke({
            "code": project_text,
            "errors": "\n".join(state["errors"]),
            "format_instructions": parser.get_format_instructions()
        })
        return {"project_files": [f.model_dump() for f in project_obj.files], "errors": []}
    except Exception as e:
        return {"errors": [f"修正失败: {str(e)}"]}

# ==========================================
# 5. 人类在环 (Human-in-the-Loop) 拦截器
# ==========================================
def should_rebuild_with_human(state: AgentState):
    """条件边：拦截校验错误，交由人工判断是否进行修复"""
    
    if len(state["errors"]) > 0 and state["revision_count"] < 3:
        print("\n" + "!"*50)
        print("🚨 [警告] 语法校验器发现潜在错误：")
        for err in state["errors"]:
            print(f"  - {err}")
        print("!"*50)
        
        print("\n👇 【当前已生成的代码】 👇")
        for file in state.get("project_files", []):
            print(f"\n[{file['filepath']}]")
            print(file['content'])
            print("-" * 40)
        
        # 阻塞终端，等待人类输入
        while True:
            choice = input("\n👉 是否调用 DeepSeek 启动自动修复？(y/Y: 继续修复, n/N: 忽略错误并输出代码): ").strip().lower()
            if choice == 'y':
                print("\n--- 授权通过，交由 DeepSeek 修复 ---")
                return "fix"
            elif choice == 'n':
                print("\n--- 人工终止修复流程，直接输出当前结果 ---")
                return "finish"
            else:
                print("无效输入，请输入 y 或 n。")
                
    return "finish"

# ==========================================
# 6. 构建与编译 LangGraph 工作流
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("parser", scenario_parser)
workflow.add_node("build_weapons", weapon_builder)
workflow.add_node("build_platforms", platform_builder)
workflow.add_node("build_events", event_builder)
workflow.add_node("architect", project_architect)
workflow.add_node("validator", syntax_validator)
workflow.add_node("corrector", code_corrector)

# # 并行执行小模型
# workflow.add_edge("parser", "build_weapons")
# workflow.add_edge("parser", "build_platforms")
# workflow.add_edge("parser", "build_events")

# # 大模型重组
# workflow.add_edge("build_weapons", "architect")
# workflow.add_edge("build_platforms", "architect")
# workflow.add_edge("build_events", "architect")

# 将小模型任务改为串行，缓解本地显存压力
workflow.add_edge("parser", "build_weapons")
workflow.add_edge("build_weapons", "build_platforms")
workflow.add_edge("build_platforms", "build_events")
workflow.add_edge("build_events", "architect")

# 校验与人机交互流
workflow.add_edge("architect", "validator")
workflow.add_conditional_edges("validator", should_rebuild_with_human, {"fix": "corrector", "finish": END})
workflow.add_edge("corrector", "validator")

workflow.set_entry_point("parser")
app = workflow.compile()

# ==========================================
# 7. 终端交互主程序
# ==========================================
if __name__ == "__main__":
    print("======================================================")
    print("  AFSIM 脚本智能生成系统 (本地小模型 + DeepSeek + 人工干预) ")
    print("======================================================")
    print("请输入军事对抗场景描述 (输入 //end 结束)：\n")
    
    input_lines = []
    while True:
        try:
            line = input()
            if line.strip() == "//end":
                break
            input_lines.append(line)
        except EOFError:
            break
            
    task_description = "\n".join(input_lines).strip()
    
    if task_description:
        print("\n启动 AFSIM 多智能体工作流...\n")
        initial_state = {"original_prompt": task_description, "revision_count": 0, "errors": []}
        
        # 运行流（遇到 should_rebuild_with_human 时会自动在终端等待您的输入）
        final_state = app.invoke(initial_state)
        
        output_dir = Path("afsim_output_project")
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("任务结束！代码已落盘到本地目录：")
        print("="*60)
        
        for file_dict in final_state.get("project_files", []):
            filepath = output_dir / file_dict["filepath"]
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(file_dict["content"])
            print(f" 创建文件: {filepath}")