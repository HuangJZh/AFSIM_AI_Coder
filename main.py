import os
import json
import shutil
from pathlib import Path
from typing import TypedDict, List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# ==========================================
# 0. 环境准备 (DeepSeek 驱动)
# ==========================================
load_dotenv()
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

deepseek_llm = ChatOpenAI(
    model="deepseek-reasoner", 
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com/v1",
    max_tokens=8192,
    temperature=0.1
)

# 加载知识库 (假设 tutorials 文件夹下有对应的 md 文件)
TUTORIALS_DIR = Path("tutorials")
def load_tutorial(filename: str) -> str:
    filepath = TUTORIALS_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

KNOWLEDGE_PLATFORMS = load_tutorial("afsim_platforms.md") + "\n" + load_tutorial("afsim_mover_types_routes.md")
KNOWLEDGE_WEAPONS = load_tutorial("afsim_weapons_and_commands.md") + "\n" + load_tutorial("afsim_signatures_and_parts.md")
KNOWLEDGE_EVENTS = load_tutorial("afsim_command_and_reports.md")
KNOWLEDGE_ERRORS = load_tutorial("task_correction.md")

# ==========================================
# 1. 数据结构定义
# ==========================================
class AfsimProjectFile(BaseModel):
    filepath: str = Field(description="文件路径，例如 'weapon.txt', 'platforms/air.txt', 'scenarios/blue.txt'")
    content: str = Field(description="AFSIM 脚本代码")

class AfsimProjectFiles(BaseModel):
    files: List[AfsimProjectFile] = Field(description="生成的文件列表")

# 全局状态
class AgentState(TypedDict):
    original_prompt: str
    scenario_json: str
    platform_files: List[dict]  # 存放第二步生成的平台定义
    scenario_files: List[dict]  # 存放第三步生成的阵营想定
    project_files: List[dict]   # 最终整合的所有文件
    errors: List[str]
    revision_count: int
    enable_validation: bool     # 新增：是否启用校验节点

# ==========================================
# 2. 工作流节点定义 (完全匹配四步走流程)
# ==========================================

def scenario_parser(state: AgentState):
    """前置节点：提取任务中的核心要素 (兵力、航线、事件)"""
    print("--- [解析阶段] 提取任务核心要素 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 AFSIM 数据提取专家。请将用户的任务描述提取为结构化的 JSON 格式。\n包含: factions(阵营及实体)、routes(航线)、events(交战事件)等。只需输出纯 JSON，不要包含 Markdown 符号。"),
        ("user", "{prompt}")
    ])
    chain = prompt | deepseek_llm
    response = chain.invoke({"prompt": state["original_prompt"]})
    json_str = response.content.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:-3].strip()
    return {"scenario_json": json_str}

def build_platforms(state: AgentState):
    """第二步：编写平台底层类型定义 (生成 platforms/*.txt)"""
    print("--- [构建阶段 - 步骤2] 生成平台定义 (platforms/ 文件夹) ---")
    parser = PydanticOutputParser(pydantic_object=AfsimProjectFiles)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是 AFSIM 平台构建专家。请根据提取的 JSON，创建 `platforms` 文件夹下的所有平台类型文件（如 air.txt, missile.txt, ship.txt 等）。
        
        【🚨 绝对红线约束 - 违反任何一条将导致编译失败】：
        1. 严格区分类型与实例：`platform_type` 是抽象模板！**绝对禁止**在 `platform_type` 或其内部的 `mover` 中出现 `side`, `position`, `altitude`, `speed`, `heading` 等具体状态参数。这些只能在场景实例化时使用。
        2. 特征块必须外部定义：`radar_signature`、`infrared_signature` 等完整的定义块（包含 constant 或 inline_table）**必须写在 `platform_type` 外部**。在 `platform_type` 内部仅用名称引用（如直接写 `radar_signature MY_SIG_NAME`）。
        3. 武器杀伤语法：定义 `WSF_GRADUATED_LETHALITY` 武器效果时，绝对禁止编造 `range` 或 `kill_probability` 块！必须严格使用单行语法：`radius_and_pk <距离> <概率>`（例如：`radius_and_pk 500 m 0.8`）。
        4. 时间单位：所有的秒数单位必须严格使用 `s`，绝对禁止使用 `sec`。
        5. 限制输出目录：你**只能**生成 `platforms/` 目录下的文件。
        6. 保持简洁：如果没有明确要求，不要随意为非智能平台（如 tank, ship, satellite）添加 `WSF_SCRIPT_PROCESSOR` 处理器。
        
        【教程参考】:\n{knowledge}\n
        {format_instructions}"""),
        ("user", "任务数据:\n{json_data}")
    ])
    chain = prompt | deepseek_llm | parser
    try:
        project_obj = chain.invoke({
            "knowledge": KNOWLEDGE_PLATFORMS + "\n" + KNOWLEDGE_WEAPONS,
            "json_data": state["scenario_json"],
            "format_instructions": parser.get_format_instructions()
        })
        return {"platform_files": [f.model_dump() for f in project_obj.files if f.filepath.startswith("platforms/")]}
    except Exception as e:
        print(f"  -> 构建平台出错: {e}")
        return {"platform_files": []}

def build_scenarios(state: AgentState):
    """第三步：编写阵营实例与航线 (生成 scenarios/blue.txt, red.txt)"""
    print("--- [构建阶段 - 步骤3] 生成阵营想定 (scenarios/ 文件夹) ---")
    
    # 提取已生成的平台文件路径，传递给大模型，作为强制约束的上下文
    generated_platform_paths = [f["filepath"] for f in state.get("platform_files", [])]
    paths_context = "\n".join(generated_platform_paths)
    
    parser = PydanticOutputParser(pydantic_object=AfsimProjectFiles)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是 AFSIM 想定部署专家。请根据提取的 JSON，创建 `scenarios` 文件夹下的阵营文件（如 blue.txt, red.txt）。
        【强制约束规则】：
        1. 使用 `include_once` 引入平台文件。
        2. 你**只能**引用以下确实存在的已生成平台文件列表，绝对禁止臆造不存在的文件（如 ground.txt, surface.txt 等）：
        <已生成的平台文件>
        {paths_context}
        </已生成的平台文件>
        3. 使用 platform 命令实例化实体，配置 side, icon, track, position, route 等。
        
        【教程参考】:\n{knowledge}\n
        {format_instructions}"""),
        ("user", "任务数据:\n{json_data}")
    ])
    chain = prompt | deepseek_llm | parser
    try:
        project_obj = chain.invoke({
            "knowledge": KNOWLEDGE_PLATFORMS,
            "json_data": state["scenario_json"],
            "paths_context": paths_context,  # 注入防幻觉上下文
            "format_instructions": parser.get_format_instructions()
        })
        return {"scenario_files": [f.model_dump() for f in project_obj.files if f.filepath.startswith("scenarios/")]}
    except Exception as e:
        print(f"  -> 构建想定出错: {e}")
        return {"scenario_files": []}

def build_main_assembler(state: AgentState):
    """第一步 & 第四步：编写核心启动文件并汇编全局文件"""
    print("--- [汇编阶段 - 步骤1&4] 生成 weapon.txt 并打包项目 ---")
    
    # 获取前面生成的场景文件列表，准备动态 include
    scenario_paths = [f["filepath"] for f in state.get("scenario_files", [])]
    include_str = "\n".join([f"include {path}" for path in scenario_paths])
    
    # 按照标准流程，自动生成标准 weapon.txt
    weapon_txt_content = f"""# AFSIM Main Execution Script
define_path_variable NAME weapon
log_file output/$(NAME).log

# 引入各个阵营文件
{include_str}

event_output
   file output/$(NAME).evt
end_event_output

event_pipe  
   file output/$(NAME).aer
end_event_pipe

end_time 30 m
"""
    
    main_file = {"filepath": "weapon.txt", "content": weapon_txt_content}
    
    # 汇总所有文件
    all_project_files = [main_file] + state.get("platform_files", []) + state.get("scenario_files", [])
    
    return {"project_files": all_project_files}

def should_run_validation(state: AgentState):
    """条件边：判断是否需要执行语法校验"""
    if state.get("enable_validation", False):
        return "validate"
    print("--- [系统提示] 已跳过大模型语法校验阶段 ---")
    return "finish"

def syntax_validator(state: AgentState):
    """节点D：跨文件全局校验"""
    print("--- [审计阶段] 跨文件语法与逻辑校验 ---")
    project_text = "\n".join([f"==== {f['filepath']} ====\n{f['content']}\n" for f in state.get("project_files", [])])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 AFSIM 语法审计专家。请检查以下项目代码，检查 include 路径是否正确，逻辑是否闭环。\n【排错指南】\n{knowledge}\n\n如果完全无误，请仅回复 PASS；如果有错误，请逐条指出。"),
        ("user", "【项目代码】\n{code}")
    ])
    response = deepseek_llm.invoke(prompt.format_messages(code=project_text, knowledge=KNOWLEDGE_ERRORS))
    feedback = response.content.strip()
    
    errors = [line for line in feedback.split("\n") if line.strip()] if "PASS" not in feedback.upper() else []
    return {"errors": errors, "revision_count": state.get("revision_count", 0) + 1}

def code_corrector(state: AgentState):
    """节点E：代码自愈"""
    print("--- [修复阶段] 根据审计意见自动修复代码 ---")
    project_text = "\n".join([f"==== {f['filepath']} ====\n{f['content']}\n" for f in state.get("project_files", [])])
    parser = PydanticOutputParser(pydantic_object=AfsimProjectFiles)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 AFSIM 修复专家。请根据错误报告修复项目代码，保持原有的多文件目录结构。\n{format_instructions}"),
        ("user", "【原始项目】\n{code}\n\n【错误报告】\n{errors}")
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

def should_rebuild_with_human(state: AgentState):
    """条件边：拦截器"""
    if len(state["errors"]) > 0 and state["revision_count"] < 3:
        print("\n" + "!"*50)
        print(" [警告] 发现潜在错误：")
        for err in state["errors"]:
            print(f"  - {err}")
        print("!"*50)
        
        while True:
            choice = input("\n👉 是否调用 DeepSeek 修复？(y/Y: 修复, n/N: 忽略并输出): ").strip().lower()
            if choice == 'y':
                return "fix"
            elif choice == 'n':
                return "finish"
    return "finish"

# ==========================================
# 3. 编排并编译 LangGraph 工作流
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("parser", scenario_parser)
workflow.add_node("build_platforms", build_platforms)     # 第二步
workflow.add_node("build_scenarios", build_scenarios)     # 第三步
workflow.add_node("assemble_main", build_main_assembler)  # 第一步&第四步组合
workflow.add_node("validator", syntax_validator)
workflow.add_node("corrector", code_corrector)

# 定义严格的串行执行流
workflow.add_edge("parser", "build_platforms")
workflow.add_edge("build_platforms", "build_scenarios")
workflow.add_edge("build_scenarios", "assemble_main")

# 根据用户选择，决定是否进入校验循环
workflow.add_conditional_edges("assemble_main", should_run_validation, {"validate": "validator", "finish": END})

# 校验循环内部路线
workflow.add_conditional_edges("validator", should_rebuild_with_human, {"fix": "corrector", "finish": END})
workflow.add_edge("corrector", "validator")

workflow.set_entry_point("parser")
app = workflow.compile()

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    print("======================================================")
    print("  AFSIM 脚本生成系统 (完全映射你的四步走架构) ")
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
        # 新增选项：是否启用校验
        val_choice = input("\n👉 是否启用大模型跨文件语法校验（耗费较多Token，调试中）？(y/n，默认 n): ").strip().lower()
        enable_val = (val_choice == 'y')

        print("\n 启动 AFSIM 工作流...\n")
        initial_state = {
            "original_prompt": task_description, 
            "revision_count": 0, 
            "errors": [],
            "platform_files": [],
            "scenario_files": [],
            "project_files": [],
            "enable_validation": enable_val  # 将选项注入状态
        }
        
        final_state = app.invoke(initial_state)
        
        # 核心修改：在写入新文件前，彻底清空输出文件夹
        output_dir = Path("afsim_output_project")
        if output_dir.exists():
            shutil.rmtree(output_dir) # 递归删除整个目录及内容
            
        # 重新创建基础文件夹结构
        (output_dir / "platforms").mkdir(parents=True, exist_ok=True)
        (output_dir / "scenarios").mkdir(parents=True, exist_ok=True)
        (output_dir / "output").mkdir(parents=True, exist_ok=True) # 输出用的空文件夹
        
        print("\n" + "="*60)
        print(" 任务结束！代码已落盘到本地目录：")
        print("="*60)
        
        for file_dict in final_state.get("project_files", []):
            filepath = output_dir / file_dict["filepath"]
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(file_dict["content"])
            print(f"  创建文件: {filepath}")