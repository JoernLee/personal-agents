"""
LangChain 实现 CSV数据分析助手

功能概述：
1. 智能CSV数据分析：自动读取和分析CSV文件
2. 代码生成：根据用户需求自动生成pandas分析代码
3. 代码执行：安全执行生成的Python代码
4. 结果分析：AI解读分析结果并提供洞察

技术栈：
- LangChain: AI应用开发框架
- pandas: 数据分析库
- 通义千问(Qwen): 大语言模型
- PythonREPLTool: 代码执行工具

工作流程：
用户需求 → 生成分析代码 → 执行代码 → 分析结果 → 提供洞察
"""

# ========== 导入依赖库 ==========
import code                    # Python交互式解释器
from io import StringIO        # 字符串IO操作，用于捕获代码执行输出
import os                      # 操作系统接口
import re                      # 正则表达式，用于提取代码块
import sys                     # 系统相关参数和函数

from dotenv import load_dotenv                    # 环境变量加载
from langchain.chains import LLMChain             # LangChain链条（已废弃，这里未使用）
from langchain.prompts import PromptTemplate      # 提示模板
from langchain.schema import HumanMessage, SystemMessage  # 消息类型
from langchain_community.chat_models import ChatOpenAI    # OpenAI兼容的聊天模型
from langchain_experimental.tools import PythonREPLTool   # Python代码执行工具
import pandas as pd                               # 数据分析库

# 加载环境变量文件(.env)
load_dotenv()

# ========== 模型配置函数 ==========
def setup_qwen_model():
    """
    设置通义千问(Qwen)大语言模型
    
    功能：
    - 从环境变量获取API配置信息
    - 初始化ChatOpenAI兼容的Qwen模型实例
    - 配置模型参数（温度、最大token等）
    
    返回：
        ChatOpenAI: 配置好的大语言模型实例
        
    异常：
        ValueError: 当环境变量未设置时抛出
    """
    # 从环境变量获取阿里云DashScope API配置
    api_key = os.getenv("DASHSCOPE_API_KEY")    # API密钥
    base_url = os.getenv("DASHSCOPE_BASE_URL")  # API基础URL
    
    # 验证必要的环境变量是否已设置
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    if not base_url:
        raise ValueError("请设置 DASHSCOPE_BASE_URL 环境变量")
    
    # 初始化ChatOpenAI模型实例（兼容通义千问API）
    llm = ChatOpenAI(
        openai_api_key=api_key,        # API认证密钥
        openai_api_base=base_url,      # API服务地址
        model_name="qwen-plus",        # 模型名称（可选：qwen-turbo, qwen-plus, qwen-max等）
        temperature=0.7,               # 温度参数：控制输出随机性（0-1，越高越随机）
        max_tokens=1024                # 最大输出token数量
    )
    
    return llm

# ========== 测试函数 ==========
def test_pandas():
    """
    测试pandas读取和分析CSV文件功能
    
    功能：
    - 读取Stress_Dataset.csv文件
    - 显示数据的基本统计信息
    - 检查数据质量（缺失值、数据类型等）
    - 预览数据内容
    
    异常处理：
    - FileNotFoundError: 文件不存在
    - Exception: 其他读取错误
    """
    try:
        # 读取压力数据集CSV文件
        df = pd.read_csv("./langchain/csv-analyze/Stress_Dataset.csv")
        
        # 输出数据基本信息
        print("✅ CSV 文件读取成功!")
        print(f"📊 数据形状: {df.shape}")              # (行数, 列数)
        print(f"📋 列名: {list(df.columns)}")          # 所有列名
        print("-" * 50)
        print(f"🔤 数据类型:\n{df.dtypes}")             # 每列的数据类型
        print("-" * 50)
        print(f"📈 数据描述:\n{df.describe()}")         # 数值列的统计摘要
        print("-" * 50)
        print(f"❌ 数据缺失值:\n{df.isnull().sum()}")   # 每列缺失值数量
        print("-" * 50)
        print("\n👀 前5行数据:")
        print(df.head())                               # 显示前5行数据
        
    except FileNotFoundError:
        print("❌ 错误: 找不到 Stress_Dataset.csv 文件")
        return
    except Exception as e:
        print(f"❌ 读取CSV文件时出错: {e}")
        return

def test_csv_analyze():
    """
    测试CSV数据分析功能（使用AI直接分析）
    
    功能：
    - 读取CSV数据并生成摘要信息
    - 使用AI模型分析数据特征和潜在用途
    - 演示流式输出功能
    
    流程：
    1. 初始化AI模型
    2. 准备系统提示和用户问题
    3. 读取数据并生成摘要
    4. 使用流式输出获取AI分析结果
    """
    # 初始化通义千问模型
    llm = setup_qwen_model()

    # 构建消息序列
    messages = [
        SystemMessage(content="你是一个CSV数据分析助手，请用中文回答问题。"),
        HumanMessage(content="请分析一下 data.csv 文件的主要功能。")
    ]

    # 读取压力数据集
    df = pd.read_csv("./langchain/csv-analyze/Stress_Dataset.csv")

    # 生成数据摘要信息供AI分析
    data_summary = f"""
    数据集基本信息:
    - 数据形状: {df.shape}
    - 列名: {', '.join(df.columns)}
    - 数据类型: {df.dtypes.to_string()}
    - 统计摘要: {df.describe().to_string()}
    - 缺失值: {df.isnull().sum().to_string()}
    """

    # 添加数据信息到消息序列
    messages.append(HumanMessage(content=f"这是CSV文件的数据信息: {data_summary}"))

    print("🤖 Qwen模型流式回复:")
    print("-" * 50) 

    # 使用流式输出获取AI分析结果
    try:
        for chunk in llm.stream(messages):
            # 实时打印AI生成的内容
            print(chunk.content, end='', flush=True)
    except Exception as e:
        print(f"❌ 流式输出时出错: {e}")
        # 流式输出失败时回退到普通输出
        response = llm.invoke(messages)
        print(response.content)
        return

# ========== 工具函数 ==========
def execute_generated_code(code_str, df):
    """
    安全执行AI生成的Python代码（已废弃，推荐使用PythonREPLTool）
    
    功能：
    - 在受限环境中执行Python代码
    - 捕获代码执行的输出结果
    - 提供基本的安全保护
    
    参数：
        code_str (str): 要执行的Python代码字符串
        df (pandas.DataFrame): 数据框架对象
        
    返回：
        str: 代码执行结果或错误信息
        
    安全措施：
    - 限制可用的内置函数
    - 捕获和重定向输出
    - 异常处理和错误恢复
    """
    try:
        # 创建受限的执行环境，只包含必要的函数和对象
        safe_globals = {
            'pd': pd,           # pandas库
            'df': df,           # 数据框架
            'print': print,     # 打印函数
            '__builtins__': {   # 限制的内置函数集合
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'round': round,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
            }
        }
        
        # 重定向标准输出以捕获print语句的输出
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # 在安全环境中执行代码
        exec(code_str, safe_globals)
        
        # 恢复原始的标准输出
        sys.stdout = old_stdout
        
        # 获取执行结果
        result = captured_output.getvalue()
        return result.strip() if result else "代码执行成功，但没有输出"
        
    except Exception as e:
        # 确保在异常情况下也能恢复标准输出
        sys.stdout = old_stdout
        return f"❌ 代码执行出错: {str(e)}"

def extract_python_code(text):
    """
    从AI回复文本中提取Python代码块
    
    功能：
    - 识别markdown格式的代码块
    - 支持多种代码块格式（```python 和 ```）
    - 提取纯净的Python代码
    
    参数：
        text (str): AI模型的回复文本
        
    返回：
        str: 提取出的Python代码，如果没有找到代码块则返回原文本
        
    支持的格式：
    - ```python\n代码\n```
    - ```\n代码\n```
    """
    # 优先查找标准的Python代码块格式
    python_pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(python_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 如果没找到python标记的代码块，查找通用代码块
    code_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 如果都没找到，返回原始文本（去除首尾空白）
    return text.strip()

# ========== 核心工作流函数 ==========
def test_complete_workflow():
    """
    完整的AI数据分析工作流
    
    这是系统的核心功能，展示了完整的AI辅助数据分析流程：
    
    工作流程：
    1. 📊 数据准备：读取CSV文件，生成数据摘要
    2. 🤖 代码生成：AI根据用户需求生成pandas分析代码
    3. ⚡ 代码执行：使用PythonREPLTool安全执行生成的代码
    4. 🧠 结果分析：AI解读执行结果，提供数据洞察
    
    技术亮点：
    - 使用LangChain的LCEL语法构建处理链
    - PythonREPLTool提供安全的代码执行环境
    - 多轮AI交互实现智能分析
    
    应用场景：
    - 自动化数据探索
    - 业务分析报告生成
    - 数据科学辅助工具
    """
    # 初始化AI模型
    llm = setup_qwen_model()
    
    # 📊 第一步：数据准备
    print("📊 正在读取数据...")
    df = pd.read_csv("./langchain/csv-analyze/Stress_Dataset.csv")
    
    # 生成数据摘要供AI理解
    data_summary = f"""
    数据集基本信息:
    - 数据形状: {df.shape}
    - 列名: {', '.join(df.columns)}
    - 前3行数据: {df.head(3).to_string()}
    """
    
    # 定义用户分析需求 - 目前写死
    user_request = "你觉得从整体数据看，你有什么数据洞察吗？"
    
    print(f"🔍 用户需求: {user_request}")
    print("=" * 60)
    
    # 第二步：AI代码生成
    print("🤖 AI正在生成分析代码...")
    
    # 创建代码生成的提示模板
    code_gen_prompt = PromptTemplate.from_template(
        """请根据用户需要生成对应的使用pandas处理csv的python代码。

        CSV数据基本信息: 
        {data_info}

        用户需求: {user_request}

        要求：
        1. 只返回可执行的Python代码，不要其他解释
        2. 使用print()输出结果
        3. 代码要简洁明了
        4. csv读取路径：./langchain/csv-analyze/Stress_Dataset.csv

        请生成代码："""
    )
    
    # 构建代码生成链：提示模板 -> AI模型
    code_gen_chain = code_gen_prompt | llm
    
    # 调用AI生成代码
    ai_response = code_gen_chain.invoke({
        "data_info": data_summary,
        "user_request": user_request
    })
    
    # 从AI回复中提取纯净的Python代码
    generated_code = extract_python_code(ai_response.content)
    print("📝 生成的代码:")
    print("-" * 40)
    print(generated_code)
    print("-" * 40)
    
    # 第三步：安全执行代码
    print("\n⚡ 正在执行生成的代码...")
    
    # 创建Python代码执行工具，预置数据框和pandas库
    python_repl = PythonREPLTool(globals={"df": df, "pd": pd})
    
    # 执行AI生成的代码
    execution_result = python_repl.run(generated_code)
    
    print("📊 代码执行结果:")
    print("-" * 40)
    print(execution_result)
    print("-" * 40)
    
    # 第四步：AI分析结果
    print("\n🧠 AI正在分析执行结果...")
    
    # 创建结果分析的提示模板
    analysis_prompt = PromptTemplate.from_template(
        """请分析以下数据分析结果，并给出简洁的中文解释：

        用户需求: {user_request}
        执行的代码: {code}
        执行结果: {result}

        请用数据分析师的角度用3-5句话解释这个结果的含义："""
    )
    
    # 构建结果分析链：提示模板 -> AI模型
    analysis_chain = analysis_prompt | llm
    
    # 调用AI分析执行结果
    analysis_response = analysis_chain.invoke({
        "user_request": user_request,
        "code": generated_code,
        "result": execution_result
    })
    
    print("💡 AI数据洞察:")
    print("-" * 40)
    print(analysis_response.content)
    print("=" * 60)
    print("✅ 完整工作流执行完成！")


# ========== 程序入口 ==========
if __name__ == "__main__":
    """
    程序主入口
    
    提供三种测试模式：
    1. test_pandas(): 测试CSV文件读取和基本分析
    2. test_csv_analyze(): 测试AI直接分析CSV数据
    3. test_complete_workflow(): 完整的AI辅助数据分析工作流（推荐）
    """
    try:
        # 选择要运行的测试函数
        # test_pandas()           # 基础pandas测试
        # test_csv_analyze()      # AI直接分析测试
        test_complete_workflow()  # 完整工作流测试（默认）
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        print("请检查环境变量配置和CSV文件路径")
