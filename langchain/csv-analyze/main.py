"""
LangChain 实现 CSV数据分析助手
"""
import code
from io import StringIO
import os
import re
import sys

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
import pandas as pd

# 加载环境变量
load_dotenv()

def setup_qwen_model():
    """设置 Qwen3 模型"""
    # 获取 API 密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    if not base_url:
        raise ValueError("请设置 DASHSCOPE_BASE_URL 环境变量")
    
    # 初始化 ChatOpenAI 模型
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name="qwen-plus",  # 可以选择不同的 Qwen 模型
        temperature=0.7,
        max_tokens=1024
    )
    
    return llm

def test_pandas():
    """测试 pandas"""
    try:
        # 读取当前目录下的 Stress_Dataset.csv 文件
        df = pd.read_csv("./langchain/csv-analyze/Stress_Dataset.csv")
        
        # 显示基本信息
        print("CSV 文件读取成功!")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("-" * 50)
        print(f"数据类型: {df.dtypes}")
        print("-" * 50)
        print(f"数据描述: {df.describe()}")
        print("-" * 50)
        print(f"数据缺失值: {df.isnull().sum()}")
        print("-" * 50)
        print("-" * 50)
        print("\n前5行数据:")
        print(df.head())
        
    except FileNotFoundError:
        print("错误: 找不到 Stress_Dataset.csv 文件")
        return
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

def test_csv_analyze():
    """测试 CSV 数据分析"""
    llm = setup_qwen_model()

    messages = [
        SystemMessage(content="你是一个CSV数据分析助手，请用中文回答问题。"),
        HumanMessage(content="请分析一下 data.csv 文件的主要功能。")
    ];

    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv("./langchain/csv-analyze/Stress_Dataset.csv")

    # 准备数据摘要信息
    data_summary = f"""
    数据集基本信息:
    - 数据形状: {df.shape}
    - 列名: {', '.join(df.columns)}
    - 数据类型: {df.dtypes.to_string()}
    - 统计摘要: {df.describe().to_string()}
    - 缺失值: {df.isnull().sum().to_string()}
    """

    messages.append(HumanMessage(content=f"这是CSV文件的数据信息: {data_summary}"))

    print("Qwen3 流式回复:")
    print("-" * 50) 

    # 使用流式输出
    try:
        for chunk in llm.stream(messages):
            # 打印流式输出
            print(chunk.content, end='', flush=True)
    except Exception as e:
        print(f"流式输出时出错: {e}")
        # 如果流式输出失败，回退到普通输出
        response = llm.invoke(messages)
        print(response.content)
        return

def execute_generated_code(code_str, df):
    """安全执行生成的Python代码"""
    try:
        # 创建一个安全的执行环境
        safe_globals = {
            'pd': pd,
            'df': df,
            'print': print,
            '__builtins__': {
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
        
        # 捕获输出
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # 执行代码
        exec(code_str, safe_globals)
        
        # 恢复输出
        sys.stdout = old_stdout
        
        # 获取执行结果
        result = captured_output.getvalue()
        return result.strip() if result else "代码执行成功，但没有输出"
        
    except Exception as e:
        sys.stdout = old_stdout
        return f"代码执行出错: {str(e)}"

def extract_python_code(text):
    """从AI回复中提取Python代码"""
    # 查找```python...```代码块
    python_pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(python_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 如果没找到代码块，查找```...```
    code_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return text.strip()

def test_complete_workflow():
    """完整的工作流：需求 → 生成代码 → 执行代码 → 分析结果"""
    llm = setup_qwen_model()
    
    # 读取CSV数据
    df = pd.read_csv("./langchain/csv-analyze/Stress_Dataset.csv")
    
    # 准备数据摘要
    data_summary = f"""
    数据集基本信息:
    - 数据形状: {df.shape}
    - 列名: {', '.join(df.columns)}
    - 前3行数据: {df.head(3).to_string()}
    """
    
    # 用户需求
    user_request = "你觉得从整体数据看，你有什么数据洞察吗？"
    
    print(f"🔍 用户需求: {user_request}")
    print("=" * 60)
    
    # 第一步：生成代码
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
    
    code_gen_chain = code_gen_prompt | llm
    
    print("🤖 AI正在生成代码...")
    ai_response = code_gen_chain.invoke({
        "data_info": data_summary,
        "user_request": user_request
    })
    
    # 提取代码
    generated_code = extract_python_code(ai_response.content)
    print("📝 生成的代码:")
    print("-" * 40)
    print(generated_code)
    print("-" * 40)
    
    # 第二步：执行代码
    print("\n⚡ 执行代码中...")
    # 创建Python执行工具
    python_repl = PythonREPLTool(globals={"df": df, "pd": pd})
    execution_result = python_repl.run(generated_code)
    print("📊 执行结果:")
    print("-" * 40)
    print(execution_result)
    print("-" * 40)
    
    # 第三步：AI分析结果
    analysis_prompt = PromptTemplate.from_template(
        """请分析以下数据分析结果，并给出简洁的中文解释：

        用户需求: {user_request}
        执行的代码: {code}
        执行结果: {result}

        请用数据分析师的角度用3-5句话解释这个结果的含义："""
    )
    
    analysis_chain = analysis_prompt | llm
    
    print("\n🧠 AI正在分析结果...")
    analysis_response = analysis_chain.invoke({
        "user_request": user_request,
        "code": generated_code,
        "result": execution_result
    })
    
    print("💡 结果分析:")
    print("-" * 40)
    print(analysis_response.content)
    print("=" * 60)


if __name__ == "__main__":
    try:
        # test_pandas()
        # test_csv_analyze()
        test_complete_workflow()  # 完整工作流
        
    except Exception as e:
        print(f"测试失败: {e}")
