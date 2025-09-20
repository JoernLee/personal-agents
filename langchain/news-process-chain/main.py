"""
新闻处理链 - 使用LangChain构建的新闻生成和信息提取系统
功能：
1. 根据新闻标题生成新闻内容
2. 从新闻内容中提取结构化信息（时间、地点、事件）
"""

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.runnables import RunnableLambda

# 加载环境变量文件(.env)
load_dotenv()

# 从环境变量中获取API配置
api_key = os.getenv("DASHSCOPE_API_KEY")  # 阿里云DashScope API密钥
base_url = os.getenv("DASHSCOPE_BASE_URL")  # API基础URL

# 初始化大语言模型（使用通义千问qwen-plus模型）
llm = ChatOpenAI(
    openai_api_key=api_key,        # API密钥
    openai_api_base=base_url,      # API基础URL
    model_name="qwen-plus",        # 模型名称
    temperature=0.7,               # 温度参数，控制输出的随机性（0-1，越高越随机）
    max_tokens=1024                # 最大输出token数量
)

# ========== 第一步：新闻内容生成链 ==========
# 创建新闻生成的提示模板，输入变量为title（新闻标题）
news_gen_prompt = PromptTemplate.from_template("请根据以下新闻标题撰写一段简短的新闻内容（100字以内）：标题：{title}")

# 构建新闻生成链：提示模板 -> 大语言模型
news_chain = news_gen_prompt | llm

# ========== 第二步：信息提取链 ==========
# 定义输出结构的Schema，指定需要提取的字段
schemas = [
    ResponseSchema(name="time", description="事件发生的时间"),
    ResponseSchema(name="location", description="事件发生的地点"),
    ResponseSchema(name="event", description="发生的具体事件")
]

# 创建结构化输出解析器，用于将模型输出解析为JSON格式
parser = StructuredOutputParser.from_response_schemas(schemas)

# 创建信息提取的提示模板
# {news}: 新闻内容变量
# {format_instructions}: 格式说明变量（由parser自动生成）
summary_prompt = PromptTemplate.from_template("请从下面这段新闻内容中提取关键信息，并返回结构化JSON格式：\n\n{news}\n\n{format_instructions}")

# 构建信息提取链：提示模板（预填充格式说明） -> 大语言模型 -> 结构化解析器
summary_chain = summary_prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

# ========== 第三步：调试和完整链条构建 ==========
def debug_print(x):
    """
    调试函数：打印中间结果
    参数：x - 新闻生成链的输出结果
    返回：原样返回输入，用于链条传递
    """
    print("中间结果（新闻正文）:", x)
    return x

# 将调试函数包装为可运行的节点
debug_node = RunnableLambda(debug_print)

# ========== 构建完整的处理链 ==========
# 完整流程：新闻标题 -> 生成新闻内容 -> 调试输出 -> 提取结构化信息
full_chain = news_chain | debug_node | summary_chain

# ========== 执行示例 ==========
if __name__ == "__main__":
    # 测试输入：苹果AI芯片发布新闻
    test_title = "苹果近日在加州发布了AI芯片"
    
    print(f"输入标题: {test_title}")
    print("=" * 50)
    
    # 执行完整链条
    result = full_chain.invoke({"title": test_title})
    
    print("=" * 50)
    print("最终结果（结构化信息）:")
    print(result)