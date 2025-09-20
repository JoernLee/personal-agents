import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")

llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base=base_url,
    model_name="qwen-plus",
    temperature=0.7,
    max_tokens=1024
)

news_gen_prompt = PromptTemplate.from_template("请根据以下新闻标题撰写一段简短的新闻内容（100字以内）：标题：{title}")

news_chain = news_gen_prompt | llm

schemas = [
    ResponseSchema(name="time", description="事件发生的时间 "),
    ResponseSchema(name="location", description="事件发生的地点"),
    ResponseSchema(name="event", description="发生的具体事件")
]

parser = StructuredOutputParser.from_response_schemas(schemas)

summary_prompt = PromptTemplate.from_template("请从下面这段新闻内容中提取关键信息，并返回结构化JSON格式：\n\n{news}\n\n{format_instructions}")

summary_chain = summary_prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

full_chain = news_chain | summary_chain

result = full_chain.invoke({"title": "苹果近日在加州发布了AI芯片"})

print(result)