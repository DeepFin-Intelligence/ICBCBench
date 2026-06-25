import os
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

BOYUE = OpenAI(
    api_key=os.getenv('BOYUE_API_KEY'),
    base_url='http://35.220.164.252:3888/v1',
    timeout=1200.0,
    max_retries=1
)

DEEPSEEK = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url='https://api.deepseek.com/v1',
    timeout=600.0,
    max_retries=1
)

UIUI = OpenAI(
    api_key=os.getenv('UIUI_API_KEY'),
    base_url='https://sg.uiuiapi.com/v1',
    timeout=3600.0,
    max_retries=1
)

ALLMHUB = OpenAI(
    api_key=os.getenv('ALLMHUB_API_KEY'),
    base_url='https://api.allmhub.com/v1/',
    timeout=1200.0,
    max_retries=1
)

BAILIAN = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    timeout=600.0,
    max_retries=1
)

APIYI = OpenAI(
    api_key=os.getenv('APIYI_API_KEY'),
    base_url="https://vip.apiyi.com/v1"
)

OPENROUTER = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
    max_retries=1
)

OPENROUTER2 = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY2'),
    base_url="https://openrouter.ai/api/v1",
    max_retries=1
)

TEMP_API = OpenAI(
    api_key=os.getenv('TEMP_API_KEY'),
    base_url="http://172.96.141.132:3001/v1",
    max_retries=1
)

DMX = OpenAI(
    api_key=os.getenv('DMX_API_KEY'),
    base_url="https://www.dmxapi.cn/v1",
    max_retries=1,
    timeout=6000.0
)

MYDMX = OpenAI(
    api_key=os.getenv('MY_DMX_API_KEY'),
    base_url="https://www.dmxapi.cn/v1",
    max_retries=1,
    timeout=6000.0
)

MOONSHOT = OpenAI(
    api_key=os.getenv('KIMI_API_KEY'),
    base_url="https://api.moonshot.cn/v1",
    max_retries=1,
    timeout=6000.0
)