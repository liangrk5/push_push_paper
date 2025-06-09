import os
import requests
import time
import json
from abc import ABC, abstractmethod

from tqdm import tqdm


class ModelClient(ABC):
    @abstractmethod
    def translate(self, source, system_prompt, temperature):
        pass

    @abstractmethod
    def call(self, source, system_prompt, temperature):
        pass

    def retry_call(self, source, system_prompt, temperature, attempts=3, base_delay=60):
        for attempt in range(attempts):
            try:
                return self.call(source, system_prompt, temperature)
            except requests.exceptions.RequestException as e:
                print(f"请求失败（尝试 {attempt + 1}/{attempts}）：", e)
            except requests.exceptions.HTTPError as e:
                print(f"HTTP错误（尝试 {attempt + 1}/{attempts}）：", e)
            except requests.exceptions.ConnectionError as e:
                print(f"连接错误（尝试 {attempt + 1}/{attempts}）：", e)
            except requests.exceptions.Timeout as e:
                print(f"请求超时（尝试 {attempt + 1}/{attempts}）：", e)
            except Exception as e:
                print(f"未知错误（尝试 {attempt + 1}/{attempts}）：", e)
            if attempt < attempts - 1:
                time.sleep(base_delay * (attempt + 1))
        return None

class GeminiClient(ModelClient):
    def __init__(self, api_key):
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def call(self, source, system_prompt=None, temperature=1.0):
        # Create messages array properly to handle None system_prompt
        from google.genai import types
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=source,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )
        return response.text.strip()

    def translate(self, source, system_prompt=None, temperature=1.0):
        translations = []
        for s in source:
            response = self.retry_call(s, system_prompt, temperature)
            translations.append(response if response is not None else '')
        return translations

    def assess_relevance(self, summary, system_prompt=None, temperature=0.2):
        """Assess if a paper is relevant to search, advertising, or recommendation systems"""
        prompt = """请评估以下论文摘要是否与搜索系统(Search)、广告技术(Advertising)或推荐系统(Recommendation)相关。
                        详细分析论文内容，判断其是否讨论了搜索引擎、计算广告、点击率预估、个性化推荐、CTR预测、排序算法等相关技术。
                        如果相关，请返回'Yes'并简要说明原因；如果不相关，请返回'No'。

                        论文摘要:
                        """
        full_prompt = prompt + summary

        # Provide a default system prompt if none is provided
        if system_prompt is None:
            system_prompt = "你是一位专业的论文分析专家，擅长评估论文是否与特定领域相关。"

        response = self.retry_call(full_prompt, system_prompt, temperature)
        if response is None:
            return False  # Default to not relevant if API call fails

        # Check if response contains "yes"
        return "yes" in response.strip().lower()

class DeepSeekClient(ModelClient):
    def __init__(self, api_key, base_url):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(self, source, system_prompt=None, temperature=1.0):
        # Create messages array properly to handle None system_prompt
        messages = []
        if system_prompt:
            messages.append(system_prompt)
        messages.append({"role": "user", "content": source})

        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            temperature=temperature,
            stream=False
        )
        ret = response.choices[0].message.content.strip()
        return ret

    def translate(self, source, system_prompt=None, temperature=1.0):
        translations = []
        for s in source:
            response = self.retry_call(s, system_prompt, temperature)
            translations.append(response if response is not None else '')
        return translations

    def assess_relevance(self, summary, system_prompt=None, temperature=0.2):
        """Assess if a paper is relevant to search, advertising, or recommendation systems"""
        prompt = """请评估以下论文摘要是否与搜索系统(Search)、广告技术(Advertising)或推荐系统(Recommendation)相关。
                        详细分析论文内容，判断其是否讨论了搜索引擎、计算广告、点击率预估、个性化推荐、CTR预测、排序算法等相关技术。
                        如果相关，请返回'Yes'并简要说明原因；如果不相关，请返回'No'。

                        论文摘要:
                        """
        full_prompt = prompt + summary

        # Provide a default system prompt if none is provided
        if system_prompt is None:
            system_prompt = {
                "role": "system",
                "content": "你是一位专业的论文分析专家，擅长评估论文是否与特定领域相关。"
            }

        response = self.retry_call(full_prompt, system_prompt, temperature)
        if response is None:
            return False  # Default to not relevant if API call fails

        # Check if response contains "yes"
        return "yes" in response.strip().lower()


def init_model_client(is_gemini=True):
    api_key = os.environ.get("API_KEY", 'AIzaSyCMrShkjkit8fAMR_1_7P6v5u2YGUMCyGw')
    base_url = "https://api.deepseek.com"
    if is_gemini:
        return GeminiClient(api_key=api_key)
    return DeepSeekClient(api_key=api_key, base_url=base_url)


def translate(source):
    system_prompt = "你是一位专业的翻译人员，擅长在人工智能领域内进行高质量的英文到中文翻译。你将会接收到一篇涉及自然语言处理（NLP）、信息检索（IR）、计算机视觉（CV）等方向的英文论文摘要。请准确翻译摘要内容，确保所有专业术语和技术细节得到正确的表达。请记住从英文翻译到中文"

    model_client = init_model_client()

    return model_client.translate(source, system_prompt=system_prompt, temperature=1.0)


def filter_relevent_papers(papers):
    relevant_papers = []
    model_client = init_model_client()

    print('------ 使用Gemini Flash筛选搜广推相关论文 ------')

    for paper in tqdm(papers, desc='论文筛选进度'):
        summary = paper['summary']
        if model_client.assess_relevance(summary):
            relevant_papers.append(paper)

    print(f'[+] 总论文数: {len(papers)} | 相关论文数: {len(relevant_papers)}')

    return relevant_papers