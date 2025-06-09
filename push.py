import os
import requests
import time
import json
import datetime
from tqdm import tqdm
from translate import translate, filter_relevent_papers
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

QUERYS = os.environ.get('QUERYS', 'cs.IR,cs.AI,cs.CL')
LIMITS = int(os.environ.get('LIMITS', 10))
FEISHU_URL = os.environ.get('FEISHU_URL', 'https://open.feishu.cn/open-apis/bot/v2/hook/523c7ec0-413b-40b7-97a5-4ebdb723b67f')
API_KEY = os.environ.get('API_KEY', 'AIzaSyCMrShkjkit8fAMR_1_7P6v5u2YGUMCyGw')


def get_yesterday():
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')

def search_arxiv_papers(search_term, max_results=50):
    papers = []

    url = f'http://export.arxiv.org/api/query?' + \
          f'search_query=all:{search_term}' + \
          f'&start=0&&max_results={max_results}' + \
          f'&sortBy=submittedDate&sortOrder=descending'

    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 500, 502, 503, 504 ])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    s.mount('https://', HTTPAdapter(max_retries=retries))
    response = s.get(url)

    if response.status_code != 200:
        return []

    feed = response.text
    entries = feed.split('<entry>')[1:]

    if not entries:
        return []

    for entry in tqdm(entries):
        title = entry.split('<title>')[1].split('</title>')[0].strip()
        summary = entry.split('<summary>')[1].split('</summary>')[0].strip().replace('\n', ' ').replace('\r', '')
        url = entry.split('<id>')[1].split('</id>')[0].strip()
        pub_date = entry.split('<published>')[1].split('</published>')[0]
        pub_date = datetime.datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

        papers.append({
            'title': title,
            'url': url,
            'pub_date': pub_date,
            'summary': summary,
            'translated': '',
        })
    return papers


def send_feishu_message(title, content, url=FEISHU_URL):
    card_data = {
        "config": {
            "wide_screen_mode": True
        },
        "header": {
            "template": "green",
            "title": {
                "tag": "plain_text",
                "content": title
            }
        },
        "elements": [
            {
                "tag": "img",
                "img_key": "img_v2_9781afeb-279d-4a05-8736-1dff05e19dbg",
                "alt": {
                    "tag": "plain_text",
                    "content": ""
                },
                "mode": "fit_horizontal",
                "preview": True
            },
            {
                "tag": "markdown",
                "content": content
            }
        ]
    }
    card = json.dumps(card_data)
    body = json.dumps({"msg_type": "interactive", "card": card})
    headers = {"Content-Type": "application/json"}
    requests.post(url=url, data=body, headers=headers)


def save_and_translate(papers, filename='push_papers.json'):
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([], f)
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.decoder.JSONDecodeError:
            results = []

    cached_title2idx = {result['title'].lower(): i for i, result in enumerate(results)}

    untranslated_papers = []

    translated_papers = []

    for paper in papers:
        title = paper['title'].lower()
        if title in cached_title2idx:
            translated_papers.append(
                results[cached_title2idx[title]]
            )
        else:
            untranslated_papers.append(paper)

    source = []
    for paper in untranslated_papers:
        source.append(paper['summary'])

    if source:
        target = translate(source)
        if len(target) == len(untranslated_papers):
            for i in range(len(untranslated_papers)):
                untranslated_papers[i]['translated'] = target[i]

    results.extend(untranslated_papers)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(
        f'[+] 总检索条数: {len(papers)} | 命中缓存: {len(translated_papers)} | 实际翻译: {len(untranslated_papers)}....')

    return translated_papers + untranslated_papers

def cronjon():
    print('------ 开始执行每日推送任务 ------')

    yesterday = get_yesterday()
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    print('------ 开始检索每日最新论文 ------')
    query_list = QUERYS.split(',')
    all_papers= []
    for query in query_list:
        papers = search_arxiv_papers(query, LIMITS)
        all_papers.extend(papers)

    print(len(all_papers))

    if not all_papers:
        push_title = f'Arxiv:{QUERYS}[X]@{today}'
        if FEISHU_URL:
            send_feishu_message(push_title, '[WARN] NO UPDATE TODAY!')
        print('[+] 每日推送任务执行结束')
        return True

    try:
        relevant_papers = filter_relevent_papers(all_papers)
        print(f'[+] 已筛选出{len(relevant_papers)}篇搜广推相关论文')
    except Exception as e:
        print(f'[!] 筛选搜广推相关论文失败: {e}')
        relevant_papers = all_papers

    if not relevant_papers:
        push_title = f'Arxiv:搜广推[X]@{today}'
        if FEISHU_URL:
            send_feishu_message(push_title, '[INFO] NO RELEVANT PAPERS TODAY!')
        print('[+] 未找到搜广推相关论文，推送任务结束')
        return True

    print('[+] 开始翻译搜广推相关论文并缓存....')
    translated_papers = save_and_translate(relevant_papers)

    print('[+] 开始推送搜广推相关论文....')

    for ii, paper in enumerate(tqdm(translated_papers, total=len(translated_papers), desc=f"论文推送进度")):
        title = paper['title']
        url = paper['url']
        pub_date = paper['pub_date']
        summary = paper['summary']
        translated = paper['translated']

        if pub_date == yesterday:
            msg_title = f'[Newest]{title}'
        else:
            msg_title = f'{title}'

        msg_url = f'URL: {url}'
        msg_pub_date = f'Pub Date：{pub_date}'
        msg_summary = f'Summary：\n\n{summary}'
        msg_translated = f'Translated (Powered by DeepSeek R1):\n\n{translated}'

        push_title = f'Arxiv:搜广推[{ii + 1}/{len(translated_papers)}]@{today}'
        msg_content = f"[{msg_title}]({url})\n\n{msg_pub_date}\n\n{msg_url}\n\n{msg_translated}\n\n{msg_summary}\n\n"

        if FEISHU_URL:
            send_feishu_message(push_title, msg_content, FEISHU_URL)

        time.sleep(12)


if __name__ == '__main__':
    cronjon()