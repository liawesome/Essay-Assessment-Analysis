import requests
import bs4
import re
import json
import concurrent.futures
import math
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

headers = {
    'user-agent': 'Mozilla/5.0'
}

one_thread_save_pages = 200  # every thread handle 200 html.
output = "/Users/zhali/PycharmProjects/NLP/task1/save_html/"


def get_list_content(url):
    link = requests.get(url, headers=headers)
    link.raise_for_status()
    link.encoding = link.apparent_encoding

    soup = BeautifulSoup(link.text, 'html.parser')
    paging = soup.find_all('h3')
    # print(paging)
    result = []
    # print(paging)
    for item in paging:
        a = item.find_all('a', {'class': 'entry_tit'})
        # print(a)

        if len(a) > 0:
            title, url = a[0].get("title"), a[0].get("href")
            title = title.strip("\u200b").strip()
            print(title, url)
        if [title, url] in result:  # remove duplicates
            print("same title: ", title)
        else:
            result.append([title, url])

    return result


def get_all_webpage():
    all_list = []
    start = 2
    end = 20
    base_url = 'https://gaokao.koolearn.com/yingyu/zuowen/'
    # first html page
    # extend means append website
    all_list.extend(get_list_content(base_url))

    # get other pages, notice: range [start,end + 1), so is end+1 not end.
    for i in range(start, end + 1):
        # loop through each link
        url = base_url + str(i) + ".html"
        print(url)
        a = get_list_content(url)
        all_list.extend(a)
    print(len(all_list))

    # save the list of all html pages.
    # example of all_list[[title1, url1], [title2, url2]]

    with open(output + "list.txt", "w") as f:
        f.write(json.dumps(all_list))
    return all_list


def save_all_pages():
    list_page = get_all_webpage()
    f = open(output + "list.txt", "r")

    list_page = json.loads(f.read())
    f.close()
    # print(len(list_page))

    pool_len = math.floor(len(list_page) / one_thread_save_pages)
    # print("pool size:", pool_len + 1)
    # fix max_workers first
    executor = ThreadPoolExecutor(max_workers=pool_len + 1)

    for i in range(pool_len):
        # for list_page[0:200] [200,400]
        executor.submit(saveHtml, list_page[i * one_thread_save_pages: (i + 1) * one_thread_save_pages])
    # add the remaining,for [400, 532]
    executor.submit(saveHtml, list_page[pool_len * one_thread_save_pages:len(list_page)])


def saveHtml(f_l):
    print(len(f_l))
    for i, page in enumerate(f_l):
        url = page[1]
        title = page[0]
        html = requests.get(url, headers=headers)
        html.raise_for_status()
        html.encoding = html.apparent_encoding
        try:

            title = title.replace("/", "_")
            f_name = output + title + ".html"
            # find if f_name exists

            fh = open(f_name, "w")
            fh.write(html.text)
            print("save file index: ", i, " file: ", f_name, " successfully! for url:", url)
            fh.close()
        except:
            print("error occurs in saving file:", title, url)


if __name__ == "__main__":
    save_all_pages()
