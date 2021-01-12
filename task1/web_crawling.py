

import re
from bs4 import BeautifulSoup
import os

headers = {
    'user-agent': 'Mozilla/5.0'
}


dirPath = '/Users/zhali/PycharmProjects/NLP/task1/save_html/'
output = "/Users/zhali/PycharmProjects/NLP/task1/save_html/content/"
num = 0
eng_writing = []
filename = 'content.txt'


def get_type_files(dirPath, type):
    files = os.listdir(dirPath)
    fileList = []
    for f in files:
        # print(f)
        # match files' type
        if os.path.isfile(dirPath + "/" + f):
            res = re.compile('.*\.' + type).match(f)
            if res != None:
                fileList.append(dirPath + res.group())

            res = re.compile('.*范文.\.' + type).match(f)
            if res != None:
                fileList.append(dirPath + res.group())

            res = re.compile('.*满分作文.*\.' + type).match(f)
            if res != None:
                fileList.append(dirPath + res.group())

            res = re.compile('.*写作.*\.' + type).match(f)
            if res != None:
                fileList.append(dirPath + res.group())
        else:
            print("not a file.")

    #  print(fileList)
    for item in fileList:
        # for specific chunk
        # print(item)
        crawl_xdf(item)
    return fileList


def crawl_xdf(url):
    num = 0
    # make a get request to fetch the raw HTML content
    # parse the HTML content
    soup = BeautifulSoup(open(url), 'html.parser')
    # get each url sw
    html_content = soup.find_all("div", class_="xqy_core_text")
    # get each title
    title = soup.find('h1', {'class': 'xqy_core_tit'}).text
    # print(title)
    for item in html_content:
        p_list = item.find_all('p')
        # print(p_list)

    content = ''
    pos = 0

    # find content
    # find the beginning of the content
    for i, p in enumerate(p_list):
        if re.compile('.*参考范文').match(p.text) or re.compile('.*范文').match(p.text) or \
                re.compile('.*优秀范文').match(p.text) or re.compile('.\ADear+')\
                or re.compile('.*模版.\.'):
            # if match_string.match(p.text):
            pos = i+1
            break
    # Use the new pos to find the content below
        else:
            return

    for j, p in enumerate(p_list[pos:len(p_list)]):
        # r'...' denotes raw strings which ignore escape code
        # start with English words
        if re.findall(r'[a-zA-Z]+', p.text):
            # print(p.text)
            content += p.text + '\n'

    num = num + 1
    eng_writing.append([title, content])
    # print('Title:', title, '\ncontent:', content)

    try:
        f_name = output + title + ".txt"
        print(f_name)
        file = open(f_name, "w")
        file.write(content)
        print("save file index: ", " file: ", f_name, " successfully! for url:", url)
        file.close()
    except:
        print("error occurs in saving file:", title, url)


if __name__ == "__main__":
    num = 0
    get_type_files(dirPath, 'html')
