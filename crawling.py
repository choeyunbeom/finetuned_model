import time
import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from konlpy.tag import Komoran
from collections import Counter

komoran = Komoran(userdic = "./data/user_dict.tsv")

with open('./data/stopwords.txt', 'r') as f:
    list_file = f.readlines()
STOPWORDS = [line.rstrip('\n') for line in list_file] 

def infinite_loop(driver):
    #스크롤 내리기
    last_page_height = driver.execute_script("return document.documentElement.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(1.0)
        new_page_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_page_height == last_page_height:
            time.sleep(1.0)
            if new_page_height == driver.execute_script("return document.documentElement.scrollHeight"):
                break
        else:
            last_page_height = new_page_height


def common_words(url):
    options = webdriver.ChromeOptions()
    # 창 숨기는 옵션 추가
    options.add_argument("headless")   
    driver = webdriver.Chrome(options=options)
    novel_title = []
    
    driver.get(url)
    
    infinite_loop(driver)
    
    soup = BeautifulSoup(driver.page_source, "lxml")
    titles = soup.select('div.jsx-2792908821.font-small1.line-clamp-2.break-all.text-el-60')
    
    for title in titles:
        novel_title.append(title.text)
    driver.quit()
    
    title_list = []
    for noun in pd.Series(novel_title).map(lambda x: komoran.nouns(x)):
        title_list.extend(noun)
    
    noun_total_selected = [noun for noun in title_list if noun not in STOPWORDS and len(noun) != 1]
    noun_total_selected = Counter(noun_total_selected)
    return noun_total_selected.most_common(20)