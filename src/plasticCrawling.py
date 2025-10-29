from icrawler.builtin import GoogleImageCrawler # icrawler
import os

path = "../dataset/train/PLASTIC"

def count_img(p):
    return len([f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

def download(p, keyword, target):
    while count_img(p) < target:
        remain = target - count_img(p)
        GoogleImageCrawler(storage={"root_dir": p}).crawl(keyword=keyword, max_num=remain)

download(path, "페트병", 100)