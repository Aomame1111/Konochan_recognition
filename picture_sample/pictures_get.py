# 松田好花のブログに投稿されている写真を枚数またはページ数を指定してblog_picturesに保存するプログラム
# 一部の写真はpathが違うため現在取得不可
import re
import os
import sys
import requests
from bs4 import BeautifulSoup


def pic_count(pic_images, pic_number):
    pic_number += len(pic_images)
    # print(pic_number)

    return pic_number


def pic_save(pic_images, pic_number, pic_name, number_limit=0, count=0):
    for img in pic_images:
        pic = requests.get(img['src'])
        pic_number_in_date = pic_images.index(img) + 1
        pic_number += 1
        while os.path.exists('blog_pictures/' + pic_name + '-' + str(pic_number_in_date) + '.jpg'):
            pic_number_in_date += 1
        with open('blog_pictures/' + pic_name + '-' + str(pic_number_in_date) + '.jpg', 'wb') as f:
            f.write(pic.content)
        if count != 0:
            print(pic_number)
        if pic_number == number_limit:
            break
    return pic_number


if __name__ == '__main__':
    compile_keyword = "^https://cdn.hinatazaka46.com/files/14/diary/official/member/moblog"
    picture_number = 0

    # 枚数指定ではpage_numberを十分大きい数にする。
    # ページ数指定ではpicture_limitを0にする。
    picture_limit = 100
    page_number = 200

    member_num = 20

    # 1ページ目
    url = 'https://www.hinatazaka46.com/s/official/diary/member/list?ima=0000&ct=%s'
    url = url % member_num
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    blogs = soup.find_all('div', class_="p-blog-article")

    for blog in blogs:
        date_text = blog.find('div', class_="c-blog-article__date").text
        date = re.search(r'\w[^ ]*', date_text).group(0)
        images = blog.find_all('img', src=re.compile(compile_keyword))
        picture_number = pic_save(images, picture_number, date, picture_limit)
        # picture_number = pic_count(images, picture_number)
        if picture_number == picture_limit:
            sys.exit('You got ' + str(picture_limit) + 'pictures')

    # 2ページ目以降
    for p in range(1, page_number):
        url = 'https://www.hinatazaka46.com/s/official/diary/member/list?ima=0000&page=%s&ct=%s&cd=member'
        r = requests.get(url % (p, member_num))
        soup = BeautifulSoup(r.text, 'lxml')
        if soup.title.text == '日向坂46 公式ブログ':
            print('The number of all pictures is ' + str(picture_number))
            sys.exit(str(p+1) + 'page is nothing')
        blogs = soup.find_all('div', class_="p-blog-article")

        for blog in blogs:
            date_text = blog.find('div', class_="c-blog-article__date").text
            date = re.search(r'\w[^ ]*', date_text).group(0)
            images = blog.find_all('img', src=re.compile(compile_keyword))
            picture_number = pic_save(images, picture_number, date, picture_limit)
            # picture_number = pic_count(images, picture_number)
            if picture_number == picture_limit:
                sys.exit('You got ' + str(picture_limit) + 'pictures')

    print('The number of pictures in ' + str(page_number) + 'pages is ' + str(picture_number))
