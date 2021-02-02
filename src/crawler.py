from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pickle
import json

articles = []

Articles_till_now = ['2981549002', '2950893734', '3105081694']

addresses = ['2981549002', '2950893734', '3105081694']
counter = 1

to_fetch=5000

while len(articles) <= to_fetch:
    browser = webdriver.Chrome(ChromeDriverManager().install())

    try:
        with open('filename.pickle', 'rb') as handle:
            b = pickle.load(handle)
        with open('articles.pickle', 'rb') as handle:
            articles = pickle.load(handle)
        addresses, Articles_till_now, counter = b[0], b[1], b[2]
    except:
        pass

    try:
        while len(addresses) != 0 and counter <= to_fetch:
            print(counter, len(articles))
            address = addresses.pop(0)
            browser.get('https://academic.microsoft.com/paper/' + address)
            id = address

            time.sleep(4)

            title = browser.find_element_by_class_name('name')
            title = title.text

            abstract = ''

            els = browser.find_elements(By.TAG_NAME, 'p')
            for e in els:
                if len(e.text) > 50:
                    abstract = e.text

            el = browser.find_element_by_class_name('year')
            year = el.text

            authors = []
            el = browser.find_element_by_class_name('authors')
            els = el.find_elements_by_class_name('author-item')
            for el in els:
                e = el.find_element_by_class_name('author')
                authors.append(e.text)

            refs = []
            els = browser.find_elements_by_class_name('primary_paper')

            for el in els:
                e = el.find_element_by_tag_name('a')
                a_to_add = e.get_attribute('href').split('/')[-2]
                refs.append(a_to_add)
                if a_to_add not in Articles_till_now:
                    addresses.append(a_to_add)
                    Articles_till_now.append(a_to_add)

            articles.append(
                {"id": id, "title": title, "abstract": abstract, "date": year, "authors": authors, "references": refs})
            counter += 1

            a = [addresses, Articles_till_now, counter]
            with open('filename.pickle', 'wb') as handle:
                pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('articles.pickle', 'wb') as handle:
                pickle.dump(articles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except:
        browser.quit()
        print(len(articles))

with open('result.json', 'w') as fp:
    json.dump(articles, fp)