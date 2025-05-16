import requests
import re

#Вводим Url и парсим страницу
url = input("Введите url ")
response = requests.get(url)
responses = response.text.split()
log = {}
#Находим herf, добавляем в словарь ключ с названием проекта и значение пустой массив
for tag in responses[50:]:
    if tag[:6] == 'href="':
        tag = tag.split(sep='"')[1]
        build_error_pattern = re.compile(
            r'(E:|error:|ERROR|Error|failed|FAILED|Could NOT find|could not find):?\s*(.*?)(?=\n\S|\Z)',
            re.DOTALL | re.IGNORECASE)
        log[f"{tag}"] = []
        log_content = requests.get(f"{url}{tag}").text
        #С помощью регулярных выражений находим ошибки и добавляем их в массив
        for match in build_error_pattern.finditer(log_content):
            log[f"{tag}"].append(match.group(0).strip())

for lg in log:
    print(lg, log[f"{lg}"])
