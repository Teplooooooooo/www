import requests
import re
import os


def parse_errors(log_file):
    errors = []
    with open(log_file, 'r',
              #encoding='utf-8'
              ) as f:
        log_content = f.read()

    # Регулярное выражение для ошибок сборки
    build_error_pattern = re.compile(
        r'(error|ERROR|Error|failed|FAILED|Could NOT find|could not find):?\s*(.*?)(?=\n\S|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    # Поиск ошибок сборки
    for match in build_error_pattern.finditer(log_content):
        if not any(match.group(0) in e['description'] for e in errors):
            errors.append({
                'description': match.group(0).strip()
            })

    return errors


def analyze_logs(logs_dir):
    results = {}
    for filename in os.listdir(logs_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(logs_dir, filename)
            errors = parse_errors(filepath)
            if errors:
                results[filename] = errors

    return results

# Замените на нужный эндпоинт или параметры
url = input("Введите url")
response = requests.get(url)
response.raise_for_status()  # Проверка на ошибки
responses = response.text.split() #Вся страница, разбитая построчно
herfs = []

#Достаем части ссылок на логи
for tag in responses:
    if tag[:6] == 'href="':
        herfs.append(tag.split(sep='"')[1])
for herf in herfs[2:]:
    file = open(f"logs/{herf}.txt", "w")
    try:file.write(requests.get(f"{url}{herf}").text)
    except:pass
    file.close()




log_directory = "logs"
analysis_results = analyze_logs(log_directory)
k = 0
    # Вывод результатов
for filename, errors in analysis_results.items():
    print(f"\nФайл: {filename}")
    k += 1
    for error in errors:
        print(f'''"{error['description']}\n"''')

print(k)
