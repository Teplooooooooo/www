# организаторы буськи, кейс кайф
import requests
import re
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Вводим Url и парсим страницу
url = input("Введите url ")
response = requests.get(url)
responses = response.text.split()
log = {}
# Находим herf, добавляем в словарь ключ с названием проекта и значение пустой массив
for tag in responses[50:]:
    if tag[:6] == 'href="':
        tag = tag.split(sep='"')[1]
        build_error_pattern = re.compile(
            r'(error:|failed|could not find):?\s*(.*?)(?=\n\S|\Z)',
            re.DOTALL | re.IGNORECASE)
        log[f"{tag}"] = []
        log_content = requests.get(f"{url}{tag}").text
        # С помощью регулярных выражений находим ошибки и добавляем их в массив
        for match in build_error_pattern.finditer(log_content):
            line = match.group(0).strip()
            tl = line.split(sep='\n')
            # Убираем ненужные строки
            if line not in log[f"{tag}"]:
                k = 0
                for l in tl:
                    if '|' in l or l == 'failed.':
                        k = 1
                if k == 0:
                    log[f"{tag}"].append(line)

f = open("file.txt", "w", encoding="UTF-16")
for lg in log:
    f.write(str(lg))
    for l in log[f"{lg}"]:
        f.write(f"\n{str(l)}")
    f.write("\n\n\n")
f.close()

# принимает словарь и переводит всю эту хуйню в векторы которые обрабатывает карина
model = SentenceTransformer("all-MiniLM-L6-v2")
input_dict = log
embeddings_dict = {}
# Обрабатываем каждую группу строк по ключу
for key, sentences in input_dict.items():
    embeddings = model.encode(sentences)  # Получаем массив эмбеддингов
    embeddings_dict[key] = embeddings.tolist()  # Сохраняем как список списков

# Вход: embeddings_dict = { "Файл: файл1.txt": [[...], [...]], ... }

pca_embeddings_dict = {}

# Сначала собираем все эмбеддинги для нормализации и PCA
all_embeddings = []
for sentences in embeddings_dict.values():
    all_embeddings.extend(sentences)

if len(all_embeddings) == 0:
    raise ValueError("Нет эмбеддингов для обработки. Проверьте входные данные.")

# Нормализация
scaler = StandardScaler()
all_embeddings_scaled = scaler.fit_transform(all_embeddings)

# Обучаем PCA
pca = PCA(n_components=128)
pca.fit(all_embeddings_scaled)

# Применяем ко всем элементам
for key, embeddings in embeddings_dict.items():
    if not embeddings:
        # Пропускаем пустые списки
        pca_embeddings_dict[key] = []
        continue

    # Нормализуем и применяем PCA
    scaled = scaler.transform([e for e in embeddings if len(e) > 0])  # фильтруем пустые эмбеддинги
    reduced = pca.transform(scaled)

    pca_embeddings_dict[key] = reduced.tolist()

f = open("ffffile.txt", "w",  encoding='utf-8')
f.write(str(pca_embeddings_dict ))
f.close()



