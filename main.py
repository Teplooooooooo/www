print('''
-----------------------------------------
     Выполнение программы началось
-----------------------------------------    

''')

# организаторы буськи, кейс кайф
from requests import get
from re import compile, DOTALL, IGNORECASE
from sentence_transformers import SentenceTransformer
from torch import sum, no_grad, clamp
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def get_embeddings(model, tokenizer, sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with no_grad():
        model_output = model(**encoded_input)

    # Применяем mean pooling по токенам (ось 1)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = clamp(input_mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask
    return sentence_embeddings.numpy()



models_list = {
        1: "microsoft/codebert-base",
        2: "paraphrase-multilingual-mpnet-base-v2",
        3: "paraphrase-multilingual-MiniLM-L12-v2",
        4: "paraphrase-MiniLM-L3-v2",
        5: "all-mpnet-base-v2",
        6: "all-MiniLM-L6-v2",
        7: "all-MiniLM-L12-v2",
        8: "all-distilroberta-v1",
        9: "multi-qa-mpnet-base-dot-v1",
        10: "multi-qa-distilbert-cos-v1",
        11: "multi-qa-MiniLM-L6-cos-v1",
        12: "distiluse-base-multilingual-cased-v2",
        13: "distiluse-base-multilingual-cased-v1",
        14: "paraphrase-albert-small-v2"
    }

print("""
Выберите модель для работы.

Общие модели для семантической схожести:
-----------------------------------------
1  - microsoft/codebert-base (обучалась на коде, но работает долго)
2  - paraphrase-multilingual-mpnet-base-v2
3  - paraphrase-multilingual-MiniLM-L12-v2
4  - paraphrase-MiniLM-L3-v2
5  - all-mpnet-base-v2
6  - all-MiniLM-L6-v2 (работает максимально быстро, но неточно)
7  - all-MiniLM-L12-v2
8  - all-distilroberta-v1

Модели для вопросно-ответных систем:
-----------------------------------
9  - multi-qa-mpnet-base-dot-v1
10 - multi-qa-distilbert-cos-v1
11 - multi-qa-MiniLM-L6-cos-v1

Многоязычные модели:
---------------------
12 - distiluse-base-multilingual-cased-v2
13 - distiluse-base-multilingual-cased-v1
14 - paraphrase-albert-small-v2

15 - Выбрать свою модель основе BERT и RoBERTa

""")

while True:
    try:
        choice = int(input("Введите число, соответствующее выбранной модели."))
        break
    except:
        print("Введите именно целое число из списка!")

n_clusters = 30


check = 0
while check == 0:
    if choice == 1:
        name_of_model = models_list[choice]
        # Загружаем CodeBERT
        tokenizer = AutoTokenizer.from_pretrained(name_of_model)
        model = AutoModel.from_pretrained(name_of_model)         # размерность вектора - 768

        print("Модель загружена.")
        print("Запущенно преобразование ошибок в числа...")
        # Обрабатываем каждую группу строк по ключу
        embeddings_dict = {}

        check = 1
    elif 2 <= choice <= 14:
        name_of_model = models_list[choice]
        model = SentenceTransformer(name_of_model)

        print("Модель загружена.")
        print("Запущенно преобразование ошибок в числа...")

        check = 1
    elif choice == 15:
        while True:
            name_of_model = input("Введите название модели: ")
            try:
                tokenizer = AutoTokenizer.from_pretrained(name_of_model)
                model = AutoModel.from_pretrained(name_of_model)  # размерность вектора - 768
                break
            except:
                print('''
                Вы ввели не то название модели.
                Используйте модели на основе BERT и RoBERTa.
                ''')

        print("Модель загружена.")
        check = 1
    else:
        print("Вы ввели число, которого нет в списке")

print("""
Поиск ошибок запущен...
""")

# Вводим Url и парсим страницу
url = "https://git.altlinux.org/beehive/logs/Sisyphus/x86_64/latest/error/"

response = get(url)
responses = response.text.split()
log = {}
# Находим herf, добавляем в словарь ключ с названием проекта и значение пустой массив
build_error_pattern = compile(
    r'(E:|error:|ERROR|Error|failed|FAILED|Could NOT find|could not find):?\s*(.*?)(?=\n\S|\Z)',
    DOTALL | IGNORECASE)
for tag in responses[50:]:
    if tag[:6] == 'href="':
        tag = tag.split(sep='"')[1]
        log[f"{tag}"] = []
        log_content = get(f"{url}{tag}").text
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

print("Ошибки найдены. Запущена обработка ошибок.")

if choice in [1, 15]:
    # Обрабатываем каждую группу строк по ключу
    embeddings_dict = {}

    for key, sentences in log.items():
        if not sentences:
            embeddings_dict[key] = []
            continue

        embeddings = get_embeddings(model, tokenizer, sentences)
        embeddings_dict[key] = embeddings.tolist()
else:
    input_dict = log
    embeddings_dict = {}
    # Обрабатываем каждую группу строк по ключу
    for key, sentences in input_dict.items():
        embeddings = model.encode(sentences)  # Получаем массив эмбеддингов
        embeddings_dict[key] = embeddings.tolist()  # Сохраняем как список списков

# Функция для вычисления эмбеддингов с mean pooling

print("Обработка ошибок завершена.")
print("Запущена оптимизация...")

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
pca = PCA(n_components=64)   # меняем размерность вектора если надо
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

print("Оптимизация завершена")

# Основной поток выполнения
vectors = []
filenames = []
for filename, vectors_list in pca_embeddings_dict.items():
    for vec in vectors_list:
        vectors.append(vec)
        filenames.append(filename)


scaler = StandardScaler()
scaled_vectors = scaler.fit_transform(vectors)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_vectors)

clusters = {}
for label, filename in zip(kmeans.labels_, filenames):
    if label not in clusters:
        clusters[label] = set()
    clusters[label].add(filename)

for cluster_id, files in clusters.items():
    print(f"Cluster {cluster_id}:")
    for f in files:
        print(f"    {f}")
    print()
print("""
Работа программы завершена
""")





