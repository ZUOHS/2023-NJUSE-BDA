import json
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 从JSON文件中加载数据
def load_data_from_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # 去除行首和行尾的空白字符
            if not line:  # 跳过空行
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                print("Error loading JSON:", e)
    return data

# 提取文本和标签
def extract_text_and_labels(data):
    headlines = [item['headline'] for item in data]
    authors = [item['authors'] for item in data]
    descriptions = [item['short_description'] for item in data]
    dates = [item['date'] for item in data]
    link = [item['link'] for item in data]
    labels = [item['category'] for item in data]
    return headlines, authors, descriptions, dates, link, labels

# 加载数据
train_file_path = './data.json'  # 替换为训练集的JSON文件路径
train_data = load_data_from_json(train_file_path)

# 提取文本和标签
train_headlines, train_authors, train_descriptions, train_dates, train_link, train_labels = extract_text_and_labels(train_data)

# 特征提取
vectorizer_headline = TfidfVectorizer()
X_train_headline = vectorizer_headline.fit_transform(train_headlines)
joblib.dump(vectorizer_headline, './models/vectorizer_headline.pkl')

vectorizer_authors = TfidfVectorizer()
X_train_authors = vectorizer_authors.fit_transform(train_authors)
joblib.dump(vectorizer_authors, './models/vectorizer_authors.pkl')

vectorizer_descriptions = TfidfVectorizer()
X_train_descriptions = vectorizer_descriptions.fit_transform(train_descriptions)
joblib.dump(vectorizer_descriptions, './models/vectorizer_descriptions.pkl')

vectorizer_dates = TfidfVectorizer()
X_train_dates = vectorizer_dates.fit_transform(train_dates)
joblib.dump(vectorizer_dates, './models/vectorizer_dates.pkl')

vectorizer_link = TfidfVectorizer()
X_train_link = vectorizer_link.fit_transform(train_link)
joblib.dump(vectorizer_link, './models/vectorizer_link.pkl')

# 合并特征
X_train = hstack([X_train_headline, X_train_authors, X_train_descriptions, X_train_dates, X_train_link])

# 创建朴素贝叶斯分类器实例并指定alpha参数
alpha = 0.027  # 调整alpha的值
classifier = MultinomialNB(alpha=alpha)

# 训练模型
classifier.fit(X_train, train_labels)

model_file = './models/model.pkl'  # 替换为实际的模型文件路径
joblib.dump(classifier, model_file)
print("模型已保存到", model_file)
