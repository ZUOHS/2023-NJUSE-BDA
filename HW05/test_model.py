import json
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
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
test_file_path = './test.json'  # 替换为测试集的JSON文件路径
test_data = load_data_from_json(test_file_path)

# 提取文本和标签
test_headlines, test_authors, test_descriptions, test_dates, test_link, test_labels = extract_text_and_labels(test_data)

# 加载训练好的向量化器
vectorizer_headline = joblib.load('./models/vectorizer_headline.pkl')  # 替换为训练好的headline向量化器路径
vectorizer_authors = joblib.load('./models/vectorizer_authors.pkl')  # 替换为训练好的authors向量化器路径
vectorizer_descriptions = joblib.load('./models/vectorizer_descriptions.pkl')  # 替换为训练好的descriptions向量化器路径
vectorizer_dates = joblib.load('./models/vectorizer_dates.pkl')  # 替换为训练好的dates向量化器路径
vectorizer_link = joblib.load('./models/vectorizer_link.pkl')  # 替换为训练好的link向量化器路径

# 特征提取
X_test_headline = vectorizer_headline.transform(test_headlines)
X_test_authors = vectorizer_authors.transform(test_authors)
X_test_descriptions = vectorizer_descriptions.transform(test_descriptions)
X_test_dates = vectorizer_dates.transform(test_dates)
X_test_link = vectorizer_link.transform(test_link)

# 合并特征
X_test = hstack([X_test_headline, X_test_authors, X_test_descriptions, X_test_dates, X_test_link])

model_file = './models/model.pkl'  # 替换为实际的模型文件路径
loaded_model = joblib.load(model_file)

# 预测
y_pred = loaded_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(test_labels, y_pred)
count = int(accuracy * len(y_pred))

# 保存预测结果到JSON文件
output_file = './predictions.json'  # 替换为实际的输出文件路径
output_data = []
for i, prediction in enumerate(y_pred):
    record = {
        "headline": test_headlines[i],
        "predict_category": prediction,
        "real_category": test_labels[i],
    }
    output_data.append(record)
    print(record)

with open(output_file, 'w') as f:
    for record in output_data:
        f.write(json.dumps(record) + '\n')
    print("正确结果：", count, "正确率：", accuracy)
    record = {
        "Correct:": str(count),
        "Accuracy:": str(accuracy),
    }
    f.write(json.dumps(record) + '\n')

print("预测结果已保存到", output_file)
