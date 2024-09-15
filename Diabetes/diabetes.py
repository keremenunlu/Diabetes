import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform # geniş ve eşdağılımlı bir hyperparameter değer aralığı tanımlamak için kullanılır
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


dir_path = r"C:\Users\u47455\Desktop\diabetesFile"
file_name = "diabetes.csv"
file_path = os.path.join(dir_path, file_name)

# KAYIP VERİ GÖSTERİMİ
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(data.isna().sum()) # bunun sonucu her column için 0 geliyor, eksik değerimiz bulunmamakta
    return data

# VERİ DAĞILIMI GÖSTERİMİ
def data_distribution(data):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15)) # 9 kategoriyi 3'e 3 böldük
    axes = axes.flatten() # subplotları tek boyuta indirip, kolay iterate etmek için
    columns = data.columns

    for ax, column in zip(axes, columns): # her kategori için ayrı bir graf oluşumu
        ax.hist(data[column], bins=20, color='blue', edgecolor='black') # ax = subplots içindeki her bir subplot bir ax.
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
    plt.show()

# KORELASYON
def correlation_matrix(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Diabetes Heatmap', fontsize=16)
    plt.show()

# HER KATEGORİNİN SONUÇ İLE DOĞRUDAN İLİŞKİSİ
def feature_vs_outcome(data):
    df = data
    df = df.drop("Outcome", axis=1)
    cols = df.columns

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, col in enumerate(cols): # loop'a birden fazla variable sokarken enumerate güvenli iteration sağlar.
        sns.boxplot(y=df[col], x=data["Outcome"], ax=axes[i])
        axes[i].set_title(f"{col} vs Outcome")
    plt.tight_layout()
    plt.show() # FİGÜRE BAKILDIĞINDA BMI, YAŞ, HAMİLELİK VE GLUCOSE DIŞINDAKİ DEĞERLERDE
    # DOĞRUDAN BİR BAĞLANTI VEYA İLİŞKİ BULUNMUYOR. YANİ DİYABET SONUCU VERİLEN KATEGORİLERE DOĞRUDAN BAĞLI DEĞİL.

# MODEL EVALUATION
def train_model(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)  # training data ile modelin çalışması
    y_pred = model.predict(x_test)  # x_test'i kullanarak y'nin tahmini değerini bulma (modeli train değerleriyle çalıştır, daha sonra test değeriyle prediction yap, görülmemiş veri performansını görmek için)
    return y_pred

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)  # gerçek y değeri ile tahmini olanın ilişkisiyle accuracy ölçümü
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

    # precision = tahminlerin tutarlılığını (doğruluğunu) ölçer
    # recall = tüm pozitif tahminlerin ne kadarının pozitif olarak tahmin edildiğini gösterir
    # F1 = precision ve recall değerlerinin dengeli bir ortalama göstergesidir

def hyperparameter_tuning(model, param_grid, x_train, y_train, x_test, y_test, search_type):
    if search_type == "grid":
        # Hyperparameter tuning (GRID SEARCH CV): process of determining the combination of hyperparameters to maximize the model's performance
        search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy')
    else:
        # Hyperparameter tuning (RANDOMIZED SEARCH CV): randomly sampling from a distribution of parameters
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='accuracy', random_state=42, n_iter=10, cv=5)

    search.fit(x_train, y_train)  # train verilerinin çalıştırılması
    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)
    return search.best_params_, evaluate_model(y_test, y_pred)

data = load_data(file_path)
data_distribution(data)
correlation_matrix(data)
feature_vs_outcome(data)

x = data.drop("Outcome", axis=1) # features
y = data["Outcome"] # target

scaler = StandardScaler() # scales to unit variance, makes mean 0, helps all features to contribute equally
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42) # 80 - 20 train-test olarak ayırma işlemi

# CV : Veriyi 80-20 olarak 5'e böldük, her parçayı ayrı ayrı test ettik.

# C: Regularization: Overfitting'i azaltmak için, modelin complexity'sini azaltır, test verisi performansını iyi tutar.
models = {
    "LogisticRegression": (LogisticRegression(random_state=42), {"C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs", "saga"]}),
    "DecisionTreeClassifier": (DecisionTreeClassifier(random_state=42), {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}), # min amount of samples to split a node
    "RandomForestClassifier": (RandomForestClassifier(random_state=42), {"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}),
    "GradientBoostingClassifier": (GradientBoostingClassifier(random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}),
    "SVC": (SVC(random_state=42), {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["linear", "rbf", "poly"]})
}

#LogReg : Binary, Linear Classifier
#DecTree : Division of space into seperate rectangles, and assigning them labels
#RandFor : Made up of multiple DecTrees, n_estimators = number of DecTrees
#GradBoost : Adds models sequentially to better the overall performance, learning_rate = the contribution of each element to the whole
#SVC : Finding the best hyperplane that seperates the features optimally

for model_name, (model, param_grid) in models.items():
    print(f"\nResults for {model_name}...")
    for search_type in ["grid", "random"]:
        print(f"\n{search_type.capitalize()} Search:")
        best_params, (accuracy, precision, recall, f1) = hyperparameter_tuning(model, param_grid, x_train, y_train, x_test, y_test, search_type)
        print(f"Best parameters: {best_params}")
        print(f"Accuracy: {accuracy:.5f}")
        print(f"Precision: {precision:.5f}")
        print(f"Recall: {recall:.5f}")
        print(f"F1 Score: {f1:.5f}")

# SearchCV modelleri, classification modellerinin optimal sonucu vermesini sağlar.
