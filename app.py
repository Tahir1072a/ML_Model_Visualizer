import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, classification_report

# --- SAYFA AYARLARI ---
st.set_page_config(layout="wide")
st.title("ML Classification Algorithms Compare")
st.write("""
Bu interaktif araç, farklı makine öğrenmesi sınıflandırma algoritmalarının performansını ve karar sınırlarını
çeşitli sentetik veri setleri üzerinde görsel olarak karşılaştırmanıza olanak tanır. Algoritma skorları kutu içerisinde grafiğin sağ alt köşesinde yazmaktadır.
Hyperparametreler ile oynayarak doha iyi sonuçlar elde etmeye çalışabilirsiniz!
""")

# --- YAN PANEL (SIDEBAR) ---
st.sidebar.header("Parameter Settings")

with st.sidebar.expander("1. Veri Seti Ayarları", expanded=True):
    dataset_name = st.sidebar.selectbox(
        "Bir veri seti seçin:",
        ("Lineer Distributions", "Moons Distributions", "Circles Distributions"),
    )
    noise = st.sidebar.slider("Veri setine gürültü ekleyin:", 0.0, 1.0, 0.0, 0.05)
    n_samples = st.sidebar.slider("Örnek Sayısı:", 100, 1000, 100, 50)

st.sidebar.header("Model Hiperparametreleri")
with st.sidebar.expander("K-Nearest Neighbors"):
    knn_neighbors = st.sidebar.slider("Neighbors (k)", 1, 15, 3)

with st.sidebar.expander("Support Vector Machine (SVM)"):
    svm_c = st.sidebar.number_input("Regularization (C)", 0.01, 10.0, 1.0, 0.01)

with st.sidebar.expander("Decision Tree & Random Forest"):
    max_depth = st.sidebar.slider("Max Depth", 2, 15, 5)
    n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", 10, 100, 10, 5)


# --- VERİ SETİ OLUŞTURMA ---
def get_dataset(name, noise, n_samples):
    if name == "Lineer Distributions":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
            random_state=42, n_clusters_per_class=1
        )
        if noise == 0:
            return X, y
        else:
            X += (1 + noise) * np.random.uniform(size=X.shape)
        return X, y
    elif name == "Moons Distributions":
        return make_moons(n_samples=n_samples, noise=noise, random_state=42)
    else:  # Circles Distributions
        return make_circles(n_samples=n_samples, noise=noise, factor=0.3, random_state=42)


X, y = get_dataset(dataset_name, noise, n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SINIFLANDIRICILAR (Sidebar'daki değerlerle dinamik olarak oluşturuluyor) ---
names = [
    "Nearest Neighbors", "Linear SVM", "Decision Tree",
    "Random Forest", "Naive Bayes"
]
classifiers = [
    KNeighborsClassifier(n_neighbors=knn_neighbors),
    SVC(kernel="linear", C=svm_c),
    DecisionTreeClassifier(max_depth=max_depth),
    RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators),
    GaussianNB(),
]

# --- GÖRSELLEŞTİRME ---
st.header(f"'{dataset_name}' veri seti için elde edilen sonuçlar")
cols_1 = st.columns(2)

# Önce orijinal veri setini gösterelim
fig, ax = plt.subplots()
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", label="Class A")
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k", label="Class B")
ax.set_title("Orijinal Veri Dağılımı")
ax.set_xticks(())
ax.set_yticks(())
ax.legend()
with cols_1[0]:
    st.pyplot(fig)

with cols_1[1]:
    st.subheader("Veri Seti Detayları")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Toplam Örnek", value=X.shape[0])
    metric_cols[1].metric("Eğitim Verisi", value=X_train.shape[0])
    metric_cols[2].metric("Test Verisi", value=X_test.shape[0])

    st.write(f"**Sınıf Sayısı:** {len(np.unique(y))}")
    st.write(f"**Gürültü Seviyesi:** {noise:.2f}")

    st.markdown("---")
    if dataset_name == "Lineer Distributions":
        st.info("""
        **Karakteristik:** Bu veri seti, bir doğru ile büyük ölçüde ayrılabilen iki sınıf içerir. 
        Lineer modellerin (örneğin Linear SVM) bu tür problemlerde başarılı olması beklenir.
        """)
    elif dataset_name == "Moons Distributions":
        st.info("""
        **Karakteristik:** Veri setleri, iç içe geçmiş ay şeklindeki iki kümeden oluşur. 
        Bu, lineer olmayan karar sınırları gerektiren klasik bir problemdir.
        """)
    else:  # Circles Distributions
        st.info("""
        **Karakteristik:** Biri diğerinin içinde olan iki dairesel veri kümesinden oluşur. 
        Bu problem de lineer olmayan, dairesel bir karar sınırı çizebilen algoritmalar gerektirir.
        """)


st.markdown("---")

cols = st.columns(3)

for i, (name, clf) in enumerate(zip(names, classifiers)):
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    with cols[i % 3]:
        fig, ax = plt.subplots(figsize=(5, 5))

        DecisionBoundaryDisplay.from_estimator(
            model, X, cmap=plt.cm.RdBu, ax=ax, alpha=0.8, response_method="predict"
        )

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

        ax.set_title(f"{name}")
        ax.set_xticks(())
        ax.set_yticks(())

        ax.text(
            0.95, 0.05, f"{score:.2f}",
            transform=ax.transAxes, fontsize=12, horizontalalignment="right",
            verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7)
        )
        st.pyplot(fig)

        # Classification Report'u oluştur ve göster
        st.write(f"**{name} Raporu:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report, use_container_width=True)
        st.markdown("---")