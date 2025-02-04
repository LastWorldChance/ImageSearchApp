import os
import time
import streamlit as st
import tempfile
from PIL import Image
from search_engine import AdvancedImageSearch


class StreamlitImageSearchApp:
    def __init__(self):
        self.search_engine = None
        self.initialized = False

    def initialize_engine(self, database_path, crop_ratio):
        self.search_engine = AdvancedImageSearch(database_path, query_crop_ratio=crop_ratio)
        self.search_engine.build_database()
        self.initialized = True

    def update_db(self):
        if self.search_engine:
            with st.spinner("Обновление базы..."):
                self.search_engine.build_database(force_rebuild=True)
            st.sidebar.success("База обновлена!")


app = StreamlitImageSearchApp()

st.set_page_config(page_title="Поиск изображений", layout="wide")
st.title("🔍 Поиск изображений")

with st.sidebar:
    st.header("Управление базой")
    database_path = st.text_input("Путь к базе", value=r"C:\Users\daole\Downloads\test_img")
    st.sidebar.button("🔄 Обновить базу данных", on_click=app.update_db)
    top_k = st.slider("Количество результатов", 1, 20, 5)
    crop_ratio = st.slider("Область обрезки", 0.5, 1.0, 0.9, 0.1)

if not app.initialized and os.path.exists(database_path):
    app.initialize_engine(database_path, crop_ratio)

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
if uploaded_file and app.initialized:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        query_path = tmp_file.name

    try:
        start_time = time.time()
        results = app.search_engine.search(query_path, top_k=top_k)
        search_time = time.time() - start_time

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(Image.open(query_path), caption="Запрос", use_container_width=True)
            st.caption(f"Время поиска: {search_time:.2f} сек")

        with col2:
            # Разбиваем результаты на группы по 5 элементов
            for i in range(0, len(results), 5):
                group = results[i:i + 5]
                cols = st.columns(5)  # Всегда создаем 5 колонок

                for idx, (col, result) in enumerate(zip(cols, group)):
                    path, score = result
                    with col:
                        img = Image.open(path)
                        filename = os.path.basename(path)
                        st.image(
                            img,
                            caption=f"#{i + idx + 1} | {filename} | Сходство: {score:.1f}%",
                            use_container_width=True
                        )
                        # Добавляем пустое пространство между рядами
                        st.write("")

    finally:
        os.unlink(query_path)
