import os
import tensorflow as tf
import numpy as np
import faiss
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
import time


class AdvancedImageSearch:
    def __init__(self, database_dir, query_crop_ratio=0.9):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.database_dir = database_dir
        self.database_crop_ratio = 0.9  # Фиксированное значение для базы
        self.query_crop_ratio = query_crop_ratio  # Для запросов
        self.index = None
        self.image_paths = []
        self.features_file = os.path.join(database_dir, "features.faiss")
        self.meta_file = os.path.join(database_dir, "meta.pkl")

    def preprocess(self, img_path, crop_ratio=None):
        """Обработка изображения с указанным crop_ratio"""
        if crop_ratio is None:
            crop_ratio = self.query_crop_ratio

        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)

        # Центральная обрезка
        h, w = img.shape[:2]
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        img = img[(h - new_h) // 2:(h + new_h) // 2, (w - new_w) // 2:(w + new_w) // 2]

        # Ресайз с сохранением пропорций
        img = tf.image.resize_with_pad(img, 224, 224)
        return preprocess_input(img)

    def build_database(self, force_rebuild=False):
        """Построение/обновление базы с инкрементальным обновлением"""
        if force_rebuild:
            self._full_rebuild()
            return

        if not self._load_cached_data():
            self._full_rebuild()
            return

        # Проверяем существование всех файлов в image_paths
        missing_files = [p for p in self.image_paths if not os.path.exists(p)]
        if missing_files:
            print(f"Found {len(missing_files)} missing files, rebuilding...")
            self._full_rebuild()
            return

        # Получаем текущий список файлов
        current_files = self._get_image_list()

        # Определяем новые файлы
        current_files_set = set(current_files)
        existing_set = set(self.image_paths)
        new_files = [f for f in current_files if f not in existing_set]

        if not new_files:
            print("No new files to add.")
            return

        print(f"Found {len(new_files)} new files. Adding to database...")

        # Извлекаем фичи новых файлов
        new_features = self._extract_features(new_files)

        # Нормализуем и добавляем в индекс
        faiss.normalize_L2(new_features)
        self.index.add(new_features.astype('float32'))

        # Обновляем список путей
        self.image_paths += new_files

        # Сохраняем обновленные данные
        self._save_data()
        print(f"Added {len(new_files)} new images. Total: {len(self.image_paths)}")

    def _full_rebuild(self):
        """Полная перестройка базы"""
        file_list = self._get_image_list()
        features = self._extract_features(file_list)

        self.index = faiss.IndexFlatIP(features.shape[1])
        faiss.normalize_L2(features)
        self.index.add(features.astype('float32'))

        self.image_paths = file_list
        self._save_data()
        print(f"Database rebuilt. Total images: {len(file_list)}")

    def _get_image_list(self):
        """Получаем отсортированный список изображений"""
        return sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(self.database_dir)
            for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def _extract_features(self, file_list):
        """Извлечение фич с использованием database_crop_ratio"""
        batch_size = 32
        features = []

        for i in range(0, len(file_list), batch_size):
            batch_paths = file_list[i:i + batch_size]
            batch_images = [self.preprocess(p, self.database_crop_ratio) for p in batch_paths]
            batch_features = self.model.predict(np.array(batch_images), verbose=0)
            features.append(batch_features)
            print(f"Processed {i + len(batch_images)}/{len(file_list)} images")

        features = np.vstack(features)
        return features.astype('float32')

    def _save_data(self):
        """Сохранение данных в файлы"""
        if os.path.exists(self.features_file):
            os.remove(self.features_file)
        if os.path.exists(self.meta_file):
            os.remove(self.meta_file)

        faiss.write_index(self.index, self.features_file)
        with open(self.meta_file, 'wb') as f:
            pickle.dump({
                'paths': self.image_paths,
                'timestamp': time.time(),
                'version': 3
            }, f)

    def _load_cached_data(self):
        """Загрузка кэшированных данных"""
        try:
            with open(self.meta_file, 'rb') as f:
                meta = pickle.load(f)
                if meta.get('version', 1) != 3:
                    return False

            self.index = faiss.read_index(self.features_file)
            self.image_paths = meta['paths']
            print(f"Loaded cached database with {len(self.image_paths)} images")
            return True
        except Exception as e:
            print(f"Cache loading failed: {str(e)}")
            return False

    def search(self, query_path, top_k=5):
        """Поиск с использованием query_crop_ratio"""
        query_img = self.preprocess(query_path)
        query_feat = self.model.predict(np.expand_dims(query_img, 0), verbose=0)[0].astype('float32')

        faiss.normalize_L2(query_feat.reshape(1, -1))

        if not self.index:
            raise ValueError("Index not initialized")

        distances, indices = self.index.search(query_feat.reshape(1, -1), top_k)

        scores = np.clip(distances[0] * 100, 0, 100)
        return [(self.image_paths[i], float(scores[j])) for j, i in enumerate(indices[0])]