# Laporan Proyek Machine Learning - Muhammad Salman Alfarisy

## Project Overview

Analisis ini berfokus pada sektor hiburan dengan tujuan mengembangkan sebuah model _machine learning_ yang dapat diaplikasikan sebagai sistem rekomendasi anime. Model tersebut dirancang untuk memberikan rekomendasi anime yang sesuai dengan preferensi pengguna berdasarkan data yang dianalisis.

Pengembangan sistem rekomendasi menjadi penting di era digital ini karena banyaknya pilihan konten yang tersedia. Pengguna seringkali kesulitan menemukan anime baru yang sesuai dengan selera mereka di antara ribuan judul yang ada. Sistem rekomendasi yang efektif dapat meningkatkan pengalaman pengguna dengan menyajikan pilihan yang relevan, personal, dan menarik. Proyek ini bertujuan untuk mengatasi masalah tersebut dengan membangun model yang mampu mempelajari preferensi pengguna dan karakteristik anime untuk memberikan rekomendasi yang akurat.

Referensi yang mendukung pentingnya sistem rekomendasi dalam platform hiburan:

- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. In Recommender Systems Handbook (pp. 1-35). Springer, Boston, MA. (Menjelaskan dasar-dasar dan pentingnya sistem rekomendasi)
- Gomez-Uribe, C. A., & Hunt, N. (2016). The netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 1-19. (Studi kasus tentang dampak sistem rekomendasi pada platform besar)

---

## Business Understanding

Bagian ini menjelaskan proses klarifikasi masalah yang ingin diselesaikan oleh proyek ini.

### Problem Statements

- Bagaimana cara membangun sistem yang dapat merekomendasikan anime kepada pengguna berdasarkan kemiripan konten (misalnya, genre)?
- Bagaimana cara membangun sistem yang dapat merekomendasikan anime kepada pengguna berdasarkan pola preferensi dari pengguna lain (_collaborative filtering_)?
- Bagaimana mengevaluasi performa model rekomendasi yang dibangun untuk memastikan relevansi dan akurasi rekomendasi?

### Goals

- Mengembangkan model _Content-Based Filtering_ yang merekomendasikan anime berdasarkan kesamaan genre.
- Mengembangkan model _Collaborative Filtering_ yang merekomendasikan anime berdasarkan riwayat rating pengguna dan kemiripan dengan pengguna lain.
- Menganalisis dan membandingkan metrik evaluasi seperti _Mean Absolute Error_ (MAE) dan _Root Mean Squared Error_ (RMSE) untuk model _Collaborative Filtering_.

### Solution Approach

- **Solution Statement 1 (Content-Based Filtering):** Menggunakan teknik TF-IDF (_Term Frequency-Inverse Document Frequency_) pada genre anime untuk membuat representasi vektor dari setiap anime. Kemudian, menghitung _cosine similarity_ antar anime untuk menemukan anime yang paling mirip berdasarkan genrenya.
- **Solution Statement 2 (Collaborative Filtering):** Mengimplementasikan model _neural network_ dengan _embedding layers_ untuk pengguna dan anime. Model ini akan mempelajari _latent features_ dari interaksi pengguna-anime (rating) dan memprediksi rating yang mungkin diberikan pengguna untuk anime yang belum ditonton.

---

##  Data Understanding

Proyek ini menggunakan dua dataset utama yang diperoleh dari Kaggle:

1. **Dataset Anime:** [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)

   - Berisi informasi metadata dari 17.562 judul anime.
   - Memiliki total 35 fitur (kolom).

2. **Dataset Rating & Review:** [MyAnimeList Dataset Animes Profiles Reviews](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews)

   - Berisi 192.112 ulasan dan rating pengguna terhadap anime.
   - Memiliki total 7 fitur.
   - Setelah dilakukan pembersihan dan penggabungan, diperoleh 130.488 interaksi unik yang digunakan untuk pelatihan model _collaborative filtering_.

### 📑 Deskripsi Fitur Dataset

#### 🔹 Dataset `anime.csv` (35 fitur)

| Fitur                    | Deskripsi                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------- |
| `MAL_ID`                 | ID unik dari MyAnimeList untuk setiap anime                                           |
| `Name`                   | Judul utama anime                                                                     |
| `Score`                  | Skor rata-rata dari pengguna (string numerik)                                         |
| `Genres`                 | Daftar genre anime (dipisahkan koma)                                                  |
| `English name`           | Judul dalam Bahasa Inggris                                                            |
| `Japanese name`          | Judul dalam Bahasa Jepang                                                             |
| `Type`                   | Tipe anime (TV, Movie, OVA, dll)                                                      |
| `Episodes`               | Jumlah episode                                                                        |
| `Aired`                  | Periode penayangan                                                                    |
| `Premiered`              | Musim dan tahun anime pertama kali tayang                                             |
| `Producers`              | Daftar produser yang terlibat                                                         |
| `Licensors`              | Penerbit lisensi untuk distribusi                                                     |
| `Studios`                | Studio yang memproduksi anime                                                         |
| `Source`                 | Sumber cerita (Original, Manga, Light Novel, dll)                                     |
| `Duration`               | Durasi rata-rata per episode                                                          |
| `Rating`                 | Batasan usia penonton (misalnya PG-13, R-17+)                                         |
| `Ranked`                 | Ranking anime berdasarkan skor                                                        |
| `Popularity`             | Ranking berdasarkan jumlah pengguna yang menyimpan                                    |
| `Members`                | Total pengguna yang menambahkan ke daftar                                             |
| `Favorites`              | Total pengguna yang memfavoritkan                                                     |
| `Watching`               | Pengguna yang sedang menonton                                                         |
| `Completed`              | Pengguna yang telah selesai menonton                                                  |
| `On-Hold`                | Pengguna yang menunda menonton                                                        |
| `Dropped`                | Pengguna yang berhenti menonton                                                       |
| `Plan to Watch`          | Pengguna yang berniat menonton                                                        |
| `Score-10` s/d `Score-1` | Distribusi skor yang diberikan pengguna (jumlah pengguna per nilai skor dari 10 ke 1) |

> Total 11 kolom terakhir (`Score-10` s.d. `Score-1`) menggambarkan distribusi frekuensi skor yang diberikan oleh pengguna.

#### 🔹 Dataset `reviews.csv` (7 fitur)

| Fitur       | Deskripsi                                                                                |
| ----------- | ---------------------------------------------------------------------------------------- |
| `uid`       | ID unik dari review                                                                      |
| `profile`   | Nama pengguna pemberi ulasan                                                             |
| `anime_uid` | ID anime yang direview (relasi ke `MAL_ID`)                                              |
| `text`      | Isi ulasan dalam bentuk teks panjang                                                     |
| `score`     | Skor keseluruhan dari pengguna terhadap anime                                            |
| `scores`    | Dictionary berisi skor per aspek: Overall, Story, Animation, Sound, Character, Enjoyment |
| `link`      | Tautan ke halaman review di MyAnimeList                                                  |

### 🔍 Kondisi Data dan Kualitas

#### Missing Values dan Data Duplikat
- **Dataset Anime:** Tidak ditemukan missing values yang signifikan pada kolom-kolom utama yang digunakan (MAL_ID, Name, Genres).
- **Dataset Reviews:** Tidak ditemukan missing values pada kolom score dan anime_uid yang merupakan kolom kunci.
- **Data Duplikat:** Setelah penggabungan dataset, ditemukan 61.584 duplikat rating dari pengguna yang sama untuk anime yang sama. Duplikat ini kemudian dihapus untuk memastikan setiap interaksi pengguna-anime adalah unik.

### 🔍 Fitur yang Digunakan untuk Pemodelan

Setelah proses _preprocessing_, fitur utama berikut dipilih untuk masing-masing pendekatan:

#### Untuk **Content-Based Filtering** (dari `anime.csv`):

- `anime_id` (alias dari `MAL_ID`)
- `name`
- `Genres`
- `Score` (digunakan sebagai fitur numerik tambahan)
- `Popularity` dan `Members` (opsional untuk penyusunan bobot konten)

#### Untuk **Collaborative Filtering** (dari `reviews.csv`):

- `user_id` (dari `uid`)
- `anime_id` (dari `anime_uid`)
- `rating` (dari `score`, dinormalisasi ke rentang 0.0–1.0)

### 📈 Visualisasi dan EDA

**Distribusi Genre Anime:**

- Terdapat 44 genre unik.
- Genre seperti **Comedy**, **Action**, dan **Fantasy** paling banyak ditemukan.
- Genre yang lebih jarang muncul: **Yuri**, **Harem**, **Samurai**.

![Distribusi Sebaran Genre Anime](https://github.com/user-attachments/assets/e8ba2784-7513-4ec1-8b00-e42031e67168)

**Distribusi Anime Berdasarkan Rating Terpopuler:**

- Anime seperti _Sword Art Online_, _Death Note_, dan _Steins;Gate_ memiliki jumlah ulasan tertinggi.
- Menunjukkan bias popularitas yang dapat memengaruhi hasil model jika tidak ditangani dengan benar.

![Distribusi Sebaran Rating dari 10 Anime dengan Jumlah Rating Terbanyak](https://github.com/user-attachments/assets/1a7ed2f5-5f66-4c05-b2af-0dba079944fb)

### 📌 Statistik Dataset untuk Collaborative Filtering

| Informasi                          | Nilai      |
| ---------------------------------- | ---------- |
| Total interaksi unik               | 130.488    |
| Jumlah pengguna unik               | 130.488    |
| Jumlah anime unik                  | 8.107      |
| Rentang rating awal                | 0.0 – 11.0 |
| Rentang rating setelah normalisasi | 0.0 – 1.0  |

---

## Data Preparation

Tahapan persiapan data yang dilakukan meliputi:

1.  **Environment Setup & Dependencies:**

    - Menginstal dan mengimpor _library_ yang diperlukan seperti TensorFlow, Keras, Scikit-learn, Surprise, Pandas, Numpy, Matplotlib, Seaborn, dan Bokeh.
    - Melakukan reinstalasi `numpy` dan `scikit-surprise` untuk menghindari konflik versi di Google Colab.
    - Hasil: Semua _library_ berhasil diinstal dan dikonfigurasi.

2.  **Secure Kaggle API Setup:**

    - Mengkonfigurasi Kaggle API _credentials_ secara aman untuk mengunduh dataset.
    - Memastikan file `kaggle.json` berada di lokasi yang tepat dengan izin akses yang benar.
    - Hasil: Kaggle API berhasil dikonfigurasi.

3.  **Smart Data Loading with KaggleHub:**

    - Mengunduh dataset anime dan rating secara otomatis menggunakan `kagglehub`.
    - Memuat dataset ke dalam Pandas DataFrame dengan validasi ukuran data.
    - Hasil: Dataset anime (17.562 baris) dan dataset _reviews_ (192.112 baris) berhasil diunduh dan dimuat.

4.  **Genre Exploration and One-Hot Encoding (untuk _Content-Based_):**

    - Mengekstrak genre individual dari kolom `Genres` yang berupa _string_ gabungan.
    - Melakukan _one-hot encoding_ untuk mengubah data kategorikal genre menjadi format numerik biner.
    - Menghitung total kemunculan setiap genre.
    - Hasil: Berhasil mengekstrak 44 genre unik dan membuat representasi _one-hot encoded_.

5.  **TF-IDF Vectorization (untuk _Content-Based_):**

    - Mengubah teks genre menjadi vektor numerik menggunakan TF-IDF _vectorizer_.
    - Alasan: TF-IDF membantu memberikan bobot pada genre yang lebih penting dan spesifik untuk sebuah anime.
    - Hasil: TF-IDF _matrix_ dengan _shape_ (17562, 47) berhasil dibuat.

6.  **Cosine Similarity Matrix (untuk _Content-Based_):**

    - Menghitung kesamaan kosinus antar anime berdasarkan vektor TF-IDF genre.
    - Alasan: _Cosine similarity_ adalah metrik yang umum digunakan untuk mengukur kesamaan antar item dalam sistem rekomendasi berbasis konten.
    - Hasil: _Cosine similarity matrix_ dengan _shape_ (17562, 17562) berhasil dihitung.

7.  **Data Merging and Cleaning (untuk _Collaborative Filtering_):**

    - Menyeragamkan nama kolom (`MAL_ID` menjadi `anime_id`, `uid` menjadi `user_id`, `score` menjadi `rating`).
    - Menggabungkan dataset anime dan rating berdasarkan `anime_id`.
    - Memeriksa dan menangani _missing values_ (tidak ditemukan _missing values_).
    - Menghapus duplikasi rating dari pengguna yang sama untuk anime yang sama (ditemukan dan dihapus 61.584 duplikat).
    - Alasan: Langkah-langkah ini penting untuk memastikan konsistensi, kualitas, dan integritas data sebelum dimasukkan ke model.
    - Hasil: Dataset gabungan `rating_anime` dengan 130.488 interaksi unik berhasil dibuat.

8.  **ID Encoding (untuk _Collaborative Filtering_):**

    - Mengubah `user_id` dan `anime_id` menjadi _sequential integers_ (0 hingga N-1).
    - Membuat mapping dictionary untuk encoding: `user2user_encoded` dan `anime2anime_encoded`.
    - Alasan: Model _neural network_ (khususnya _embedding layers_) memerlukan input berupa integer sekuensial.
    - Hasil: `user_id` dan `anime_id` berhasil di-_encode_ dengan total 130.488 pengguna unik dan 8.107 anime unik.

9.  **Data Type Conversion:**

    - Mengubah tipe data kolom `rating` menjadi `float32` untuk optimalisasi memori dan komputasi.
    - Alasan: `float32` memberikan presisi yang cukup untuk rating sambil menghemat penggunaan memori dibandingkan `float64`.
    - Hasil: Kolom `rating` berhasil dikonversi ke tipe data `float32`.

10. **Rating Normalization (untuk _Collaborative Filtering_):**

    - Menganalisis distribusi rating (ditemukan rentang 0.0 - 11.0).
    - Menormalisasi nilai `rating` ke rentang [0, 1] menggunakan min-max scaling.
    - Alasan: Normalisasi membantu dalam stabilitas proses _training_ model _neural network_ dan konsistensi _output_ dari fungsi aktivasi sigmoid.
    - Rumus: `rating_normalized = (rating - min_rating) / (max_rating - min_rating)`
    - Hasil: Kolom `rating_normalized` berhasil dibuat dengan rentang 0.0-1.0.

11. **Train-Test Split (untuk _Collaborative Filtering_):**
    - Mengacak urutan data.
    - Memisahkan fitur (`user`, `anime`) dan target (`rating_normalized`).
    - Membagi data menjadi data latih (80%) dan data validasi (20%).
    - Alasan: Untuk melatih model pada sebagian data dan mengevaluasi kinerjanya pada data yang belum pernah dilihat.
    - Hasil: Data latih (104.390 sampel) dan data validasi (26.098 sampel) siap digunakan.

---

## Modeling

Dua pendekatan model sistem rekomendasi dikembangkan dalam proyek ini:

### 1. Content-Based Filtering

Pendekatan ini merekomendasikan anime berdasarkan kesamaan atribut konten, khususnya genre.

- **Proses:**

  1.  Genre setiap anime direpresentasikan sebagai vektor menggunakan TF-IDF.
  2.  Kesamaan antar anime dihitung menggunakan _cosine similarity_ pada matriks TF-IDF tersebut.
  3.  Sebuah fungsi `AnimeRecommendations` dibuat untuk mengambil _K_ anime teratas yang paling mirip dengan anime input berdasarkan skor kesamaan genre.

- **Implementasi Fungsi Rekomendasi:**

  ```python
  def AnimeRecommendations(anime_name, similarity_data=cosine_sim_df, items=anime_df[['MAL_ID','Name','Genres']], k=10):
      index = similarity_data.loc[:, anime_name].to_numpy().argpartition(range(-1, -k, -1))
      closest = similarity_data.columns[index[-1:-(k+2):-1]]
      closest = closest.drop(anime_name, errors='ignore') # Hindari merekomendasikan anime itu sendiri
      return pd.DataFrame(closest).merge(items).head(k)
  ```

- **Kelebihan:**
  - Dapat merekomendasikan item baru (_cold start problem_ untuk item) selama item tersebut memiliki deskripsi fitur (genre).
  - Tidak memerlukan data rating dari pengguna lain.
  - Rekomendasi transparan dan mudah dijelaskan (berdasarkan kesamaan genre).
- **Kekurangan:**
  - Terbatas pada fitur yang digunakan (hanya genre dalam kasus ini), sehingga kurang bisa menangkap nuansa preferensi yang lebih kompleks.
  - Kurang adanya _serendipity_ atau penemuan item di luar profil pengguna yang sudah ada.
  - Membutuhkan _domain knowledge_ untuk _feature engineering_ yang baik.

### 2. Collaborative Filtering

Pendekatan ini merekomendasikan anime dengan memanfaatkan pola rating dari banyak pengguna. Model yang digunakan adalah _neural network_ berbasis _matrix factorization_ (menggunakan _embedding layers_).

- **Arsitektur Model (`RecommenderNet`):**

  - **Embedding Layers:**
    - _User Embedding_: `layers.Embedding(input_dim=num_users, output_dim=embedding_size, ...)`
    - _Anime Embedding_: `layers.Embedding(input_dim=num_anime, output_dim=embedding_size, ...)`
    - Regularisasi L2 ditambahkan pada _embedding layers_ untuk mencegah _overfitting_.
  - **Bias Terms:**
    - _User Bias_: `layers.Embedding(input_dim=num_users, output_dim=1, ...)`
    - _Anime Bias_: `layers.Embedding(input_dim=num_anime, output_dim=1, ...)`
  - **Proses Prediksi:**
    1.  Vektor _embedding_ pengguna dan anime diambil.
    2.  Dilakukan _dot product_ antara vektor _embedding_ pengguna dan anime.
    3.  Hasil _dot product_ dijumlahkan dengan _bias_ pengguna dan _bias_ anime.
    4.  Output dilewatkan melalui fungsi aktivasi `sigmoid` untuk menghasilkan prediksi rating dalam rentang [0, 1] (sesuai dengan rating yang telah dinormalisasi).

  ```python
  class RecommenderNet(tf.keras.Model):
      def __init__(self, num_users, num_anime, embedding_size=50, verbose=False, **kwargs):
          super(RecommenderNet, self).__init__(**kwargs)
          self.user_embedding = layers.Embedding(
              input_dim=num_users, output_dim=embedding_size,
              embeddings_initializer='he_normal',
              embeddings_regularizer=keras.regularizers.l2(1e-6)
          )
          self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)
          self.anime_embedding = layers.Embedding(
              input_dim=num_anime, output_dim=embedding_size,
              embeddings_initializer='he_normal',
              embeddings_regularizer=keras.regularizers.l2(1e-6)
          )
          self.anime_bias = layers.Embedding(input_dim=num_anime, output_dim=1)

      def call(self, inputs):
          inputs = tf.cast(inputs, tf.int32)
          user_id = inputs[:, 0]
          anime_id = inputs[:, 1]
          user_vector = self.user_embedding(user_id)
          user_bias = self.user_bias(user_id)
          anime_vector = self.anime_embedding(anime_id)
          anime_bias = self.anime_bias(anime_id)
          dot_user_anime = tf.reduce_sum(user_vector * anime_vector, axis=1, keepdims=True)
          x = dot_user_anime + user_bias + anime_bias
          return tf.nn.sigmoid(x)
  ```

- **Proses Training:**

  - _Loss Function_: `MeanSquaredError`
  - _Optimizer_: `Adam` (learning rate awal 0.001)
  - _Metrics_: `MeanAbsoluteError` (MAE), `RootMeanSquaredError` (RMSE)
  - _Callbacks_:
    - `EarlyStopping`: Menghentikan _training_ jika `val_MAE` tidak membaik setelah beberapa _epoch_ (patience=10). Bobot terbaik akan dikembalikan.
    - `ReduceLROnPlateau`: Mengurangi _learning rate_ jika `val_MAE` tidak membaik (factor=0.5, patience=5).
  - _Batch Size_: 128
  - _Epochs_: 100 (dengan _early stopping_)

- **Kelebihan:**
  - Mampu menangkap pola preferensi yang kompleks dan implisit dari interaksi pengguna.
  - Tidak memerlukan fitur eksplisit dari item (seperti genre), hanya ID pengguna dan item.
  - Dapat menghasilkan rekomendasi yang lebih personal dan beragam (_serendipity_).
- **Kekurangan:**
  - Mengalami _cold start problem_ untuk pengguna baru atau item baru yang belum memiliki interaksi.
  - Hasil rekomendasi kurang transparan (_black box_).
  - Membutuhkan jumlah data interaksi yang cukup besar untuk performa yang baik.

### Hasil Rekomendasi Contoh

- **Content-Based (untuk anime "Bleach"):**
  Model berhasil merekomendasikan anime-anime yang sangat mirip genrenya, mayoritas merupakan bagian dari waralaba Bleach itu sendiri atau anime dengan genre serupa seperti Shaman King.

  ![Bleach](https://github.com/user-attachments/assets/3d3aa3f9-d6f1-4d51-ad72-05498c411c35)

###  **Output**

1. **Bleach: The Sealed Sword Frenzy** (Action, Adventure, Comedy, Super Power, Supernatural, Shounen)
2. **Bleach: Sennen Kessen-hen** (Action, Adventure, Comedy, Super Power, Supernatural, Shounen)
3. **Bleach Movie 1: Memories of Nobody** (Action, Adventure, Comedy, Super Power, Supernatural, Shounen)
4. **Bleach Movie 4: Jigoku-hen** (Action, Adventure, Comedy, Super Power, Supernatural, Shounen)
5. **Bleach Movie 2: The DiamondDust Rebellion** (Action, Adventure, Comedy, Super Power, Supernatural, Shounen)
6. **Shaman King** (Action, Adventure, Comedy, Super Power, Supernatural, Shounen)
7. **Bleach Movie 3: Fade to Black** (Action, Adventure, Comedy, Super Power, Supernatural, Shounen)
8. **Shaman King (2021)** (Action, Adventure, Comedy, Super Power, Supernatural, Shounen)
9. **Zettai Karen Children: The Unlimited - Hyoubu Kyousuke** (Action, Super Power, Supernatural, Shounen)
10. **Toriko: Barbarian Ivy wo Hokaku Seyo!** (Action, Super Power, Adventure, Comedy, Shounen)

---
- **Collaborative Filtering (untuk User ID: 226500):**
  Model memberikan rekomendasi yang tampak relevan dengan anime yang pernah ditonton dan diberi rating tinggi oleh pengguna tersebut (Inazuma Eleven: Sports, Super Power, Shounen). Rekomendasi mencakup genre seperti Sports, Shounen, Comedy, dan Drama, serta memperluas ke Sci-Fi dan Slice of Life.
  
  **Anime dengan Rating Tinggi dari User:**
  - Inazuma Eleven (Sports, Super Power, Shounen)
  
  **Top 15 Rekomendasi:**
  1. Hajime no Ippo (Comedy, Sports, Drama, Shounen)
  2. Gintama (Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen)
  3. Clannad: After Story (Slice of Life, Comedy, Supernatural, Drama, Romance)
  4. Fullmetal Alchemist: Brotherhood (Action, Military, Adventure, Comedy, Drama, Magic, Fantasy, Shounen)
  5. Suzumiya Haruhi no Shoushitsu (Comedy, Mystery, Romance, School, Sci-Fi, Supernatural)
  6. Steins;Gate (Thriller, Sci-Fi)
  7. Uchuu Kyoudai (Comedy, Sci-Fi, Seinen, Slice of Life, Space)
  8. Gintama Movie 2: Kanketsu-hen - Yorozuya yo Eien Nare (Action, Sci-Fi, Comedy, Historical, Parody, Samurai, Shounen)
  9. Haikyuu!! (Comedy, Sports, Drama, School, Shounen)
  10. Gintama° (Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen)
  11. Ansatsu Kyoushitsu 2nd Season (Action, Comedy, School, Shounen)
  12. Kuroko no Basket: Saikou no Present Desu (Action, Sports, School, Shounen)
  13. Natsume Yuujinchou Roku (Slice of Life, Demons, Supernatural, Drama, Shoujo)
  14. 3-gatsu no Lion 2nd Season (Drama, Game, Seinen, Slice of Life)
  15. Owarimonogatari 2nd Season (Mystery, Comedy, Supernatural, Vampire)

---

## Evaluation

Metrik evaluasi digunakan untuk mengukur performa model _collaborative filtering_ dalam memprediksi rating. Rating telah dinormalisasi ke rentang [0, 1]. Benchmark error ditetapkan sebesar 10% dari rentang rating, yaitu 0.1.

### Metrik Evaluasi yang Digunakan

1.  **Mean Absolute Error (MAE)**

    - Formula: $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
    - Cara Kerja: MAE mengukur rata-rata selisih absolut antara nilai prediksi ($\hat{y}_i$) dan nilai aktual ($y_i$). Semakin kecil nilai MAE, semakin baik performa model. MAE memberikan bobot yang sama untuk semua error.

2.  **Root Mean Squared Error (RMSE)**
    - Formula: $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
    - Cara Kerja: RMSE mengukur akar dari rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. RMSE memberikan bobot yang lebih besar pada error yang besar karena adanya proses pengkuadratan. Semakin kecil nilai RMSE, semakin baik performa model.

### Hasil Evaluasi Model Collaborative Filtering

Model dilatih selama 31 _epoch_ sebelum dihentikan oleh _early stopping_ (bobot terbaik dari _epoch_ ke-21 dikembalikan).

- **Training & Validation MAE Curve:**
  ![Performa Model Berdasarkan MAE per Epoch](https://github.com/user-attachments/assets/de63f332-6a36-46a7-8207-55d7455a0f5a)

  - Nilai minimum y (rating dinormalisasi): 0.0
  - Nilai maksimum y (rating dinormalisasi): 1.0
  - Benchmark MAE (10% rentang): 0.1000
  - **MAE model saat ini (pada data validasi): 0.1522**
  - Kesimpulan MAE: Model saat ini memiliki MAE 0.1522, yang berada di atas benchmark 0.1. Ini menunjukkan bahwa akurasi prediksi model masih perlu ditingkatkan.

- **Training & Validation RMSE Curve:**
  ![Performa Model Berdasarkan RMSE per Epoch](https://github.com/user-attachments/assets/7822cdec-acf9-4e1a-b368-005c82598372)
  - Benchmark RMSE (10% rentang): 0.1000
  - **RMSE model saat ini (pada data validasi): 0.1903**
  - Kesimpulan RMSE: RMSE sebesar 0.1903 juga berada di atas benchmark, menandakan model masih memprediksi dengan error yang cukup besar dan memerlukan perbaikan lebih lanjut.

### Diskusi Hasil Evaluasi

Meskipun model _collaborative filtering_ berhasil dilatih dan menunjukkan konvergensi, nilai MAE (0.1522) dan RMSE (0.1903) pada data validasi masih lebih tinggi dari benchmark yang diharapkan (0.1). Ini mengindikasikan bahwa ada ruang untuk peningkatan. Beberapa saran perbaikan meliputi:

- **Feature Engineering Tambahan:** Menggabungkan fitur konten (genre, studio, dll.) ke dalam model _collaborative filtering_ untuk menciptakan model _hybrid_.
- **Peningkatan Arsitektur Model:** Mencoba arsitektur _neural network_ yang lebih kompleks atau model faktorisasi matriks alternatif.
- **Hyperparameter Tuning Lanjutan:** Menggunakan teknik optimasi _hyperparameter_ yang lebih sistematis seperti Grid Search atau Bayesian Optimization.
- **Penanganan Data:** Eksplorasi lebih lanjut terhadap data rating, misalnya memfilter pengguna dengan sedikit interaksi atau anime dengan sedikit rating.

Untuk model _Content-Based Filtering_, evaluasi dilakukan secara kualitatif dengan melihat relevansi genre dari anime yang direkomendasikan. Hasilnya menunjukkan bahwa model mampu mengidentifikasi anime dengan genre yang sangat mirip, yang sesuai dengan tujuan dari pendekatan ini.

---
