# Submission 1: Stream Game Review

Nama : Hana Octavia Trinida Malo
Username dicoding : Hana Octavia Trinida Malo


|  | Deskripsi |
| ------ | ------ |
| Dataset | Dataset ini bernama Steam Game Review. Dataset ini merupakan kumpulan review game-game yang ada di platform Steam. Review ini mencakup 2 jenis review yaitu positif atau negatif. Link Dataset : https://www.kaggle.com/datasets/arashnic/game-review-dataset|
| Masalah | Proyek ini bertujuan untuk membangun sebuah model Machine Learning yang dapat menghasilkan output yang tepat terhadap inputan review dari user Steam Game. Sebelumnya review yang masuk masih di kategorikan kurang tepat |
| Solusi machine learning | Berdasarkan Masalah di atas maka dibutuhkan beberapa tahapan atau proses seperti pengolahan data dan pengembangan model machine learning |
| Metode pengolahan | Pada Proses pengolahan data, memiliki 3 proses, yaitu proses data ingestion yang menggunakan komponen ExampleGen, data validation yang menggunakan beberapa komponen yang disediakan oleh TFX, seperti StatisticGen, SchemaGen, dan ExampleValidator, dan yang terakhir data preprocessing yang menggunakan komponen Transform|
| Arsitektur model | Pada proses pengembangan model menggunakan komponen Trainer, sedangkan pada proses analisis dan validasi model menggunakan komponen Resolver dan Evaluator. Untuk arsitektur model bisa dilihat di gambar 1 |
| Metrik evaluasi | Untuk Mengevaluasi model, ada beberapa metrik yang digunakan, seperti   ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, BinaryAccuracy', |
| Performa model | Performa model yang dibangun menunjukan peforma yang cukup baik |

Gambar 1
![Screenshot (658)](https://user-images.githubusercontent.com/86582130/216260115-7dbd482f-78a5-47b7-99ef-fb8e6c066dd5.png)
