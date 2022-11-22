# Laporan Proyek Machine Learning - Hana Octavia Trinida Malo

## Domain Proyek
Merokok adalah kegiatan mengkonsumsi rokok. Orang-orang yang mengisap rokok disebut perokok. Perokok dibagi atas dua, yaitu perokok aktif dan perokok pasif. perokok aktif adalah perokok yang mengisap rokok secara langsung menggunakan mulut serta menghirup asap rokok, sedangkan perokok pasif adalah perokok yang hanya menghirup asap rokok. Di dalam satu batang rokok yang diisap akan mengeluarkan sekitar 4.000 bahan kimia berbahaya yang dapat menyebabkan efek berbahaya bagi tubuh dan menimbulkan kecanduan[1].

Berbagai cara dilakukan untuk mengurangi tingkat konsumsi rokok dalam masyarakat, salah satunya dengan membatasi area-area tertentu untuk tidak menjadi kawasan asap rokok. Kenyataannya masyarakat sering kali tidak mematuhi peraturan dan merokok di sembarang tempat.

Berdasarkan masalah-masalah di atas, banyak perusahaan ingin yang membangun alat yang dapat mendeteksi asap rokok, sehingga ketika ada masyarakat yang merokok di daerah yang bebas asap rokok dapat diberi peringatan. Dalam pembuatan alat ini bukan hanya bahan apa saja yang dipakai, tetapi bagaimana alat tersebut dapat mendeteksi adanya asap rokok atau tidak dengan benar.


## Pendefinisian Bisnis
Sebuah perusahaan ingin membuat sebuah alat yang dapat mendeteksi asap rokok. Dalam pembangunan alat tersebut menerapkan IoT dan model Machine Learning. Dalam membangun sebuah model Machine Learning yang baik, dibutuhkan beberapa variabel yang dapat mempengaruhi apakah alat tersebut (kita sebut saja alarm) dapat berbunyi saat ada asap rokok atau tidak. Variabel-variabel ini seperti senyawa organik, kelembapan udara, konsentrasi CO2, tekanan udara, serta gas etanol.



### Masalah
Berdasarkan latar belakang yang telah diuraikan di atas, maka dapat dirumuskan masalah-masalah yang harus diselesaikan antara lain :
- Dari serangkaian variabel yang ada, variabel apa yang paling berpengaruh terhadap kesuksesan alarm dapat berbunyi saat ada asap rokok?
- Apakah kelembapan udara berpengaruh terhadap dalam pendeteksian asap rokok?
- variabel apakah yang sangat berpengaruh dalam pendeteksian asap rokok?

### Tujuan
Tujuan dari pembuatan laporan proyek ini adalah : 
- Mengetahui variabel yang paling berpengaruh dalam pendeteksian asap rokok 
- Membuat model machine learning yang dapat mendeteksi asap rokok 
- Mengatasi variabel yang tidak memiliki korelasi dalam pendeteksian asap rokok

### Solusi
Solusi untuk mencapai tujuan diatas adalah :
- Menganalisis dataset menggunakan EDA
- Menggunakan Bernoullie Naive Bayes, Support Vector Machine dan Logistic Regression algoritma dalam membangun model
- Melakukan evaluasi dengan melihat akurasi dan nilai f1 dengan 'akurasi model', dan 'f1 score'


## Data Understanding
Dataset yang saya gunakan merupakan dataset untuk mendeteksi asap rokok, yang saya unduh dari link berikut : [Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset) 
Dataset ini memiliki target variabel dengan value biner (0 dan 1). Selain itu, dataset ini juga mengandung lebih dari 60.000 baris data dan 16 kolom

### Variabel-variabel pada Smoke Detection Dataset adalah sebagai berikut:
- Unnamed:0 : Penomoran baris
- UTC : Timestamp waktu dalam detik
- Temperature : Temperatur udara dalam celcius
- Humidity : kelembapan udara
- TVOC : Total senyawa Volatile organik dalam ppb (parts per billion)
- eCo2 : konsentrasi yang setara dengan CO2
- Raw H2 : Hidrogen mentah
- Raw Ethanol : gas ethanol mentah
- Pressure : tekanan udara
- PM1.0 : Materi patikulat dengan diameter kurang dari 1,0 mikrometer
- PM2.5 : Bahan patikulat dengan diameter kurang dari 2,5 mikrometer
- NC0.5 : Konsentrasi partikel dengan diameter kurang dari 0,5 mikrometer
- NC1.0 : Konsentrasi partikel dengan diameter kurang dari 1,0 mikrometer
- NC2.5 : Konsentrasi partikel dengan diameter kurang dari 2,5 mikrometer
- CNT : Hitungan Sederhana
- Alarm Kebakaran : (Realitas) 1 jika terjadi kebakaran dan 0 jika terjadi kebakaran

### Exploratory Data Analysis

#### Mengatasi Outliners
Mengecek apakah variabel-variabel yang ada mempunyai outliners dengan boxplot.
![boxplot outliners](https://user-images.githubusercontent.com/86582130/190961263-4610b2dc-cd2a-4403-82b1-c887bb086112.png)

Dari gambar di atas, terlihat bahwa variabel-variabel yang ada pada dataset terdeteksi outliners maka kita dapat mengatasinya dengan **IQR Method**
Hasil dari penggunaan IQR Method
```
(35684, 15)
```
Bisa dilihat terjadi pengurangan dataset, yang awalnya 62630 menjadi 35684
#### Unvariate variabel
Pada proses ini, saya membagi fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features.
```
numerical_features = ['UTC', 'Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT']
categorical_features =  ['Fire Alarm']
```
Kemudian menganalisisnya categorical_features menggunakan grafik yang menunjukan jumlah sample dan presentasi per value dalam variabel 'Fire Alarm'.
![grafik categorikal](https://user-images.githubusercontent.com/86582130/190961243-aeddb0ea-d359-4298-a965-e2200d4dcd98.png)

Selanjutnya melakukan visualisasi data untuk numerical_features
```
sp.hist(bins=50, figsize=(20,15), color='pink')
plt.show()
```
![Screenshot (439)](https://user-images.githubusercontent.com/86582130/190961226-d56e4502-1df0-4edc-b3dc-47607a547b21.png)


#### Multivariate variabel
Tahap ini bertujuan untuk melihat apakah ada korelasi antara data numerik dengan data target yaitu 'Fire Alarm'
```
plt.figure(figsize=(10, 8))
correlation_matrix = sp.corr().round(2)
 
sns.heatmap(data=correlation_matrix, annot=True, cmap='Purples', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
```
![metrik](https://user-images.githubusercontent.com/86582130/190961212-1564349a-7200-4903-ad4a-e51465f36f4b.png)

Dari visualisasi di atas ada beberapa kesimpulan
- variabel CNT memiliki korelasi yang paling tinggi dengan Fire Alarm, namun sayangnya data yang dimiliki tidak berpengaruh pada Fire alarm karena hanya merupakan data Simple count
- variabel TVOC[ppb] memiliki korelasi yang paling tinggi dengan variabele fire alarm
- variabel Temperature, humidity, eCO2 memiliki korelasi yang sangat rendah dengan data target kita, sehingga kita bisa mengabaikan variabel ini dengan cara mendropnya

## Data Preparation
Pada Data Preparation ada beberapa tahap yang harus dilakukan
1. Membagi data train dan data test

Pembagian dataset menjadi data train dan data test ini dilakukan dengan perbandingan 8:2 sehingga pada test_size diset 0.2. Pembagian dataset ini bertujuan agar memudahkan kita dalam proses evaluasi performa model dan agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. 

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123) 
```
Berikut ini adalah jumlah keseluruhan data, data train, dan data set
```
Jumlah Seluruh Dataset: 35684
Jumlah data train: 28547
Jumlah data test: 7137
```

2. Melakukan Standarisasi Data numerik

Pada tahap ini kita akan melakukan standarisasi pada data numerik agar data yang kita miliki mempunyai skala yang sama sehingga dapat diolah oleh algoritma
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. 

Setelah itu kita dapat mengecek nilai mean dan standar deviasi, menggunakan kode di bawah ini
```
X_train[numerical_features].describe().round(4)
```

![mean dan deviasi](https://user-images.githubusercontent.com/86582130/190961165-24272516-7a28-4655-9db1-e2b8577a3115.png)

Dari gambar di atas menunjukan nilai mean = 0 dan standar deviasi = 1.

## Modeling
Model Machine Learning yang saya gunakan untuk menyelesaikan masalah adalah :
1. Bernoulli Naive Bayes

Dalam penelitian Dewi dkk menyatakan bahwa Bernoulli Naïve bayes lebih unggul dibanding metode Gaussian Naïve Bayes, dikarenakan data yang digunakan untuk penelian terdiri dari beberapa variabel yang memiliki nilai biner. [2] Cara kerja Naive bayes sendiri adalah menghitung peluang dari satu kelas dari masing-masing kelompok atribut yang ada dan menentukan kelas mana yang paling optimal. Pada kasus ini saya menghitung peluang adanya asap rokok dengan menggunakan beberapa variabel yang ada.
```
bnb = BernoulliNB()
```

2. Support Vector Machine

Dalam penelitian Ichwan dkk metode SVM berdasarkan penelitian ini menyatakan bawah SVM mampu menghasilkan model klasifikasi yang baik meskipun dilatih dengan data yang sedikit. Kelemahan metode SVM berdasarkan penelitian ini adalah sulit diterapkan untuk data yang memiliki jumlah dimensi yang sangat besar[3]. Cara kerja SVM adalah dengan menempatkan garis lurus(decision boundary) untuk membagi 2 kelas. Untuk mendapatkan decision boundary terbaik, svm menggunakan metode Soft-Margin, dan kernel. Dalam pelatihan model, saya menggunakan SVM untuk 2 kelas yaitu ada asap rokok dan tidak ada asap rokok.
```
svc = SVC(random_state = 42)
```

3. Logistic Regression

Regresi Logistik adalah metode klasifikasi yang memperkirakan probabilitas suatu peristiwa terjadi. Pada kasus ini, yaitu pendeteksi asap rokok yang merupakan kalasifikasi ada atau tidaknya asap rokok, logistic regression bekerja dengan menghitung probabilitas kelas dari sampel yang ada. 
```
LR = LogisticRegression(solver='lbfgs', max_iter=1000)
```
Tuning hyperparameters :
solver : metode optimasi kita set 'lbfgs'

Setelah dilakukan pelatihan 1 di antara 3 model ini menghasilkan hasil yang cukup jauh perbedaannya. Model ini adalah Bernoulli Naive Bayes, saat melakukan pelatihan dan melihat nilai akurasi dan f1 score, model ini menghasilkan nilai yang rendah dan perbedaan cukup signifikan dari model yang menggunakan algoritma SVM dan Logistic Regression.
Maka dari itu **model terbaik** untuk solusi dari masalah yang dipaparkan adalah menggunakan model yang dilatih dengan **algoritma SVM dan Logistic Regression**

## Evaluation
Untuk mengevaluasi model saya menggunakan 2 metrik yaitu akurasi dan f1 score
1. Akurasi
Akurasi adalah teknik pengukuran model yang nantinya hasil pengukurannya dapat mengetahui kinerja dari model[3]

Formula perhitungan akurasi adalah :
-----**Akurasi = (data berhasil/jumlah data) × 100%**----

Berikut ini adalah potongan kode untuk melihat akurasi model yang sudah dilatih
```
for nama_model, model in models:
    # Prediksi y_pred
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # melihat akurasi model
    print('{:s} acc score : {:.3f}'.format(nama_model, acc))
```

Hasil dari metrik akurasi adalah :
```
bnb acc score : 0.820
svc acc score : 1.000
LR acc score : 1.000
```
2. f1 Score
F1-Score memiliki teknik pengukuran pada model klasifikasi yang baik. Dengan menggunakan f1-score sebagai acuan, dapat mengukur kinerja sistem lebih baik dari pada akurasi dan precision[4]

-----**F1 Score = 2 × (Recall×Precission) / (Recall + Precission)**----
Berikut ini adalah potongan kode untuk melihat f1 score model yang sudah dilatih

```
for nama_model, model in models:
    # Prediksi y_pred
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    # melihat akurasi model
    print('{:s} f1 score : {:.3f}'.format(nama_model, f1))
```
Hasil dari metrik akurasi adalah :
```
bnb f1 score : 0.884
svc f1 score : 1.000
LR f1 score : 1.000
```
Plot metrik dengan bar chart :

![plot metrik](https://user-images.githubusercontent.com/86582130/190961097-7d86cf58-cce0-4de4-a34e-a0b07b354d2d.png)


Dari gambar di atas, skor akurasi dan f1 untuk model svc dan LR bernilai 1 sesuai dengan perhitungan di atas, tetapi model bnb juga pada skor akurasi dan f1 bernilai 1 di plot tetapi tidak sesuai dengan perhitungan di atas yaitu acc = 0.820 dan f1_score = 0,820.

## Kesimpulan
Berdasarkan hasil pelatihan model menggunakan 3 algoritma berbeda dan evaluasi menggunakan 2  metrik evaluasi yaitu akurasi dan f1 score, kemudian melakukan prediksi, menunjukan model yang tepat untuk melakukan predeksi ada tidaknya asap rokok adalah model yang dibangun dengan Bernaullie Naive Bayes, algoritma Support Vector Machine dan Logistic Regression. Ketiga algortima ini menunjukan hasil prediksi yang baik dan hasil akurasi di atas 80%

## Referensi
[1] Buleleng, Admin., 2021. *Mengenal Rokok Serta Dampaknya Bagi Kesehatan*. [Online] 
Available at: https://buleleng.bulelengkab.go.id/informasi/detail/artikel/58-mengenal-rokok-serta-dampaknya-bagi-kesehatan
[Accessed 17 September 2022].

[2] Dewi, I. P., Lhaksmana, K. M. & Jondri, 2021. Prediksi Retweet Menggunakan Metode Bernoulli dan Gaussian Naive Bayes di Media Sosial Twitter Dengan Topik Vaksinasi Covid-19. *e-Proceeding of Engineering*. 2021. 8(5), pp. 11216-11225.

[3]Ichwan, M., Dewi, I. A. & S, Z. M., 2018. Klasifikasi Support Vector Machine (SVM) Untuk Menentukan TingkatKemanisan Mangga Berdasarkan Fitur Warna. *MIND Journal* , 3(2), pp. 16-24.

[4]Satria, F., Zamhariri & Syaripudin, M. A., 2020. Prediksi Ketepatan Waktu Lulus Mahasiswa Menggunakan Algoritma C4.5 Pada Fakultas Dakwah Dan Ilmu Komunikasi UIN Raden Intan Lampung. *Jurnal Ilmiah MATRIK*, 22(1), pp. 28-35.



