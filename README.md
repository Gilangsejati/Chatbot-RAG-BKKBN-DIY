
# Chatbot FAQ BKKBN DIY Berbasis Retrieval Augmented Generation (RAG)

## Deskripsi Proyek
Proyek ini merupakan pengembangan prototype chatbot berbasis Retrieval Augmented Generation (RAG) yang dibuat selama kegiatan magang di **BKKBN Daerah Istimewa Yogyakarta (DIY)**. Chatbot ini memanfaatkan dataset Frequently Asked Questions (FAQ) BKKBN sebagai sumber pengetahuan untuk menjawab pertanyaan pengguna secara otomatis.

Tujuan dari pengembangan chatbot ini adalah untuk memberikan gambaran mengenai pemanfaatan teknologi **Artificial Intelligence dan Large Language Model (LLM)** dalam meningkatkan layanan informasi digital. Sistem ini masih bersifat prototype dan belum digunakan secara resmi oleh BKKBN, namun diharapkan dapat menjadi referensi untuk pengembangan sistem chatbot di masa mendatang.

---

## Teknologi yang Digunakan
Beberapa teknologi yang digunakan dalam proyek ini antara lain:

- **Python**
- **Embedding Model**
- **Vector Database (FAISS / ChromaDB)**
- **Large Language Model (LLM)**
- **Framework antarmuka chatbot**

---

## Alur Pengembangan Sistem

![Alur Sistem Chatbot](https://github.com/Gilangsejati/Chatbot-RAG-BKKBN-DIY/blob/main/Chatbot_Kemendukbangga/images/alur.png)

### 1. Dataset FAQ
Tahap awal dilakukan dengan mengumpulkan dataset berupa **Frequently Asked Questions (FAQ)** yang berkaitan dengan informasi BKKBN. Dataset ini digunakan sebagai sumber pengetahuan utama bagi chatbot.

### 2. Preprocessing
Dataset FAQ kemudian diproses melalui tahap **pembersihan teks**, seperti menghapus karakter yang tidak diperlukan dan menyesuaikan format data agar dapat diproses oleh sistem.

### 3. Embedding
Setiap teks dalam dataset diubah menjadi **representasi vektor (embedding)** menggunakan model embedding. Proses ini memungkinkan sistem untuk memahami hubungan makna antar teks.

### 4. System Design
Pada tahap ini dilakukan perancangan **arsitektur sistem chatbot berbasis RAG**, termasuk alur proses pencarian informasi dan integrasi model bahasa.

### 5. Retrieval
Ketika pengguna mengajukan pertanyaan, sistem akan mencari informasi yang paling relevan dari dataset menggunakan **vector similarity search** pada vector database.

### 6. Development
Tahap ini merupakan proses implementasi sistem chatbot menggunakan bahasa pemrograman dan framework yang telah dipilih.

### 7. User Interface (UI)
Pengembangan antarmuka chatbot dilakukan agar pengguna dapat berinteraksi dengan sistem dengan mudah.

### 8. Testing and Evaluation
Tahap terakhir adalah melakukan pengujian sistem untuk memastikan chatbot dapat memberikan jawaban yang relevan dan sesuai dengan dataset FAQ yang digunakan.

---

## Tampilan Chatbot

![Chatbot Interface]([images/chatbot_interface.png](https://github.com/Gilangsejati/Chatbot-RAG-BKKBN-DIY/blob/main/Chatbot_Kemendukbangga/images/tampilan_chatbot.png))

Gambar di atas menunjukkan contoh tampilan chatbot yang digunakan untuk berinteraksi dengan pengguna. Pengguna dapat mengajukan pertanyaan terkait informasi BKKBN dan sistem akan memberikan jawaban berdasarkan data FAQ yang tersedia.

---

## Catatan
Proyek ini merupakan **prototype yang dikembangkan selama program magang** dan belum diimplementasikan secara resmi oleh BKKBN. Hasil pengembangan ini diharapkan dapat menjadi **referensi atau usulan pengembangan sistem chatbot informasi di masa mendatang**.

---

## Author

**Gilang Sejati**  
Mahasiswa Informatika Amikom Yogyakarta
Program Magang – BKKBN DIY
