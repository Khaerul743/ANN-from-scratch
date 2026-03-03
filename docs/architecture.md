# Architecture Overview — Feed Forward

Bagian ini menjelaskan bagaimana proses **feed forward** bekerja pada Artificial Neural Network yang dibangun pada repository ini. Feed forward merupakan tahap di mana data bergerak dari input layer menuju output layer untuk menghasilkan prediksi.

Tujuan utama dari proses ini adalah melakukan transformasi data secara bertahap menggunakan operasi linear dan non-linear sehingga model mampu menangkap pola pada data.

---

## 1. Gambaran Umum Feed Forward

Sebelum masuk ke model, data biasanya melalui proses **normalisasi** (jika diperlukan) agar distribusi nilai lebih stabil selama proses training.

Setelah itu, data akan melewati serangkaian layer yang masing-masing terdiri dari dua tahap utama:

1. Transformasi linear (operasi matriks)
2. Fungsi aktivasi (transformasi non-linear)

Secara konseptual:

```text
Input → Linear Transformation → Activation → ... → Prediction
```

---

## 2. Transformasi Linear (Pre-Activation)

Langkah pertama dalam setiap layer adalah melakukan operasi linear antara input dan parameter model.

Secara matematis:

$$
z = XW + b
$$

Dimana:

* $X$ : matriks input (data)
* $W$ : matriks bobot (*weights*)
* $b$ : bias
* $z$ : nilai **pre-activation** (hasil sebelum fungsi aktivasi)

Pada implementasi menggunakan NumPy:

```python
z = X @ W + b
```

### Penjelasan Komponen

* **z**
  Variabel untuk menyimpan hasil transformasi linear sebelum masuk fungsi aktivasi.

* **X**
  Data input yang akan diproses oleh layer.

* **W**
  Parameter bobot model. Biasanya diinisialisasi menggunakan distribusi acak:

```python
W = np.random.randn(input_size, output_size)
```

* **b**
  Bias yang ditambahkan untuk memberikan fleksibilitas translasi fungsi:

```python
b = np.zeros((1, output_size))
```

* **@**
  Operator matrix multiplication di NumPy (setara dengan `np.dot()`).

Transformasi ini disebut **linear transformation** karena hanya melakukan kombinasi linear terhadap input.

---

## 3. Fungsi Aktivasi (Non-Linearity)

Jika neural network hanya terdiri dari operasi linear, maka beberapa layer linear yang ditumpuk tetap ekuivalen dengan satu transformasi linear saja. Artinya, model tidak mampu mempelajari pola non-linear.

Oleh karena itu digunakan **fungsi aktivasi**.

Setelah memperoleh nilai pre-activation $z$, kita menghitung:

$$
a = f(z)
$$

Dimana:

* $f$ = fungsi aktivasi
* $a$ = output layer (*activation*)

Dalam implementasi:

```python
a = ReLU(z)
```

---

### Fungsi Aktivasi yang Umum Digunakan

Beberapa fungsi aktivasi yang umum:

* ReLU (Rectified Linear Unit)
* Sigmoid
* Tanh

Pada project ini digunakan **ReLU** untuk hidden layer.

#### ReLU

$$
ReLU(z) = \max(0, z)
$$

ReLU mengubah semua nilai negatif menjadi nol dan mempertahankan nilai positif.

Keuntungan penggunaan ReLU:

* membantu menjaga stabilitas gradient dibanding sigmoid/tanh
* komputasi sederhana dan efisien
* mempercepat proses training pada banyak kasus

Turunan ReLU:

$$
ReLU'(z) =
\begin{cases}
1 & \text{jika } z > 0 \
0 & \text{jika } z \le 0
\end{cases}
$$

Turunan ini memungkinkan gradient tetap mengalir pada neuron yang aktif selama training.

---

## 4. Stacking Layer

Satu layer neural network terdiri dari:

```text
Linear Transformation → Activation
```

Layer-layer ini kemudian ditumpuk (*stacked*) beberapa kali:

$$
a^{(l)} = f\left(W^{(l)} a^{(l-1)} + b^{(l)}\right)
$$

Penumpukan layer memungkinkan model mempelajari hubungan **non-linear kompleks** pada data.

Semakin dalam jaringan, semakin abstrak representasi fitur yang dipelajari.

---

## 5. Output Layer dan Prediction

Tahap terakhir feed forward adalah menghasilkan prediksi.

Pemilihan fungsi aktivasi output bergantung pada jenis masalah.

---

### Binary Classification — Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Sigmoid mengubah output menjadi rentang:

$$
0 \le \hat{y} \le 1
$$

sehingga dapat diinterpretasikan sebagai probabilitas.

---

### Regression

Untuk regresi biasanya:

* menggunakan aktivasi linear (tanpa fungsi aktivasi tambahan), atau
* dalam beberapa kasus menggunakan ReLU.

Hal ini karena output regresi tidak dibatasi pada rentang tertentu.

---

## 6. Perhitungan Error (Loss Function)

Setelah model menghasilkan prediksi $\hat{y}$, langkah berikutnya adalah menghitung error terhadap nilai sebenarnya $y$.

Loss function digunakan untuk mengukur seberapa jauh prediksi model dari target.

---

### Mean Squared Error (MSE) — Regression

$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Digunakan untuk masalah regresi.

---

### Binary Cross Entropy (BCE) — Binary Classification

$$
L = -\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
$$

Digunakan untuk klasifikasi biner.

Loss inilah yang nantinya menjadi sinyal utama dalam proses **backpropagation** untuk memperbarui parameter model.

---

## Ringkasan Feed Forward

Secara keseluruhan, proses feed forward dapat dirangkum sebagai berikut:

```text
Input Data
   ↓
Linear Transformation (XW + b)
   ↓
Activation Function
   ↓
Stacked Layers
   ↓
Output Activation
   ↓
Prediction
   ↓
Loss Computation
```

Feed forward menghasilkan prediksi dan nilai error yang akan digunakan pada tahap berikutnya: **backpropagation**.
