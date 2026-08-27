# Language Classification using BERT and PyTorch

Praktična implementacija sistema za automatsku klasifikaciju jezika na osnovu tekstualnih podataka, korišćenjem neuronske mreže u PyTorch-u i pretreniranog modela `bert-base-multilingual-cased`.
---

## 📌 Opis projekta

Cilj projekta je demonstracija kompletnog ML pipeline-a — od sirovih podataka do istreniranog i evaluiranog modela:

- učitavanje i analiza podataka
- čišćenje i transformacija podataka
- priprema podataka za neuronsku mrežu
- definisanje arhitekture modela
- treniranje modela
- optimizacija hiperparametara
- izvođenje reproduktivnih eksperimenata
- evaluacija modela
- poređenje različitih konfiguracija
- praćenje eksperimenata pomoću **MLflow**-a
- verzionisanje projekta pomoću **Git**-a i **GitHub**-a

Model klasifikuje tekst na **12 različitih jezika**, koristeći pretrenirani `bert-base-multilingual-cased` kao osnovu.

---

## 🛠 Korišćene tehnologije

| Kategorija | Alati |
|---|---|
| Jezik | Python |
| Deep Learning | PyTorch |
| Modeli | Hugging Face Transformers (BERT) |
| Podaci | pandas, NumPy, scikit-learn |
| Vizualizacija | Matplotlib |
| Tracking | MLflow |
| Okruženje | Google Colab, Jupyter Notebook |
| Verzionisanje | Git, GitHub |

---

## 📊 Dataset

Skup podataka sadrži tekstualne primere na **12 jezika**, ukupno **60.000 primera**, učitanih iz JSON fajlova i organizovanih u `DataFrame` strukturu.

### Podela podataka

| Skup | Broj primera |
|---|---:|
| Training | 47.890 |
| Validation | 5.987 |
| Test | 5.987 |
| **Ukupno** | **59.864** |

> Razlika u odnosu na inicijalnih 60.000 primera nastaje nakon procesa validacije i pripreme podataka.

### Obrada podataka

1. Učitavanje JSON fajlova
2. Kombinovanje podataka
3. Provera validnosti podataka
4. Formiranje `DataFrame` strukture
5. Kodiranje oznaka jezika pomoću `LabelEncoder`
6. Podela na trening / validacioni / test skup
7. Tokenizacija teksta pomoću BERT tokenizer-a

### Analiza podataka

Analiza (`language_classification.ipynb`) obuhvata deskriptivnu statistiku, broj primera po jeziku, proveru nedostajućih vrednosti, strukturu podataka, distribuciju klasa i vizuelni prikaz.

---

## 🧠 Arhitektura modela

Osnova modela je pretrenirani transformer:

```
bert-base-multilingual-cased
```

Na izlaz BERT-a dodat je klasifikacioni sloj sa:

- **12 izlaznih klasa**
- **dropout slojem**

Arhitektura je implementirana u `src/models.py`.

---

## 🏋️ Treniranje modela

| Komponenta | Vrednost |
|---|---|
| Optimizator | AdamW |
| Funkcija gubitka | Cross Entropy Loss |

Tokom treniranja se prate: training/validation loss, accuracy i F1-score.
Evaluacija uključuje: test loss, test accuracy, test F1-score i vreme inferencije.

---

## 🧪 Eksperimentalne konfiguracije

Izvršeno je ukupno **5 eksperimenata** sa različitim hiperparametrima:

| Experiment | LR | Dropout | Epochs | Accuracy | F1 | Loss | Inference Time (s) |
|---|---|---|---|---|---|---|---|
| 1 | 2e-5 | 0.1 | 2 | 0.998330 | 0.998331 | 0.008856 | 52.9684 |
| 2 | 1e-5 | 0.1 | 2 | 0.998497 | 0.998495 | 0.006062 | 43.3687 |
| **3** ⭐ | **1e-5** | **0.3** | **2** | **0.998664** | **0.998665** | **0.007595** | **43.4687** |
| 4 | 1e-5 | 0.3 | 3 | 0.998163 | 0.998163 | 0.009695 | 46.9204 |
| 5 | 3e-5 | 0.3 | 2 | 0.995490 | 0.995497 | 0.020175 | 45.9855 |

### 🏆 Najbolji model — Experiment 3

- **Learning rate:** 1e-5
- **Dropout:** 0.3
- **Epochs:** 2
- **Accuracy:** 0.998664
- **F1-score:** 0.998665
- **Loss:** 0.007595
- **Inference time:** 43.4687 s

---

## 📈 MLflow

MLflow se koristi za evidenciju i poređenje eksperimenata.

**Parametri koji se loguju:**
- naziv modela, broj klasa, learning rate, dropout, broj epoha, batch size, uređaj za treniranje

**Metrike koje se loguju:**
- training/validation/test loss, accuracy, F1
- training time, inference time

---

## 🔁 Reproduktivnost

Radi reproduktivnosti eksperimenata koristi se fiksiranje random seed-a — funkcija se nalazi u `src/utils.py`. Verzije korišćenih biblioteka navedene su u `requirements.txt`.

---

## ✅ Evaluacija modela

Korišćene metrike: **Accuracy**, **F1-score**, **Loss**.

Dodatno se analiziraju: vreme treniranja, vreme inferencije, veličina modela i iskorišćenost računarskih resursa.

### Vizualizacije

- Poređenje modela (Accuracy / F1 / Loss po eksperimentu)
- Vremenske performanse (training / inference time)
- **Confusion Matrix** — analiza klasifikacije 12 jezika za najbolji model (Experiment 3)
- **ROC kriva** — odnos True Positive Rate i False Positive Rate (multiclass)
- **Precision-Recall kriva** — odnos preciznosti i odziva
- **Learning Curve** — training/validation loss i accuracy po epohama, za praćenje eventualnog overfitting-a

---

## 📦 Veličina modela

| Karakteristika | Vrednost |
|---|---:|
| Ukupan broj parametara | 177.862.668 |
| Trainable parametara | 177.862.668 |
| Procenjena veličina modela | 678.49 MB |

---

## 💻 Korišćeni računarski resursi

Eksperimenti su izvršavani u **Google Colab** okruženju (CPU/GPU, RAM, PyTorch, BERT model i tokenizer). Za zahtevnije eksperimente preporučuje se GPU okruženje.

---

## 📁 Struktura projekta

```
Language-Classification/
│
├── language_classification.ipynb
├── README.md
├── requirements.txt
│
└── src/
    ├── data.py            # Učitavanje i priprema podataka
    ├── dataset.py         # PyTorch Dataset klasa
    ├── experiment.py      # Pokretanje eksperimenata + MLflow logovanje
    ├── models.py           # Arhitektura modela
    ├── preprocessing.py    # Obrada i transformacija podataka
    ├── train.py             # Treniranje, evaluacija, metrike
    └── utils.py            # Pomoćne funkcije (npr. seed)
```

---

## ⚙️ Instalacija

### 1. Kloniranje repozitorijuma

```bash
git clone https://github.com/minadjordjevic/Language-Classification.git
cd Language-Classification
```

### 2. Kreiranje virtuelnog okruženja

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalacija biblioteka

```bash
pip install -r requirements.txt
```

---

## ▶️ Pokretanje projekta

Glavni deo projekta nalazi se u `language_classification.ipynb`.

```bash
jupyter notebook
```

Nakon pokretanja, otvoriti `language_classification.ipynb`. Notebook sadrži kompletan eksperimentalni proces: učitavanje podataka → analiza → preprocessing → kreiranje dataset-a i DataLoader-a → definisanje modela → treniranje → evaluacija → MLflow logovanje → poređenje rezultata.

---

## ☁️ Google Colab

Projekat je moguće pokrenuti i u Google Colab okruženju. Ukoliko se dataset nalazi na Google Drive-u, potrebno je povezati Drive nakon otvaranja notebook-a. Za ubrzavanje treniranja preporučuje se GPU runtime.

---

## 🔧 Git i verzionisanje

Projekat je razvijan korišćenjem Git-a za kontrolu verzija, uz praćenje razvoja projekta, izmena modela, promena hiperparametara, eksperimenata i rezultata.

🔗 **GitHub repozitorijum:** [minadjordjevic/Language-Classification](https://github.com/minadjordjevic/Language-Classification)

---

## 🏁 Rezultati

Eksperimenti pokazuju da izbor hiperparametara značajno utiče na performanse modela. Najbolji rezultat ostvario je **Experiment 3**:

- Learning rate = `1e-5`
- Dropout = `0.3`
- Epochs = `2`

sa **Accuracy = 0.998664** i **F1-score = 0.998665** — veoma visokom uspešnošću u klasifikaciji jezika.

---

## 🎯 Zaključak

U okviru projekta implementiran je kompletan sistem za klasifikaciju jezika korišćenjem pretreniranog BERT modela i PyTorch biblioteke — obrada podataka, treniranje, evaluacija i poređenje više eksperimentalnih konfiguracija.

MLflow je omogućio sistematsko praćenje eksperimenata, dok su Git i GitHub obezbedili verzionisanje i praćenje razvoja projekta.

Najbolji rezultat je ostvario **Experiment 3**, što potvrđuje da promena learning rate-a i dropout vrednosti može imati značajan uticaj na performanse modela.
