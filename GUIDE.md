# Danusys Model Eval Platform — Developer Guide

> **Bu döküman kimin için?**
> Kodu ilk kez gören, AI/ML bilgisi çok olmayan, "bu ne yapıyor, neden böyle yazılmış" sorusunu soran herkes için yazılmıştır.

---

## İçindekiler

1. [Sistem Ne Yapar?](#1-sistem-ne-yapar)
2. [Nasıl Çalıştırılır?](#2-nasıl-çalıştırılır)
3. [Büyük Resim — Mimari](#3-büyük-resim--mimari)
4. [Dosya Yapısı](#4-dosya-yapısı)
5. [Backend — Dosya Dosya Açıklama](#5-backend--dosya-dosya-açıklama)
   - [main.py](#51-mainpy)
   - [state.py](#52-statepy)
   - [evaluator.py](#53-evaluatorpy)
   - [gt_analyzer.py](#54-gt_analyzerpy)
   - [exporter.py](#55-exporterpy)
   - [routers/model.py](#56-routersmodelpy)
   - [routers/analysis.py](#57-routersanalysispy)
   - [routers/export.py](#58-routersexportpy)
   - [routers/upload.py](#59-routersuploadpy)
6. [Frontend — gradio_app.py](#6-frontend--gradio_apppy)
7. [Temel Kavramlar — Metrikler Ne Anlama Gelir?](#7-temel-kavramlar--metrikler-ne-anlama-gelir)
8. [Veri Akışı — Butona Tıkladığında Ne Olur?](#8-veri-akışı--butona-tıkladığında-ne-olur)
9. [API Referansı](#9-api-referansı)
10. [Dataset Dizin Yapısı](#10-dataset-dizin-yapısı)
11. [Excel Export — Sheet Açıklamaları](#11-excel-export--sheet-açıklamaları)
12. [Bilinen Kısıtlamalar ve Gelecek Geliştirmeler](#12-bilinen-kısıtlamalar-ve-gelecek-geliştirmeler)

---

## 1. Sistem Ne Yapar?

Bu platform, eğitilmiş bir **YOLO nesne tespit modelini** (.pt dosyası) bir **görüntü dataseti** üzerinde test ederek ne kadar iyi çalıştığını ölçer.

### Bir cümleyle:
> Modele görüntüler gösterirsin, model kutucuklar çizer, sistem "ne kadar doğru çizdin?" diye sorar.

### Pratik kullanım örneği:
- Elinde `falldown_best.pt` adlı bir model var (düşen insanları tespit etmek için eğitilmiş)
- 500 görüntülük bir test dataseti var, her görüntünün yanında doğru kutucukların koordinatları `.txt` dosyasında var
- Platform bu modeli bu dataset üzerinde çalıştırır ve şunu söyler:
  - "627 görüntüdeki 3200 nesnenin %92.3'ünü doğru buldu"
  - "Her 100 gerçek nesneden 19'unu kaçırdı (Recall = 0.81)"
  - "Bulduğu her 100 nesnenin 7'si aslında orada yoktu (Precision = 0.93)"

---

## 2. Nasıl Çalıştırılır?

### Gereksinimler
```bash
pip install ultralytics gradio fastapi uvicorn pandas openpyxl xlsxwriter pillow opencv-python pyyaml
```

### Backend'i Başlat (Terminal 1)
```bash
cd C:\Users\admin\fur\model_eval
python backend/main.py
# → http://localhost:8000 adresinde çalışır
# → http://localhost:8000/docs → tüm API endpoint'lerini interaktif görmek için
```

### Frontend'i Başlat (Terminal 2)
```bash
python frontend/gradio_app.py
# → http://localhost:7861 adresinde çalışır (browser'da açılır)
```

### Kullanım Adımları (UI'da)
1. **Model Path** kutusuna `.pt` dosyasının yolunu yaz → **Load Model** butonuna bas
2. **Dataset Folder** kutusuna dataset klasörünün yolunu yaz
3. **Run Both** butonuna bas → validation + per-image analiz birlikte çalışır
4. Sonuçları **Validation Metrics**, **Image Viewer**, **Worst Images** tab'larından incele
5. **Export** tab'ından Excel'e aktar

> **Not:** Backend ve frontend ayrı process'ler. İkisi de aynı anda çalışmalı. Backend dursa frontend "Backend not running" hatası verir.

---

## 3. Büyük Resim — Mimari

```
┌─────────────────────────────────────────────────────┐
│                  KULLANICI (Browser)                 │
│                  http://localhost:7861               │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (JSON / JPEG)
┌──────────────────────▼──────────────────────────────┐
│            FRONTEND  (frontend/gradio_app.py)        │
│            Gradio UI — Python                        │
│  • Kullanıcıdan path/ayar alır                       │
│  • Backend'e HTTP isteği atar                        │
│  • Gelen JSON'u tablolara / resimlere çevirir        │
│  • YOLO veya model koduna hiç dokunmaz               │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP REST API
┌──────────────────────▼──────────────────────────────┐
│            BACKEND   (backend/)                      │
│            FastAPI — port 8000                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  routers/  (URL → fonksiyon eşleştirmesi)    │   │
│  │    model.py     /api/model/*                 │   │
│  │    analysis.py  /api/analysis/*              │   │
│  │    export.py    /api/export/*                │   │
│  │    upload.py    /api/upload/*                │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │ fonksiyon çağrısı                  │
│  ┌──────────────▼───────────────────────────────┐   │
│  │  evaluator.py  (ModelEvaluator sınıfı)        │   │
│  │    • YOLO modelini yükler/çalıştırır          │   │
│  │    • Resimlere kutucuk çizer                  │   │
│  │    • Label dosyalarını okur                   │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│  ┌──────────────▼───────────────────────────────┐   │
│  │  state.py  (Singleton — global bellek)        │   │
│  │    Son çalıştırılan sonuçları RAM'de tutar    │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Neden backend/frontend ayrı?

**Frontend sadece HTTP bilir.** Gradio kodu `requests.post(...)` yapar, JSON alır, gösterir. YOLO import etmez, model dosyası açmaz. Bu sayede:
- Frontend'i istersen Svelte/React ile değiştirebilirsin, backend değişmez
- Backend'i farklı bir sunucuya taşıyabilirsin
- Frontend'de crash olursa backend çalışmaya devam eder

---

## 4. Dosya Yapısı

```
model_eval/
│
├── backend/                    ← Tüm model/hesap mantığı burada
│   ├── main.py                 ← FastAPI uygulaması, port 8000
│   ├── state.py                ← Global RAM belgisi (singleton)
│   ├── evaluator.py            ← ModelEvaluator sınıfı (asıl iş burada)
│   ├── gt_analyzer.py          ← Label dosyası okuma + IoU eşleştirme
│   ├── exporter.py             ← Excel export (12 sheet)
│   └── routers/
│       ├── model.py            ← /api/model/* endpoint'leri
│       ├── analysis.py         ← /api/analysis/* endpoint'leri
│       ├── export.py           ← /api/export/* endpoint'leri
│       └── upload.py           ← /api/upload/* endpoint'leri
│
├── frontend/
│   └── gradio_app.py           ← Tüm UI kodu (tek dosya)
│
├── requirements.txt
└── GUIDE.md                    ← Bu dosya
```

---

## 5. Backend — Dosya Dosya Açıklama

---

### 5.1 `main.py`

**Ne yapar:** FastAPI uygulamasını oluşturur, tüm router'ları bağlar, sunucuyu başlatır.

```python
app = FastAPI(...)

# CORS: Her yerden istek kabul et (Gradio farklı portta olduğu için gerekli)
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Router'ları ekle (her biri kendi URL prefix'ine sahip)
app.include_router(model.router)     # /api/model/*
app.include_router(analysis.router)  # /api/analysis/*
app.include_router(export.router)    # /api/export/*
app.include_router(upload.router)    # /api/upload/*

@app.get("/health")
def health():
    # Frontend her butona basmadan önce bunu çağırır
    # Model yüklü mü? Backend çalışıyor mu? kontrol eder
```

**`/health` endpoint neden var?**
Frontend'de her "Run" butonuna tıklandığında önce `/health` çağrılır. Backend cevap vermezse "Backend not running" hatası gösterilir, kullanıcı boşuna beklemez.

---

### 5.2 `state.py`

**Ne yapar:** Backend'in "çalışma belleği". Son çalıştırılan analiz sonuçlarını RAM'de tutar.

```python
evaluator              = ModelEvaluator()  # tek model instance
last_predict_results   = []  # son predict sonuçları (image başına dict)
last_val_metrics       = {}  # son val() sonuçları (mAP, P, R, vs.)
last_threshold_results = []  # threshold curve için ham veriler
last_raw_data          = []  # conf=0.01'deki tüm tahminler (threshold sweep için)
has_gt                 = False  # dataset'te label dosyası var mı?
run_history            = []  # tüm geçmiş çalışmalar (disk'e de yazılır)
```

**`run_history` kalıcı mı?**
Evet. `~/.model_eval_history.json` dosyasına yazılır. Backend her başladığında bu dosyayı okur. Yani uygulama kapatılıp açılsa bile geçmiş kaybolmaz.

**Neden global singleton?**
Sistem tek kullanıcı için tasarlandı (lokal test aracı). Birden fazla kullanıcı aynı anda kullanırsa sonuçlar karışır. İleride çok kullanıcıya geçilecekse Redis veya veritabanı kullanılmalı.

---

### 5.3 `evaluator.py`

**Ne yapar:** Tüm model işlemlerinin merkezi. YOLO modelini yükler, tahmin çalıştırır, label dosyalarını tarar, görüntüleri annotate eder.

#### Sınıf: `ModelEvaluator`

```python
class ModelEvaluator:
    model       # YOLO nesnesi
    model_path  # "C:/models/best.pt" gibi
    class_names # {0: "person", 1: "car", ...}
```

#### Metod: `load_model(model_path)`

```
.pt dosyasını açar → YOLO(path)
class_names sözlüğünü doldurur → {0: "person", 1: "car", ...}
```

Model yüklenince class_names otomatik dolur. Sonraki tüm işlemler bu sözlüğü kullanır.

---

#### Metod: `_find_image_dir(dataset_path)`

Dataset klasöründe resimleri nerede arayacağını bulur. YOLO'nun standart dizin yapısını dener:

```
dataset/val/images/  ← önce bunu dener
dataset/test/images/
dataset/images/
dataset/val/
dataset/test/
dataset/             ← son çare, kök dizin
```

Her adımda "bu klasörde gerçekten resim var mı?" diye kontrol eder.

---

#### Metod: `_create_yaml(dataset_path)`

YOLO'nun `model.val()` fonksiyonu, dataset'i tanımlamak için bir `.yaml` dosyası ister. Bu metod onu geçici olarak oluşturur.

```yaml
# Oluşturulan yaml örneği:
path: C:/datasets/cctv_test
train: val/images
val: val/images
nc: 13
names:
  0: person
  1: car
  ...
```

**Neden `yaml.dump()` kullanılmıyor, f-string kullanılıyor?**
`yaml.dump()` Windows yollarını bozuyor. `C:\Users\...` içindeki `\U` ve `\t` gibi karakterleri YAML escape sequence olarak yorumluyor ve yol bozuluyor. F-string + forward slash ile bu sorun tamamen ortadan kalktı.

---

#### Metod: `_scan_labels(dataset_path)`

Label `.txt` dosyalarını okuyarak her class için kaç görüntüde ve kaç instance'da göründüğünü sayar.

**Neden bu metod var?**
YOLO'nun `model.val()` terminale "627 images, 3200 instances" gibi bir tablo basar. Biz bu sayıları programatik olarak elde etmek için label dosyalarını kendimiz okuyoruz.

```
Label dosyası okuma akışı:

images/val/img001.jpg
  → labels/val/img001.txt dosyasını bul
     her satır: "0 0.5 0.3 0.2 0.1"  (class_id cx cy w h)
     class_id=0 → "person"
     inst_by_cls["person"] += 1
     img_by_cls["person"] += 1  (bu dosyada ilk defa görüldü)

→ Sonuç: {"person": 449, "car": 157, ...} (images)
          {"person": 686, "car": 836, ...} (instances)
```

---

#### Metod: `_print_val_table(...)`

YOLO'nun terminal çıktısını taklit eden tabloyu terminale basar:

```
                 Class     Images  Instances          P          R      mAP50   mAP50-95
────────────────────────────────────────────────────────────────────────────────────────
                   all        627       3200      0.923      0.813      0.882      0.732
                person        449        686      0.951      0.815      0.917       0.75
                   car        157        836      0.938      0.777      0.905      0.742
```

---

#### Metod: `run_validation(dataset_path, conf, iou)`

YOLO'nun resmi `model.val()` fonksiyonunu çalıştırır. Bu fonksiyon:
- Her görüntüyü model'e verir
- Tahminleri gerçek label'larla karşılaştırır
- AP (Average Precision) hesaplar
- Confusion matrix oluşturur

**Döndürdüğü veriler:**

| Alan | Açıklama |
|------|----------|
| `map50` | IoU≥0.50'de ortalama mAP (genel başarı skoru) |
| `map50_95` | IoU 0.50–0.95 arasında ortalama mAP (daha katı) |
| `mp` | Ortalama Precision |
| `mr` | Ortalama Recall |
| `total_images` | Kaç görüntü test edildi |
| `total_instances` | Toplam kaç nesne vardı |
| `class_metrics` | Her class için P/R/F1/mAP |
| `confusion_matrix` | Model neyi ne zannetti? |

---

#### Metod: `run_predict_with_gt(dataset_path, conf, iou_thresh)`

Bu metod `model.val()`'dan farklı çalışır:

1. **conf=0.01 ile tahmin yap** → çok düşük threshold, her şeyi yakala
2. **Her görüntü için GT label'larını yükle** (gt_analyzer kullanarak)
3. **İstenen conf threshold'unda filtrele** (örn. 0.25)
4. **IoU ile eşleştir** → TP/FP/FN hesapla
5. **Her görüntü için sonuçları döndür**

**Neden önce conf=0.01 ile çalıştırıyor?**
Threshold analizi için. "Eğer conf=0.50 kullansaydık ne olurdu?" sorusunu cevaplamak için aynı tahminleri farklı threshold'larla filtrelemek gerekiyor. Modeli 18 kez çalıştırmak yerine bir kez çalıştırıp sonuçları RAM'de saklıyoruz.

**`raw_data` ne?**
Her görüntü için conf=0.01'deki tüm tahminler + GT kutuları. Export sırasında threshold curve hesaplamak için kullanılır.

---

#### Metod: `run_threshold_analysis(raw_data, iou_thresh)`

0.10'dan 0.95'e kadar her conf değeri için Precision/Recall/F1 hesaplar. Excel'deki "Threshold_Curve" sheet'ini ve grafiği oluşturmak için kullanılır.

```
Threshold 0.10 → TP=2800, FP=800, FN=400 → P=0.78, R=0.88
Threshold 0.25 → TP=2600, FP=400, FN=600 → P=0.87, R=0.81
Threshold 0.50 → TP=2100, FP=200, FN=1100 → P=0.91, R=0.66
...
```

Bu tablo "hangi threshold en iyi F1 veriyor?" sorusunu cevaplar.

---

#### Metod: `get_comparison_image(image_path, tp_pairs, fp_preds, fn_gts)`

GT Comparison görüntüsünü çizer. OpenCV kullanır.

| Renk | Anlam |
|------|-------|
| Yeşil kutu | TP — Doğru tespit (GT kutusu) |
| Kırmızı kutu | FP — Yanlış tespit (GT karşılığı yok) |
| Turuncu kutu | FN — Kaçırılan nesne (tahmin yok ama GT var) |

---

#### Metod: `get_annotated_image(image_path, detections, highlight_idx)`

Normal tahmin görüntüsünü çizer. Her class farklı renkte. `highlight_idx` ile seçilen satırın kutucuğuna sarı halo eklenir.

---

### 5.4 `gt_analyzer.py`

**Ne yapar:** YOLO format `.txt` label dosyalarını okur ve tahminlerle eşleştirir.

#### YOLO Label Formatı

Her `.txt` dosyası bir görüntüye karşılık gelir. Her satır bir nesne:

```
0 0.512 0.341 0.198 0.245
│  │      │     │     │
│  └─cx   └─cy  └─w   └─h   (hepsi 0-1 arasında normalize edilmiş)
└─ class_id
```

Piksel koordinatlarına çevirmek için:
```
x1 = (cx - w/2) * image_width
y1 = (cy - h/2) * image_height
x2 = (cx + w/2) * image_width
y2 = (cy + h/2) * image_height
```

---

#### Fonksiyon: `find_label_path(image_path)`

Bir resim dosyasının label dosyasını arar. Desteklenen yapılar:

```
dataset/images/img001.jpg  →  dataset/labels/img001.txt
dataset/val/images/img.jpg →  dataset/val/labels/img.txt
dataset/img001.jpg         →  dataset/img001.txt  (flat layout)
```

---

#### Fonksiyon: `compute_iou(a, b)`

İki kutucuğun ne kadar örtüştüğünü hesaplar (Intersection over Union).

```
IoU = Kesişim Alanı / Birleşim Alanı

IoU = 0.0 → hiç örtüşme yok
IoU = 0.5 → yarısı örtüşüyor
IoU = 1.0 → tam üst üste
```

Bir tahminin TP mi FP mi olduğuna bu değere göre karar verilir.

---

#### Fonksiyon: `match_detections(gt_boxes, pred_boxes, iou_thresh)`

Her tahmini bir GT kutusuyla eşleştirir. Greedy (açgözlü) algoritma:
1. Tahminleri confidence'a göre büyükten küçüğe sırala
2. Her tahmin için en yüksek IoU'ya sahip GT kutusunu bul
3. IoU ≥ threshold ve aynı class ise → TP (eşleşme)
4. Eşleşen GT tekrar kullanılamaz

```
Örnek:
GT: [person@sol, person@sağ, car@orta]
Pred: [person@sol (0.95), car@orta (0.87), person@sol2 (0.43)]

Sıralama: 0.95 → 0.87 → 0.43

0.95 person → sol GT ile IoU=0.89 ✓ → TP
0.87 car    → orta GT ile IoU=0.91 ✓ → TP
0.43 person → sağ GT ile IoU=0.12 ✗ → FP (IoU çok düşük)

Kalan GT: [person@sağ] → FN (kaçırıldı)

Sonuç: TP=2, FP=1, FN=1
```

---

### 5.5 `exporter.py`

**Ne yapar:** Analiz sonuçlarını 12 sheet'li Excel dosyasına yazar. `xlsxwriter` kütüphanesi kullanır (openpyxl yerine, çünkü xlsxwriter grafik desteği daha iyi).

| Sheet | İçerik |
|-------|--------|
| Summary | Tek sayfalık özet (mAP, P, R, toplam sayılar) |
| Val_Class_Metrics | Class bazında P/R/F1/mAP (model.val'den) |
| Class_Performance | TP/FP/FN bazında class performansı (GT matching'den) |
| Per_Image | Her görüntü için tespit sayıları |
| Per_Image_GT | Her görüntü × class: GT/Pred/Kaçırılan |
| All_Detections | Her tek bounding box (class, conf, koordinat) |
| Class_Distribution | Class başına confidence istatistikleri |
| Confusion_Matrix | Model "A" derken aslında ne gördü? |
| Threshold_Curve | conf=0.10→0.95 için P/R/F1 + grafik |
| Size_Analysis | Küçük/Orta/Büyük nesne performansı |
| Worst_Images | En fazla kaçırılan görüntüler (Top 20) |
| Spatial_Bias | Görüntünün hangi bölgesinde daha çok hata? |

---

### 5.6 `routers/model.py`

**Endpoint'ler:**

| Method | URL | Açıklama |
|--------|-----|----------|
| GET | `/api/model/status` | Model yüklü mü? Hangi path? |
| POST | `/api/model/load` | Model yükle `{"model_path": "..."}` |

`load_model` sadece `evaluator.load_model()` çağırır ve sonucu döndürür. İş mantığı evaluator'da.

---

### 5.7 `routers/analysis.py`

**Endpoint'ler:**

| Method | URL | Açıklama |
|--------|-----|----------|
| POST | `/api/analysis/validate` | Sadece model.val() çalıştır |
| POST | `/api/analysis/predict` | Sadece per-image predict çalıştır |
| POST | `/api/analysis/both` | Her ikisini birden çalıştır |
| GET | `/api/analysis/image/annotated` | Annotated görüntü döndür (JPEG) |
| GET | `/api/analysis/image/comparison` | GT comparison görüntüsü döndür |
| GET | `/api/analysis/history` | Tüm çalıştırma geçmişi |
| DELETE | `/api/analysis/history` | Geçmişi temizle |

**`_make_predict_entry()` yardımcı fonksiyon:**
Her çalıştırmadan sonra history kaydı oluşturur. TP/FP/FN toplayıp Precision/Recall/F1 hesaplar.

**`_model_name()` yardımcı fonksiyon:**
Model path'inden sadece dosya adını çeker: `C:\models\best_v2.pt` → `best_v2.pt`

---

### 5.8 `routers/export.py`

**Endpoint:** `POST /api/export/excel`

Excel export endpoint'i lazy (tembel) çalışır:
- Threshold curve hesabı pahalı (tüm görüntüler için 18 farklı threshold)
- Bu yüzden predict sırasında değil, **sadece export istendiğinde** hesaplanır
- `state.last_threshold_results` doluysa tekrar hesaplamaz

---

### 5.9 `routers/upload.py`

**Endpoint:** `POST /api/upload/dataset`

Kullanıcı browser'dan `.zip` yüklediğinde:
1. Zip'i geçici klasöre çıkart
2. `_find_dataset_root()` ile gerçek dataset kökünü bul
3. Path'i frontend'e döndür → frontend `dataset_inp` kutusuna yazar

**`_find_dataset_root()` nasıl çalışır?**
Zip içindeki klasör yapısına bakar. `val/images`, `test/images`, `images` içeren bir yapı arar ve o yapının kökünü döndürür.

```
zip içeriği:
  my_dataset/
    val/
      images/
        img001.jpg
      labels/
        img001.txt

→ my_dataset/ döndürülür (val/images'ın 2 üstü)
```

---

## 6. Frontend — `gradio_app.py`

**Ne yapar:** Gradio ile web arayüzü oluşturur. Backend'e HTTP istek atar, gelen veriyi tablolara/resimlere çevirir. **Model kodu içermez.**

### Temel Prensipler

- `API_BASE = "http://localhost:8000"` — tüm istekler buraya gider
- `_post(endpoint, payload)` — JSON body ile POST
- `_get(endpoint, params)` — GET
- Model import yok, sadece `requests`, `pandas`, `gradio`, `PIL`

### State Management

Gradio'da backend gibi global değişken kullanmak yerine `gr.State()` kullanılır:

```python
val_state     = gr.State({})    # Son validation sonuçları
predict_state = gr.State([])    # Son predict sonuçları (image listesi)
has_gt_state  = gr.State(False) # GT label var mı?
```

Bu state'ler callback fonksiyonlar arasında aktarılır, backend'e tekrar istek atmak gerekmez.

### Önemli Callback Fonksiyonlar

| Fonksiyon | Ne Yapar |
|-----------|----------|
| `load_model_cb` | `/api/model/load` çağırır, class listesini gösterir |
| `run_val_cb` | `/api/analysis/validate` çağırır, tabloyu günceller |
| `run_predict_cb` | `/api/analysis/predict` çağırır |
| `run_both_cb` | `/api/analysis/both` çağırır, hem val hem predict sonuçlarını günceller |
| `image_load_cb` | Dropdown'dan resim seçilince annotated resmi getirir |
| `image_filter_cb` | Class filtresi değişince resmi yeniden çizer |
| `on_det_select` | Tabloda satıra tıklanınca o kutucuğu vurgular |
| `view_comparison_cb` | GT Comparison görüntüsünü ve tablosunu getirir |
| `export_cb` | Excel oluşturulmasını tetikler, indirme linki verir |
| `export_history_csv_cb` | Run History'yi CSV olarak indirir |

### DataFrame Oluşturucu Fonksiyonlar

| Fonksiyon | Açıklama |
|-----------|----------|
| `_val_combined_df(val_data, overall_p, overall_r)` | Validation Metrics tablosu (YOLO formatı) |
| `_per_image_df(results, has_gt)` | Per-Image Results tablosu |
| `_worst_images_df(results, has_gt)` | En kötü 20 görüntü tablosu |
| `_history_df()` | Run History tablosu (mAP50'ye göre sıralı) |
| `_build_det_df(dets)` | Image Viewer'daki tespit tablosu |

### `run_both_cb` — GT Matching P/R Hesabı

```python
# Validation Metrics tablosunun "all" satırındaki P/R,
# model.val()'den değil, GT matching'den geliyor.
# Bu sayede Run History ile tutarlı.

total_tp = sum(r.get("tp", 0) for r in results)
total_fp = sum(r.get("fp", 0) for r in results)
total_fn = sum(r.get("fn", 0) for r in results)
overall_p = total_tp / (total_tp + total_fp)  # GT-matching Precision
overall_r = total_tp / (total_tp + total_fn)  # GT-matching Recall
```

---

## 7. Temel Kavramlar — Metrikler Ne Anlama Gelir?

### TP / FP / FN

Sistemde 3 tür sonuç var:

```
Gerçekte adam var, model de buldu  → TP (True Positive)  ✅
Gerçekte adam yok, model uydurdu   → FP (False Positive) ❌
Gerçekte adam var, model kaçırdı   → FN (False Negative) ❌
```

### Precision (Kesinlik)

> "Modelin bulduklarının kaçı gerçekten doğruydu?"

```
Precision = TP / (TP + FP)

Örnek: Model 100 nesne buldu, 85'i gerçekten oradaydı
Precision = 85 / (85 + 15) = 0.85
```

Precision düşükse → model çok uydurma yapıyor (yanlış alarm fazla)

### Recall (Duyarlılık / Hatırlama)

> "Gerçekte olan nesnelerin kaçını model bulabildi?"

```
Recall = TP / (TP + FN)

Örnek: Dataset'te 200 nesne var, model 160'ını buldu
Recall = 160 / (160 + 40) = 0.80
```

Recall düşükse → model çok şeyi kaçırıyor

### Güvenlik uygulamalarında ne istiyoruz?

**Düşme tespiti gibi kritik senaryolarda Recall öncelikli:**

- Düşen bir kişiyi kaçırmak (FN) → hayati tehlike
- Yanlış alarm vermek (FP) → sadece rahatsız edici

Yüksek Recall için → conf threshold'u düşür (daha az şey kaçır, ama daha fazla yanlış alarm)

### mAP50

> "Model tüm class'larda ortalama ne kadar iyi?"

- Her class için Precision-Recall curve çizilir
- Bu curve'nin altındaki alan = AP (Average Precision)
- Tüm class'ların AP ortalaması = mAP
- "50" → IoU eşiği 0.50 (kutucuk en az %50 örtüşmeli)

```
mAP50 = 0.90 → çok iyi
mAP50 = 0.70 → orta
mAP50 = 0.50 → kötü
```

### IoU (Intersection over Union)

Tahmin edilen kutu ile gerçek kutunun ne kadar örtüştüğü:

```
┌─────────┐
│  GT     │
│    ┌────┼───┐
│    │ ∩  │   │← Pred
└────┼────┘   │
     └────────┘

IoU = ∩ alanı / ∪ alanı
```

### İki Farklı P/R Hesabı

Sistemde **iki farklı** Precision/Recall hesabı yapılıyor:

| Kaynak | Nasıl Hesaplanıyor | Nerede Gösterilir |
|--------|-------------------|-------------------|
| `model.val()` — AP tabanlı | Precision-Recall curve'den | Val Metrics: per-class satırlar |
| GT Matching — TP/FP/FN tabanlı | Direkt sayım | Val Metrics: "all" satırı, Run History |

**Neden farklı?** AP hesabı curve'in altındaki alanı ölçer (daha akademik). TP/FP/FN hesabı verilen conf threshold'undaki anlık değeri verir (daha pratik). "all" satırında pratik olanı göstermeyi tercih ettik çünkü Run History ile tutarlı.

---

## 8. Veri Akışı — Butona Tıkladığında Ne Olur?

### "Load Model" butonu

```
Kullanıcı path yazar → Load Model butonuna basar
  ↓
load_model_cb("C:/models/best.pt")
  ↓
POST /api/model/load {"model_path": "C:/models/best.pt"}
  ↓
routers/model.py → evaluator.load_model("C:/models/best.pt")
  ↓
YOLO("C:/models/best.pt") → model yüklenir, class_names dolar
  ↓
{"ok": true, "classes": {0: "person", ...}} döner
  ↓
UI'da "✓ Model Ready" + class listesi gösterilir
```

---

### "Run Both" butonu

```
Kullanıcı Run Both'a basar
  ↓
run_both_cb(dataset_path, conf=0.25, iou=0.45, iou_thresh=0.50)
  ↓
POST /api/analysis/both {...}
  ↓
routers/analysis.py → both() fonksiyonu

  [1] evaluator.run_validation(dataset_path, conf=0.25, iou=0.45)
      ↓
      _scan_labels() → label .txt dosyaları okunur → Images/Instances sayılır
      _create_yaml() → geçici yaml oluşturulur
      model.val()    → YOLO tüm görüntüleri tarar, AP hesaplar
      _print_val_table() → terminale tablo basılır
      → mAP50, P, R, per-class metrics döner

  [2] evaluator.run_predict_with_gt(dataset_path, conf=0.25, iou_thresh=0.50)
      ↓
      model.predict(conf=0.01) → tüm görüntüler, çok düşük threshold
      Her görüntü için:
        gt_analyzer.load_gt_boxes() → label .txt okunur
        conf=0.25 ile filtrele
        gt_analyzer.match_detections() → IoU ile eşleştir
        → TP, FP, FN sayılır
      → predict_results, raw_data döner

  [3] History'ye kaydet → ~/.model_eval_history.json güncellenir

  → JSON response döner
  ↓
Frontend tarafında:
  - Validation Metrics tablosu güncellenir
  - Image dropdown dolar
  - Worst Images tablosu güncellenir
  - History tablosu güncellenir
```

---

### "Show GT Comparison" butonu

```
Kullanıcı bir resim seçer → Show GT Comparison'a basar
  ↓
view_comparison_cb(image_name, predict_state, has_gt=True)
  ↓
GET /api/analysis/image/comparison?image_name=img001.jpg
  ↓
state.last_predict_results içinde o resmi bul
  → tp_pairs, fp_preds, fn_gts zaten hesaplanmış (predict sırasında)
evaluator.get_comparison_image() → OpenCV ile çiz → PIL Image döner
  ↓
JPEG olarak stream et
  ↓
Frontend'de comparison_img gösterilir
Frontend ayrıca predict_state'den TP/FP/FN tablosunu da oluşturur
```

---

### "Generate Excel" butonu

```
Kullanıcı Generate Excel'e basar
  ↓
export_cb()
  ↓
POST /api/export/excel
  ↓
routers/export.py → export_excel()
  ↓
  state.last_threshold_results boşsa:
    state.last_raw_data varsa:
      evaluator.run_threshold_analysis(raw_data)  ← lazy hesap
    yoksa:
      evaluator.run_threshold_analysis_from_dir(...)  ← fallback
  ↓
exporter.export_to_excel(predict_results, val_metrics, class_names, ...)
  → 12 sheet'li Excel dosyası oluşturulur
  ↓
FileResponse olarak döner
  ↓
Frontend geçici dosyaya yazar → Download linki gösterilir
```

---

## 9. API Referansı

Tüm endpoint'ler `http://localhost:8000/docs` adresinde interaktif denenebilir.

### Model

```
GET  /health                          Backend durumu + model yüklü mü?
GET  /api/model/status                Model detayları
POST /api/model/load                  {"model_path": "..."}
```

### Analiz

```
POST /api/analysis/validate           {"dataset_path", "conf", "iou"}
POST /api/analysis/predict            {"dataset_path", "conf", "iou_thresh"}
POST /api/analysis/both               {"dataset_path", "conf", "iou", "iou_thresh"}

GET  /api/analysis/image/annotated    ?image_name=&classes=&highlight_idx=
GET  /api/analysis/image/comparison   ?image_name=

GET  /api/analysis/history
DELETE /api/analysis/history
```

### Export & Upload

```
POST /api/export/excel
POST /api/upload/dataset              multipart/form-data, field: "file" (.zip)
```

### Parametre Açıklamaları

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `conf` | 0.25 | Min confidence — altındaki tahminler gösterilmez |
| `iou` | 0.45 | model.val() için IoU threshold |
| `iou_thresh` | 0.50 | GT matching için min IoU (TP sayılabilmek için) |

---

## 10. Dataset Dizin Yapısı

Sistem şu yapıları otomatik tanır:

### Yapı 1 — Standart YOLO (Önerilen)
```
dataset_root/
  images/
    val/
      img001.jpg
      img002.jpg
  labels/
    val/
      img001.txt    ← img001.jpg ile aynı isim
      img002.txt
```

### Yapı 2 — Alt Klasörlü
```
dataset_root/
  val/
    images/
      img001.jpg
    labels/
      img001.txt
```

### Yapı 3 — Düz (Flat)
```
dataset_root/
  img001.jpg
  img001.txt
  img002.jpg
  img002.txt
```

### Label Formatı (YOLO)
```
# class_id  cx    cy    width  height  (normalize 0-1)
0            0.512 0.341 0.198  0.245
1            0.231 0.710 0.312  0.180
```

---

## 11. Excel Export — Sheet Açıklamaları

| Sheet | Sütunlar | Kullanım Amacı |
|-------|----------|----------------|
| **Summary** | Model, Dataset, mAP50, P, R, F1, TP, FP, FN | Tek bakışta genel başarı |
| **Val_Class_Metrics** | Class, Images, Instances, P, R, F1, mAP50, mAP50-95 | model.val() sonuçları, YOLO konsoluyla aynı |
| **Class_Performance** | Class, TP, FP, FN, Precision, Recall, F1 | GT matching bazlı class analizi |
| **Per_Image** | Image, Total_Pred, Total_GT, TP, FP, FN, Match_Rate | Hangi görüntü iyi/kötü? |
| **Per_Image_GT** | Image × Class: GT / Pred / Missed / Extra | Detaylı cross-tabulation |
| **All_Detections** | Image, Class, Conf, X1, Y1, X2, Y2, W, H, Area | Ham tespit verisi |
| **Class_Distribution** | Class, Count, Avg_Conf, Min_Conf, Max_Conf | Confidence dağılımı |
| **Confusion_Matrix** | Predicted vs Actual (heatmap) | model.val()'den, hangi class nereyle karışıyor? |
| **Threshold_Curve** | Conf, TP, FP, FN, P, R, F1 + grafik | Optimal threshold bulmak için |
| **Size_Analysis** | Size (S/M/L), Count, Recall, Avg_IoU | Küçük nesnelerde problem var mı? |
| **Worst_Images** | Image, FN_Missed, FP_Extra, TP, Avg_Conf | En sorunlu görüntüler |
| **Spatial_Bias** | 3×3 grid heatmap | Köşelerde mi, ortada mı daha çok hata? |

---

## 12. Bilinen Kısıtlamalar ve Gelecek Geliştirmeler

### Mevcut Kısıtlamalar

**Tek kullanıcı:**
`state.py` global singleton kullanıyor. Aynı anda iki kişi farklı model çalıştırırsa sonuçlar karışır. Çok kullanıcı için Redis + kullanıcı oturumu gerekir.

**Farklı class sayılı model + dataset:**
Model 10 class, dataset 13 class label içeriyorsa YOLO val() sırasında "class 12 exceeds dataset class count" uyarısı verir ve o label'ları yoksayar. Bu label'lar:
- `model.val()` metriklerine yansımaz (görmezden gelinir)
- GT matching tarafında ise FN olarak sayılır (tutarsızlık)

Gelecekteki çözüm: Label scan ile dataset'teki gerçek class sayısını tespit edip yaml'ı buna göre ayarlamak.

**Büyük dataset'lerde yavaşlık:**
`run_predict_with_gt()` conf=0.01 ile çalışıyor (threshold sweep için). Bu çok sayıda küçük confidence box üretiyor, işlem süresi uzuyor.

### Geliştirme Fikirleri

- [ ] Farklı class sayılı model/dataset desteği (label scan + yaml nc genişletme)
- [ ] Video dosyası desteği
- [ ] Batch model karşılaştırma (aynı dataset'te N model)
- [ ] Çok kullanıcı desteği (job queue + oturum)
- [ ] PR curve grafiği Gradio'da gösterme
