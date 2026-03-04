# Model Eval Platform

YOLO tabanlı model değerlendirme platformu. İki modda çalışır:
- **Validation** — ground-truth etiketlere karşı aggregate metrikler (mAP, P, R)
- **Per-Image** — her görsel için tespit sayısı, confidence ve annotated önizleme

---

## Mimari

```
model_eval/
├── backend/                  ← FastAPI REST API (port 8000)
│   ├── main.py               # Uygulama giriş noktası
│   ├── state.py              # Singleton: model + son sonuçlar
│   ├── evaluator.py          # YOLO load / val / predict / annotate
│   ├── exporter.py           # Excel çıktısı
│   └── routers/
│       ├── model.py          # /api/model/*
│       ├── analysis.py       # /api/analysis/*
│       └── export.py         # /api/export/*
├── frontend/
│   └── gradio_app.py         # Gradio UI (port 7860) — sadece HTTP çağrısı yapar
└── requirements.txt
```

**Backend ve frontend ayrı process.** Frontend yalnızca REST çağrısı yapar —
ileride Gradio yerine Svelte/React yazılabilir, backend'e dokunulmaz.

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Çalıştırma

```bash
# Terminal 1 — Backend
python backend/main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger)

# Terminal 2 — Frontend
python frontend/gradio_app.py
# → http://localhost:7860
```

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| Model yükleme | `.pt` dosyası — path yazarak veya `📁` ile seçerek |
| Dataset seçimi | Klasör — path yazarak veya `📁` ile seçerek |
| Validation | `model.val()` — P / R / F1 / mAP50 / mAP50-95 sınıf bazlı |
| Per-Image | `model.predict()` — görsel başına tespit count + confidence |
| Image Viewer | Bounding box çizili görsel önizleme |
| Quick Stats | Sınıf bazlı min/max/avg confidence |
| Excel Export | 5 sayfalı detaylı rapor |
| API | FastAPI Swagger UI ile doğrudan test edilebilir |

### Dataset klasör yapıları (otomatik algılanır)

```
root/val/images/*.jpg   ← standart YOLO (validation için etiket gerekir)
root/images/*.jpg
root/*.jpg              ← düz yapı
```

---

## Ekran Görüntüsü — Sekmeler

```
⚙️ Model          📊 Validation    🖼️ Per-Image    🔎 Görsel    📈 Stats    💾 Export
─────────────────────────────────────────────────────────────────────────────────────
Model path [___] 📁  │ Genel mAP özeti   │ Görsel başına  │ Dropdown   │ ...
Model Yükle          │ Sınıf bazlı tablo │ sınıf sayıları │ + BBox img │
─────────────────    │                   │ + confidence   │ + det tbl  │
Dataset path [_] 📁  │
Confidence [0.25]    │
IoU        [0.45]    │
─────────────────    │
[Validation]         │
[Per-Image  ]        │
[İkisini Çalıştır▶]  │
```

---

## Excel Çıktısı — Sayfa Örnekleri

### 1. `Summary`

| Metric | Value |
|---|---|
| Export Timestamp | 20260304_142035 |
| Total Images Analyzed | 34 |
| Total Detections | 241 |
| mAP50 | 0.55 |
| mAP50-95 | 0.421 |
| Mean Precision | 0.534 |
| Mean Recall | 0.577 |

---

### 2. `Val_Class_Metrics`

| class | class_id | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| person | 0 | 0.562 | 0.772 | 0.651 | 0.754 | 0.592 |
| car | 1 | 0.679 | 0.774 | 0.724 | 0.823 | 0.676 |
| falldown | 2 | 0.650 | 0.457 | 0.537 | 0.493 | 0.364 |
| bus | 3 | 0.962 | 1.000 | 0.981 | 0.995 | 0.796 |
| truck | 4 | 0.375 | 0.492 | 0.426 | 0.308 | 0.231 |
| motorcycle | 6 | 0.126 | 0.143 | 0.134 | 0.153 | 0.054 |

---

### 3. `Per_Image`

| Image | Total | person | person_conf | car | car_conf | falldown | falldown_conf |
|---|---|---|---|---|---|---|---|
| cam01_0001.jpg | 8 | 3 | 0.812 | 4 | 0.743 | 1 | 0.561 |
| cam01_0002.jpg | 5 | 2 | 0.774 | 3 | 0.698 | 0 | |
| cam02_0001.jpg | 12 | 7 | 0.841 | 3 | 0.712 | 2 | 0.489 |
| cam03_0001.jpg | 1 | 0 | | 0 | | 1 | 0.634 |

---

### 4. `All_Detections`

| image | class | confidence | x1 | y1 | x2 | y2 | width | height | area |
|---|---|---|---|---|---|---|---|---|---|
| cam01_0001.jpg | person | 0.921 | 120.3 | 45.1 | 198.7 | 312.4 | 78.4 | 267.3 | 20956 |
| cam01_0001.jpg | car | 0.874 | 340.0 | 180.2 | 620.5 | 380.8 | 280.5 | 200.6 | 56260 |
| cam01_0001.jpg | falldown | 0.561 | 50.0 | 400.0 | 210.0 | 490.0 | 160.0 | 90.0 | 14400 |

---

### 5. `Class_Distribution`

| Class | Total_Detections | Images_With_Det | Avg_Confidence | Min_Confidence | Max_Confidence |
|---|---|---|---|---|---|
| person | 87 | 28 | 0.812 | 0.251 | 0.976 |
| car | 115 | 26 | 0.743 | 0.261 | 0.981 |
| falldown | 12 | 10 | 0.531 | 0.252 | 0.712 |
| bus | 3 | 1 | 0.941 | 0.891 | 0.971 |
| motorcycle | 7 | 3 | 0.341 | 0.251 | 0.481 |

---

## API Endpoint Özeti

| Method | Endpoint | Açıklama |
|---|---|---|
| GET | `/health` | Backend durumu |
| GET | `/api/model/status` | Yüklü model bilgisi |
| POST | `/api/model/load` | Model yükle `{"model_path": "..."}` |
| POST | `/api/analysis/validate` | Validation çalıştır |
| POST | `/api/analysis/predict` | Per-image predict |
| POST | `/api/analysis/both` | İkisini birden çalıştır |
| GET | `/api/analysis/image/annotated` | Annotated görsel stream |
| POST | `/api/export/excel` | Excel dosyası indir |
