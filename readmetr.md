# Danusys Model Eval Platform (Türkçe)

> YOLO tabanlı model değerlendirme platformu — Gerçek CCTV görüntülerinde model performansını doğrular ve birden fazla modeli karşılaştırır.

---

## Amaç

Eğitilen YOLO modellerinin **gerçek saha CCTV görüntülerinde** ne kadar iyi çalıştığını doğrulamak için geliştirilmiş dahili bir değerlendirme aracıdır.

- Modelin eğitim sırasında görmediği gerçek CCTV görüntüleri üzerinde tespit performansını ölçer.
- Aynı görüntü seti üzerinde **en fazla 3 modeli aynı anda karşılaştırarak** sahaya alınacak modeli seçmek ya da yeniden eğitim yönüne karar vermek için kullanılır.
- Validation (nicel metrikler) ve Per-Image (görsel bazlı tespit detayı) iki perspektifi birlikte sunar.

---

## Analiz Akışı

```
1. Model Yükle    Browse ile .pt dosyası seç → Load Model'e bas → "✓ Model Ready" + sınıf listesi görünür
      ↓
2. Dataset Seç    Değerlendirilecek CCTV görüntü klasörünü gir veya Browse ile seç
      ↓
3. Parametreler   Confidence / IoU kaydırıcılarını ayarla (varsayılan değerlerle başlamak önerilir)
      ↓
4. Run Both       Validation + Per-Image birlikte çalıştır (Per-Image tek başına istatistik üretmez — Run Both önerilir)
      ↓
5. Sonuçları İncele  Sekmelerde sınıf metrikleri · görsel görüntüleyici · Worst Images · geçmiş incele
      ↓
6. Raporu Kaydet  Tüm sonuçları Excel dosyası olarak dışa aktar
```

> **Validation ile Per-Image Farkı**
> Per-Image tek başına çalıştırıldığında yalnızca görseller üzerinde tahmin (predict) yapar; mAP · P · R gibi istatistiksel metrikler üretmez.
> Validation ise GT etiket dosyalarına ihtiyaç duyar ve toplu metrikleri hesaplar.
> Her iki metriği birlikte görmek için **Run Both** kullanın.

---

## Arayüz Yapısı

> *(Ekran görüntüsü eklenecek)*

### Sol Panel — Ayarlar

| Alan | Açıklama |
|---|---|
| Model path | `.pt` dosya yolunu doğrudan gir veya Browse ile seç. Upload akordeonu ile PC'den doğrudan yükleme de yapılabilir |
| Load Model | Tıklandığında model backend'e yüklenir. Başarılıysa Status'ta `✓ Model Ready` görünür, Classes akordeonunda sınıf listesi açılır |
| Dataset folder | Görüntü klasörü yolunu gir veya Browse ile seç. ZIP yükleme de desteklenir (görüntü + etiket) |
| Confidence threshold | Bu değer ve üzerindeki tahminler gösterilir (varsayılan 0.50, sabit) |
| IoU threshold | `model.val()` içinde kullanılan IoU (varsayılan 0.45) |
| GT match IoU | Tahmin kutusuyla GT kutusunun örtüşmesi bu değerin üzerinde olduğunda TP sayılır (varsayılan 0.50) |
| Run Butonları | Validation · Per-Image ayrı ayrı, **Run Both** her ikisi birlikte |

---

### Sağ Panel — Sonuç Sekmeleri

#### 📊 Validation Metrics

`model.val()` çıktısı olan toplu metrik tablosudur. GT etiket dosyaları gereklidir.

| Sütun | İçerik |
|---|---|
| Class | Sınıf adı (ilk satır `all` — tüm sınıfların ortalaması) |
| Images | İlgili sınıfa ait örnek bulunan görüntü sayısı |
| Instances | GT örnek sayısı |
| Precision / Recall / F1 | Kesinlik · Duyarlılık · F1 |
| mAP50 / mAP50-95 | IoU 0.50 bazlı ve 0.50~0.95 ortalama mAP |

---

#### 🖼️ Image Viewer

Analiz edilen görselleri Bounding Box ile birlikte görsel olarak inceler.

**Prediction alt sekmesi**

- Görüntü açılır listesinden bir dosya seçildiğinde Bounding Box çizili görsel yüklenir.
- **Class Filter** — Görmek istediğin sınıfları seç, yalnızca o sınıfların kutuları görünür. (Çoklu seçim desteklenir)
- Alt taraftaki **Detections tablosu** — Her tespiti için sınıf, Confidence, koordinat ve boyut listelenir.
- Tabloda bir satıra tıklandığında o tespit görselde **vurgulanan** (highlight) kutu ile belirtilir.

**GT Comparison alt sekmesi**

GT etiket dosyaları varsa aktif olur. Tahmin ile GT'yi renk kodlarıyla karşılaştırır.

| Renk | Anlam |
|---|---|
| Yeşil | TP — Doğru tespit |
| Kırmızı | FP — Hatalı tespit (GT eşleşmesi yok) |
| Turuncu | FN — Kaçırılan nesne |

Alt tabloda görsel bazında TP / FP / FN ayrıntıları listelenir.

---

#### ⚠️ Worst Images

Analiz sonuçları içinde performansı en düşük olan ilk 20 görüntüyü gösterir.

- GT etiketi varsa → En çok kaçırılan nesne (FN) sayısına göre sıralı
- GT etiketi yoksa → Hiç tespit edilemeyen veya Confidence'ı en düşük görseller önce

Yeniden eğitim için veri seçimi ya da hata analizi aşamasında kullanılır.

---

#### 📋 Run History

Her çalıştırma otomatik olarak kaydedilir ve oturum kapandıktan sonra da korunur (`~/.model_eval_history.json`).

Her kayıtta model adı, dataset yolu, çalıştırma zamanı, görüntü sayısı, tespit sayısı, mAP50, Precision, Recall ve benzeri bilgiler yer alır. Farklı modelleri sırayla test ederken **karşılaştırma referansı** olarak kullanılır.

- **mAP50 değerine göre azalan sırada** otomatik sıralanır.
- **Export CSV** ile tüm geçmiş CSV olarak indirilebilir.

---

#### 🔀 Model Comparison

Aynı dataset üzerinde **en fazla 3 modeli** sırayla çalıştırarak sonuçları yan yana karşılaştırır.

- Her model için mAP50 · mAP50-95 · Precision · Recall · TP · FP · FN metrik kartı gösterilir.
- **Yakalama Oranı tablosu** — Görüntü bazında GT sayısına karşılık her modelin kaç nesneyi yakaladığı (TP/FP) karşılaştırılır. Alt satırlarda toplam ve Recall(%) özeti otomatik eklenir.
- **Görsel görüntüleyici** — Açılır listeden görüntü seçildiğinde 3 modelin Bounding Box sonuçları yan yana gösterilir.

> Sol paneldeki Confidence / IoU kaydırıcıları karşılaştırma çalıştırması için de geçerlidir.

---

#### 📈 Per-Image Results

Tüm görüntüler için ham tespit verisi tablosudur.

GT etiketi varsa GT sayısı, TP, FP, FN ve Match Rate sütunları otomatik olarak eklenir. Sınıf bazlı tespit sayısı ve ortalama Confidence da birlikte gösterilir.

---

#### 💾 Export

Analiz sonuçlarının tamamını Excel dosyası olarak dışa aktarır. Summary · Val_Class_Metrics · Class_Performance · Per_Image · Per_Image_GT · All_Detections · Class_Distribution · Confusion_Matrix · Threshold_Curve · Size_Analysis · Worst_Images · Spatial_Bias olmak üzere 12 sayfa içerir.

---

## Sistem Genel Bakış

| Alan | Detay |
|---|---|
| Backend | FastAPI · `localhost:8001` |
| Frontend | Gradio · `localhost:7860` |
| Desteklenen Model | YOLOv8 `.pt` |

Swagger UI → `http://localhost:8001/docs`

---

## Çalıştırma

```bash
pip install -r requirements.txt

# Backend (Terminal 1)
python backend/main.py

# Frontend (Terminal 2)
python frontend/gradio_app.py
```

---

## Dataset Klasör Yapısı

Validation için görüntülerle aynı dizinde YOLO formatı etiket dosyaları (`.txt`) gereklidir.

```
root/val/images/*.jpg  +  root/val/labels/*.txt   ← Standart YOLO
root/images/*.jpg
root/*.jpg                                         ← Düz yapı
```

---

## Proje Yapısı

```
model_eval/
├── backend/
│   ├── main.py          # FastAPI giriş noktası
│   ├── evaluator.py     # YOLO model yükle / çıkarsama / annotate
│   ├── exporter.py      # Excel çıktısı
│   ├── state.py         # Singleton: model + son sonuçlar
│   └── routers/         # model · analysis · export · upload
├── frontend/
│   └── gradio_app.py    # Gradio UI (yalnızca REST çağrısı yapar)
└── requirements.txt
```

Backend ve Frontend tamamen ayrıdır. Frontend yalnızca REST çağrısı yaptığından Gradio başka bir UI framework ile değiştirilebilir, Backend'e dokunulması gerekmez.
