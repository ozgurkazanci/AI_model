# ASIC Devre Tasarımı için Özel AI Modeli — Tasarım Dokümanı

**Sürüm:** 2.0
**Tarih:** 29 Ağustos 2026
**Durum:** SFT eğitimi tamamlandı, 35B cloud eğitimi hazır, 1050 SFT, 199 test, 78 eval

> Bu doküman bir kodlama ajanına (Antigravity / Claude Code) verilmek üzere yazılmıştır.
> Teknik terimler, API isimleri ve kod İngilizce bırakılmıştır.

---

## 0. Karar Özeti Tablosu

| # | Karar Alanı | Karar | Gerekçe |
|---|-------------|-------|---------|
| 1 | Proje tipi | Mevcut açık ağırlıklı modelin alan-uzmanlaştırılması | Sıfırdan pretraining bütçe dışı ve gereksiz |
| 2 | Model rolü | Ajan — araç çağıran, döngüde kalan | Tek seferlik netlist üreteci yetersiz |
| 3 | Base model | Qwen3.6-35B-A3B (MoE, agentic-coding tuned, Apache 2.0) | Ajan davranışı için ayarlanmış, ticari kullanım serbest |
| 4 | Prototip modeli | Küçük dense varyant (Qwen3.x 9B/27B sınıfı) | Dense fine-tune MoE'den kolay; hatalar ucuza yakalanır |
| 5 | Base model bağlılığı | Yok — kod model-agnostik yazılacak | Qwen4 çıkacak; veri kalıcı, base kiralık |
| 6 | Tokenizer | Sınırlı genişletme: SI ön ekleri, birimler, device isimleri | Devre tasarımı sayı işi; tam değişim base bilgisini bozar |
| 7 | Bağlam | Base'in native bağlamı + retrieval | Uzatma (YaRN vb.) sonraya; erken optimizasyon |
| 8 | Adapter | LoRA, r=32, attention + MLP | Ucuz, hızlı, geri alınabilir. Full FT ancak kanıtlanınca |
| 9 | Eğitim aşamaları | CPT → SFT (ajan yörüngeleri) → RL (GRPO) | Bilgi → davranış → beceri |
| 10 | Ödül kaynağı | Simülatör (nabla), insan değil | Doğrulanabilir ödül; alanın en büyük avantajı |
| 11 | Analog stratejisi | LLM + sayısal optimizatör hibrit | LLM sürekli sayısal aramada zayıf |
| 12 | Dijital stratejisi | Saf LLM + formal/testbench doğrulama | RTL üretimi zaten dil işi |
| 13 | PDK bilgisi | Ağırlıkta değil, retrieval ile bağlamda | NDA uyumu + PDK bağımsızlığı |
| 14 | İnşa sırası | Sistem arayüzü → eval seti → veri → eğitim | Arayüz olmadan veri üretilemez |
| 15 | İsimlendirme | Ticari araç isimleri kullanılmayacak | Marka riski + model halüsinasyonu |

---

## 1. Proje Tanımı

### Amaç
ASIC analog ve dijital CMOS devre tasarımı yapabilen, alan-uzmanı bir dil modeli üretmek.
Model, ayrı bir projede geliştirilen nabla simülatör motoru ve EDA araç setini kullanarak tasarım yapacak.

### Model rolünün netleştirilmesi
Model, spec verildiğinde tek seferde netlist basan bir üreteç **değildir**. Model bir **ajandır**:

```
spec oku → topoloji seç → netlist üret → simüle et → sonucu oku
    → eksiği teşhis et → düzelt → tekrar simüle et → spec tutana kadar devam
```

Bu ayrım her şeyi belirler: veri formatı, eğitim yöntemi, gereken compute, sistem mimarisi.

### Kapsam dışı
- Yeni Transformer mimarisi / attention mekanizması icadı
- Sıfırdan pretraining
- Ticari PDK verisiyle eğitim (NDA)

---

## 2. Base Model

### Seçim: Qwen3.6-35B-A3B

**Gerekçeler:**
- Ajan tipi kodlama için özel ayarlanmış — bizim kullanım şeklimize doğrudan oturuyor
- Apache 2.0 lisansı — ticari kullanım, değiştirme, dağıtma serbest
- MoE mimarisi — 35B toplam, 3B aktif; RL döngüsünde çıkarım maliyeti düşük
- Coder tabanlı — netlist, Verilog, TCL, SPICE deck hepsi kod benzeri
- Olgun fine-tuning ekosistemi (Axolotl, Unsloth, LLaMA-Factory)

### Ağustos 2026 itibarıyla model manzarası

| Model | Durum | Bizim için |
|-------|-------|-----------|
| Qwen3 / Qwen3-Coder | Eski nesil (2 nesil geride) | Kullanma |
| Qwen3.5 (9B / 27B / 122B-A10B / 397B-A17B) | Açık, Apache 2.0 | Prototip için dense varyant uygun |
| Qwen3.6-27B dense / 35B-A3B MoE | Açık, Apache 2.0, agentic-coding tuned | **Ana hat** |
| Qwen3.8-27B dense | Açık, Apache 2.0, 262K bağlam | Alternatif |
| Qwen3.8-2.4T-A95B | Açık ama özel lisans | Bütçe/donanım dışı |
| Qwen3.8-Flash-Next (125B-A6B) | 26 Ağu 2026, Qwen4 önizlemesi | Kullanma — FT araç desteği yok, 172 GiB |
| Qwen4 | Henüz çıkmadı | Bekleme |

### Kural: base model bağlılığı yaratma
Veri hattı, eğitim formatı ve eval seti base model'den bağımsız yazılacak. Model değiştirmek birkaç günlük iş olmalı.

**Kalıcı varlıklar** (model değişince çöpe gitmez):
- SFT yörünge veri seti
- Eval seti
- Ödül fonksiyonu
- Araç arayüzü sözleşmesi
- nabla ve EDA araçları

**Değiştirilebilir:** base model ağırlıkları.

---

## 3. Seviye 2 — Model İskeleti Kararları

### 3.1 Tokenizer
**Karar:** Sınırlı genişletme, tam değişim yok.

**Problem:** Standart BPE sayıları kötü parçalar. `4.2u` üç token, `180n` bambaşka, `1.8` ile `1.80` ayrı temsil. Devre tasarımı baştan sona sayı işi.

**Yapılacak:**
- SI ön eklerini tek token yap: `f p n u m k M G T`
- Sık geçen birim kombinasyonlarını ekle: `uA`, `pF`, `MHz`, `dB`, `V/V`, `deg`
- Sık geçen device ve model isimlerini ekle: `nfet_01v8`, `pfet_01v8`, `sky130_fd_pr_*`
- Netlist anahtar kelimelerini ekle: `.subckt`, `.tran`, `.ac`, `.measure`

**Yapılmayacak:** Rakam bazlı tokenizasyona tam geçiş — base modelin mevcut bilgisini bozar.

### 3.2 Bağlam uzunluğu
**Karar:** Base'in native bağlamıyla başla, eksiği retrieval ile kapat.

RoPE ölçekleme / YaRN uzatması sonraya bırakılıyor. Gerekçe: netlist + PDK kartı + simülasyon logu + köşe sonuçları hızla şişiyor, ama doğru çözüm bağlamı büyütmek değil, bağlama doğru şeyi koymak.

### 3.3 Adapter stratejisi
**Karar:** LoRA, rank 32, attention + MLP katmanları.
- Full fine-tune'a ancak LoRA'nın yetmediği ölçülerek kanıtlanınca geçilir
- MoE'de LoRA daha dikkatli ayar ister (expert dengesizliği riski)
- Prototip dense modelde, üretim MoE'de

### 3.4 Değerlendirilip ertelenen
Sayısal çıkış başlığı (numeric regression head): Token üretiminin yanında sürekli değer regresyonu yapan ek katman. Teorik olarak cazip, ama sayısal optimizatör katmanı bu ihtiyacı zaten karşılıyor. İlk sürümde yok.

---

## 4. Seviye 3 — Eğitim Aşamaları

### Aşama 1: Continued Pretraining (alan bilgisi)

**Amaç:** Modele devre tasarımı literatürünü ham metin olarak okutmak.

**Korpus kategorileri:**

| Kategori | İçerik |
|----------|--------|
| Ders kitapları | Razavi, Gray & Meyer, Sansen, Baker (analog); Weste & Harris, Rabaey (dijital) |
| Akademik yayınlar | JSSC, ISSCC, CICC, VLSI Symposium |
| Gerçek tasarımlar | Açık kaynak RTL, açık PDK örnek devreleri, analog IP kütüphaneleri |
| Araç dokümantasyonu | ngspice, Xyce, OpenROAD, KLayout, magic, OpenSTA |
| Kendi araçlarımız | nabla ve EDA araç seti dokümanları |

**Hacim:** 1–5 milyar token. Fazlası getiri getirmiyor.

**Tuzaklar:**
- **Katastrofik unutma:** Agresif eğitim kod yazma yeteneğini bozar. Düşük learning rate + veriye %10-20 genel kod karıştır.
- **Veri kalitesi:** Kötü taranmış PDF, bozuk formül, karışık tablo modeli zehirler. Temizlik bu aşamanın en zahmetli işi.

**Not:** Üç aşamanın en az kritik olanı. Atlanabilir — base modelde bir miktar alan bilgisi zaten var. Asıl fark Aşama 2 ve 3'te çıkıyor.

**Lisans uyarısı:** Her kaynağın lisansı korpus listesinde kayıtlı olmalı. Sonradan geri dönmek maliyetli.

---

### Aşama 2: SFT — Ajan Yörüngeleri (davranış)

**Bu projenin en ayırt edici kısmı budur.**

**Yörünge tanımı:** Bir tasarım probleminin baştan sona çözüm kaydı. Spec → düşünme → topoloji → netlist → araç çağrısı → sonuç → teşhis → düzeltme → ... → spec tuttu. Bu zincirin tamamı tek eğitim örneği.

Model üç şeyi aynı anda öğrenir:
1. Araçları doğru formatta çağırmak
2. Simülasyon çıktısını yorumlamak
3. **Hatadan sonra ne yapacağını** ← en değerlisi

**Veri kaynakları:**

| Kaynak | Yöntem | Hacim |
|--------|--------|-------|
| Distillation | Güçlü bir modele (Claude Opus vb.) simülatör geri beslemesiyle çözdür | Orta |
| Sentetik bozma | Çalışan devreyi programatik boz → onart | Sınırsız |
| İnsan oturumları | Kendi tasarım oturumlarını kaydet | Az ama en kaliteli |

**Sentetik bozma detayı** (en verimli kaynak):
```
bias akımını 3x kaydır
kompanzasyon kapasitörünü sil
W/L oranını bozacak şekilde ölçekle
kaskod bias'ını yanlış düğüme bağla
yük kapasitansını değiştir, kompanzasyonu güncelleme
mirror oranını boz
```
Sonra modele onarttır. Tam olarak öğretmek istediğimiz beceriyi hedefliyor: teşhis ve onarım.

**Rejection sampling:** Sadece başarılı yörüngeler eğitime girer. Başarısız denemeler yörüngenin içinde kalabilir — hatta kalmalı, model toparlanmayı öğrenecek — ama sonu spec tutmayan yörünge atılır.

**Hacim:** 20.000–50.000 yörünge. Her yörünge 10–20 adımlık zincir.

**En büyük tuzak — format tutarsızlığı:** Araç çağrı formatı her örnekte birebir aynı olmalı. Tek bir kaçak virgül, çıkarım zamanında modelin araç çağıramamasına yol açar. Tüm veri şema doğrulamasından geçmeli.

---

### Aşama 3: RL — Doğrulanabilir Ödül (beceri)

**Yöntem:** GRPO (Group Relative Policy Optimization)

Aynı probleme birkaç farklı çözüm ürettirip birbirleriyle kıyaslar. Ayrı value network gerektirmez, hafıza dostu, doğrulanabilir ödülle çok iyi çalışır.

**Ödül kaynağı:** simülatör. İnsan değil. Devre tasarımı ölçülebilir — kazanç 60 dB mi, faz payı 60° mi, akım 200 µA altında mı. Simülatör kesin cevap verir ve yalan söylemez.

> **Bu yüzden nabla taç mücevherdir, model değil.** Ödül sinyalini simülatör üretir ve modelin yetenek tavanını simülatör belirler.

**Ödül tasarımı kuralları:**

| Kural | Açıklama |
|-------|----------|
| Kısmi kredi ver | İkili ödül çok seyrek sinyal verir. Her spec'e uzaklığı logaritmik ölçekte ölç |
| Köşeleri dahil et | Sadece nominal ödül verirsen model köşelerde patlayan devreler öğrenir |
| Yakınsamama'yı cezalandır, sıfırlama | Sıfır ödül modeli muhafazakârlaştırır; riskli ama doğru topolojilerden kaçınmayı öğrenir |
| Fizibilite kısıtları koy | Alan, akım, cihaz boyutu sınırları ödülün içinde |

**Ödül hackleme riski:** Model simülatördeki zayıflığı bulup spec'i sahte yollarla tutturabilir. Fizibilite kısıtları bunun panzehiri.

---

## 5. Sistem Mimarisi (modelin etrafı)

### Katman 1 — Ajan döngüsü
Modelin çalışma ritmi: maksimum adım sayısı, hata tekrarı stratejisi, kullanıcıya sorma eşiği, checkpoint/geri alma.

### Katman 2 — Araç arayüzü
- Araçlar yapılandırılmış veri döndürür, insan cümlesi değil
- Model log parse etmez
- Hatalar makine okunabilir
- Her şey Python API'den çağrılabilir
- Deterministik ve seed'lenebilir

### Katman 3 — Sayısal optimizatör
En çok atlanan, en çok fark yaratan parça. Model topolojiyi seçer, optimizatör boyutlandırır.

> Bunu yapmayan projeler analogda duvara toslar. LLM'e transistör genişliği ürettirmeye çalışmak yanlış mimaridir.

### Katman 4 — Hafıza ve bilgi erişimi
PDK verisi, önceki tasarımlar, referans devreler. Model bunları ezberlemez, sorgular.

### Katman 5 — Doğrulama ve emniyet
Her tasarım köşelerden, Monte Carlo'dan ve tasarım kurallarından geçer.

---

## 6. Araç Arayüzü Sözleşmesi

```python
# Simülasyon
sim.dc(netlist, params)              → DCResult
sim.ac(netlist, params)              → ACResult
sim.tran(netlist, params)            → TranResult
sim.noise(netlist, params)           → NoiseResult
sim.stb(netlist, params)             → StabilityResult
sim.corners(netlist, pvt_list)       → CornerResult[]
sim.mc(netlist, n, seed)             → MonteCarloResult

# Ölçüm
meas.eval(signal, expr)              → typed value

# Spec ve ödül
spec.check(results, spec)            → {score: float, breakdown: {...}}

# PDK sorgulama
pdk.device_query(model, W, L, VGS, VDS) → {gm, gds, ID, ft, Cgs, Cgd, ...}

# Netlist düzenleme
netlist.patch(diff)                  → atomik düzenleme
lint.check(netlist)                  → simülasyona girmeden yapısal hata

# RL ortamı
env.reset()                          → initial state
env.step(action)                     → (obs, reward, done, info)
```

---

## 7. Analog / Dijital Ayrımı

| | Analog | Dijital |
|---|--------|---------|
| Problem tipi | Sürekli sayısal optimizasyon | Kod üretimi |
| LLM'in gücü | Zayıf (sayısal aramada kötü) | Güçlü (zaten dil işi) |
| Doğrulama | Simülasyon + köşeler + MC | Testbench + formal + coverage |
| Mimari | Hibrit: LLM + optimizatör | Saf LLM yeterli |

---

## 8. İnşa Sırası

| Adım | İş | Durum |
|------|-----|-------|
| 0 | Bu doküman | ✅ |
| 1 | Araç arayüzü şeması (JSON schema) | ✅ |
| 2 | Eval seti — 50–200 görev | 🔄 9 görev, genişletiliyor |
| 3 | Korpus listesi + lisans denetimi | ✅ Framework |
| 4 | Baseline ölçümü | 🔲 |
| 5 | Adapter katmanı | ✅ |
| 6 | Ajan döngüsü + RL env wrapper | ✅ |
| 7 | Sentetik bozma pipeline'ı | 🔄 Güçlendiriliyor |
| 8 | SFT veri üretimi | 🔲 |
| 9 | Eğitim: CPT → SFT → RL | 🔲 |
| 10 | Sayısal optimizatör entegrasyonu | ✅ |

---

## 9. Tuzaklar (kontrol listesi)

- [ ] Format tutarsızlığı — araç çağrı formatı her örnekte birebir aynı
- [ ] Katastrofik unutma — CPT'de düşük LR + genel kod karışımı
- [ ] Ödül hackleme — fizibilite kısıtları ödülün içinde
- [ ] Köşesiz eğitim — nominal-only ödül köşede patlayan devre üretir
- [ ] Seyrek ödül — kısmi kredi, logaritmik uzaklık
- [ ] Yavaş simülatör — RL'i imkânsız kılar
- [ ] Eval setsiz ilerleme — ilerleme ölçülemez
- [ ] NDA'lı PDK ile eğitim — modeli dağıtılamaz kılar
- [ ] Ticari araç isimleri — marka riski + halüsinasyon
- [ ] Base model bağlılığı — kod model-agnostik olmalı
- [ ] LLM'e sayı ürettirmek — analogda duvar; optimizatör katmanı şart
- [ ] Lisans denetimini sonraya bırakmak — geri dönmek maliyetli

---

## 10. Ajana Notlar

Bu doküman tek referanstır. Antigravity ve Claude Code arasında geçiş yapılacaksa, her ikisi de işe bu dokümandan başlamalı.

- Görevleri parça ortasında değil, sınırından böl
- Adım 1 ve 2 (arayüz şeması, eval seti) ajan işi değildir
- Eğitim döngüsünü sıfırdan yazma — mevcut çatıları konfigüre et
- Kod model-agnostik olmalı: base model bir config değişkeni olsun
