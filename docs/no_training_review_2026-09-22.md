# Eğitim yapmadan genel denetim — 22 Eylül 2026

## Sonuç

**Yalnızca az iterasyon açıklaması yeterli değil. Kontrol, reset ve değerlendirme
tarafında doğrulanmış sorunlar var.** Önceki “kök neden kesin curriculum; ağ ve
uygulama sorunsuz” değerlendirmesi fazla kesindi. Şekil/NaN testlerinden geçmek,
kontrol tasarımının ve ölçümün doğru olduğunu kanıtlamıyor.

Bu incelemede **sıfır eğitim güncellemesi** yapıldı. Tek Isaac robotu, sensöre
gerçek temas yüklemek için 30 sıfır-aksiyon adımı (1,5 simülasyon saniyesi)
çalıştırıldı. Diğer kontroller kaynak okuma, checkpoint okuma, kinematik ve
aksiyon dönüşümü hesaplarıdır. Checkpoint ve eğitim ortamı değiştirilmedi.
CPU kümesi `<HOST_CPUSET>`, nice seviyesi 15 korundu.

Ölçüm dosyaları: `reports/no_training_audit.json` ve
`reports/no_training_reference_geometry.json`.

## Öncelikli bulgular

### 1. Ayakta kontrol yetkisi kayboluyor — doğrulandı, başarısızlığa katkısı henüz ayrıştırılmadı

`relative_action.py` bütün aksiyonları yüksekliğe ve dikliğe bağlı olarak
azaltıyor. Tam dik gövdede aynı `a=1` komutu şu pozisyon artışlarına dönüşüyor:

| Pelvis yüksekliği | Limit kırpması öncesi eklem artışı |
|---|---:|
| 0,50 m | 0,19040 rad |
| 0,58 m | 0,13540 rad |
| 0,60 m | 0,04892 rad |
| 0,62 m | 0 rad |
| 0,68 m | 0 rad |

Hedef her politika kararında `q_measured + delta` olduğu için sıfır komut,
önceden sabitlenmiş bir duruş hedefine geri dönmek anlamına gelmiyor. Tam dik
üst bölgede politika ayak bileği/kalça düzeltmesi de üretemiyor. Bu özellikle
“kalkıp yeniden düşme” için güçlü bir aday; sırtüstü durumdaki yükselmeme
problemini tek başına açıklamaz, çünkü o yükseklikte fren kapalıdır.

Ayrıca fren yokken bile pozisyon hatası karar anında en fazla 0,25 rad.
`Kp=120` bacaklarda bunun nominal oransal tork karşılığı 30 Nm; ithal diz/kalça
efor limiti 120 Nm. Ayak bileğinde `80 × 0,25 = 20 Nm`, kolda `40 × 0,25 = 10 Nm`.
**Bunlar simülasyon boyunca geçerli kesin tork üst sınırları değildir:** sönüm,
alt adımlardaki hareket ve implicit solver sonucu etkiler. Yine de “URDF tork
limitleri doğru, dolayısıyla yeterli kontrol kuvveti var” çıkarımı geçersizdir.
Statik yük taşıma ve gerçek hedef/tork takibi ayrıca ölçülmelidir.

### 2. Temas geçmişi bölüm resetinde temizlenmiyor — canlı testte doğrulandı

Sensör sıfırlanmadan önce filtreli temas geçmişinin tepe değeri **170,546 N**.
`sensor.reset([0])` sonrasında güncel kuvvet sıfır, geçmiş ise **aynı 170,546 N**
ve tüm tampon bit düzeyinde aynı. Kurulu PhysX `reset_contact_sensor_kernel`
güncel `force_matrix_w` tamponunu temizliyor; `force_matrix_w_history` tamponunu
almıyor. Yerel sensör bu reset davranışını miras alıyor.

Bu, bölüm sınırlarında eski yer temasını yeni ölçüme taşıyabilir. RMA'nın mevcut
proprioseptif girdisinde temas bitleri yok; doğrudan o girdiyi kirlettiğini
söylemiyoruz. Temas ödüllerini ve değerlendirme ölçümlerini etkileyebilir.
Temas maskeleri ayrıca iki geçmiş örneğin maksimumunu kullanıyor; bu filtre
iki ayağın tam aynı anda yük taşımasını kanıtlamaz. Başarı için güncel kuvveti,
geçmişi ise yalnızca açıkça tanımlanmış filtreleme için kullanmak daha temizdir.

### 3. Referans dizisi gerçek sırtüstü dağılımına bağlanmıyor — doğrulandı

`reset_root_state_recovery_curriculum` bütün referanslara +90° pitch uyguluyor;
bu, varsayılan sırtüstü −90° dönüşümü iptal ediyor. **0–8 arası bütün referanslar
dik gövdeli eklem pozlarıdır.** En alçak pelvis yüksekliği 0,09194 m olsa bile
bu bir sırtüstü poz değildir.

Resmi URDF'nin bütün çarpışma geometrileriyle nominal referansların zemine
girmediği kontrol edildi: en düşük açıklık yaklaşık 2 mm (ayakta 5,05 mm).
Bu test öz-çarpışmayı, dengeli COM yerleşimini veya pozlar arası dinamik
erişilebilirliği kanıtlamaz. Önceki “dokuz poz sırtüstüne kadar kapsıyor”
anlatımı bu nedenle yanıltıcıydı.

Seçilen modelin son kayıtlı evresi yalnızca `reference_2` örnekliyor. Sonraki
stage-3 denemeleri de gerçek sırtüstü reset içermiyor. Bu dağılımlarda daha çok
güncelleme yapmak, asıl başlangıcı öğrenmekle aynı şey değildir.

### 4. Tanı ve imitation toplama araçları bölüm sınırlarını kaybediyor — doğrulandı

`diagnose_isaac_policy.py` ve `collect_imitation_dataset.py`, `env.step`'in done
değerini atıyor. Isaac otomatik reset yapınca aynı diziye yeni bölüm ekleniyor.
Seçilen reference-2 raporunda yaklaşık 1,40 / 2,95 / 4,55 / 6,05 / 7,55 / 9,10
saniyelerde resetle tutarlı yükseklik sıçramaları ve sıfır kök hızları var.
İlk strict örnek 2,10 saniyede, yani ilk resetten sonra. **0,25 saniyelik
frontier, tek bir kesintisiz 10 saniyelik kalkışın sonucu olarak sunulmamalı.**

Beş bölümlük `evaluate_isaac.py` done kontrolü yapıyor; onun 0/5 sonucu ayrı.
Ancak 199 adım/snapshot sırası son örneği dışarıda bırakıyor ve güvenlik
termination'ında kaydettiği “terminal” state gerçekte bir önceki adım. ROS
sunucusu da terminated/truncated bilgisini atıyor; uzun timeout'ta fark
etmeden birden fazla otomatik bölümü tek servis denemesinde yürütebilir.

BC verisi bu bölüm sınırlarıyla tekrar denetlenmeli. BC train/validation ayrımı
trajektori yerine rastgele satırlar üzerinden yapılıyor; komşu, yüksek
korelasyonlu örnekler iki kümeye dağılabiliyor. Küçük validation MSE bu haliyle
yeni bölüm genellemesini veya başarılı yerden kalkışı kanıtlamaz.

### 5. Sırtüstü gözlemde Euler tekilliği — sayısal olarak doğrulandı

Mevcut 98 değerli propriosepsiyon doğrudan roll/pitch içeriyor. Pitch −π/2
çevresinde ±0,001 rad fark için aynı dönüşüm yordamı roll'u sırasıyla π ve 0
veriyor. Beş gerçek reset örneğinde roll yaklaşık −2,802 ile +1,644 rad
arasında değişti; normalleştirilmiş gözlemler sonluydu, mutlak maksimum 4,519.

Bu bir NaN veya kanıtlanmış quaternion sırası hatası değil; yönelim
gösteriminin süreksizliği. Katkısını ayırmak için projected gravity gibi sürekli
bir temsil düşünülebilir. Boyut/anlam değişirse eski checkpoint'e yeni girdi
sessizce verilemez; ayrı sürümlü model gerekir. Geçmiş gözlem tamponunun reset
sonrası güncel gözlemle dolduğu canlı testte doğrulandı.

### 6. Checkpoint–konfigürasyon izlenebilirliği eksik — doğrulandı

Model750'nin kaynak `params/env.yaml` dosyasında post hız ödül ağırlıkları
10/10, exact stance 100. Güncel ortam/README 50/50 ve 200 kullanıyor. Kaynak
YAML aksiyon fren alanlarını içermiyor; varsayılanları sonradan değişmiş kodla
eski checkpoint yükleniyor. Bu, frenin eğitimin her anında kapalı olduğunu
kanıtlamaz; geçmiş çalışma kodu olmadan tam eşleşme kurulamadığını gösterir.

İnference sırasında ödül ağırlıkları aksiyonu değiştirmez, ama deneyi yeniden
üretme iddiasını etkiler. Aksiyon dönüşümü değişiklikleri ise davranışı doğrudan
değiştirir. Her checkpoint için kod commit'i, çözülmüş ortam/ajan ayarları ve
ön/son işlem sözleşmesi birlikte saklanmalıdır.

## Makalelerin gerçekten desteklediği yöntem

- **[HumanUP, RSS 2025](https://www.roboticsproceedings.org/rss21/p063.html):**
  önce hareket keşfi, ardından bulunan hareketi yavaşlatıp takip ettirme.
  Yerel PDF bölüm C yaklaşık 5 milyar simülasyon örneği bildiriyor; bölüm IV
  1000 Hz simülasyon ve 50 Hz kontrol belirtiyor. Biz 100/20 Hz kullanıyoruz.
  Bu nedenle ödül isimlerini ve ağ genişliklerini almak tam yöntem tekrarı
  değildir. Daha düşük fizik frekansının X2 için yeterli olup olmadığı kısa
  adım-boyu yakınsama testiyle ölçülmeli; hemen 1000 Hz'e çıkmak da kanıtsız olur.
- **[HoST, RSS 2025](https://www.roboticsproceedings.org/rss21/p064.html):**
  çoklu critic, performansa bağlı yardım/aksiyon ölçeği curriculum'u ve ayrı
  post-task ödülleri birlikte kullanılıyor. İncelenen resmi G1 kodunda ölçek
  1'den başlıyor, performans arttıkça 0,25'e kadar azalıyor. Biz baştan sabit
  0,25 ve ayrıca sıfıra kadar fren kullanıyoruz; bunlar eşdeğer değil.
  Resmi `_compute_torques` her fizik alt adımında `q_current + beta*a`
  hesaplıyor. Dolayısıyla önceki “alt adımda yeniden hesaplama mutlaka
  beş kez birikme hatasıdır” iddiası fazla geneldi: biri sürekli oransal tork
  ofseti, diğeri politika adımı boyunca sabit pozisyon hedefidir. İkisinin
  etkisi farklıdır ve seçilen kontrol semantiği açıkça test edilmelidir.
- **[Action Space Design, CoRL 2024 / PMLR 2025](https://proceedings.mlr.press/v270/esser25a.html):**
  MIT/Fraunhofer araştırması aksiyon temsilinin başarım ve öğrenmeyi göreve
  bağlı olarak değiştirdiğini gösteriyor. Burada “önce kontrol yetkisini ve
  adımlar arasındaki davranışı ölç” önceliğini destekliyor; bizim beta/gain
  değerlerimizin doğru olduğunu kanıtlamıyor.
- **[Reverse Curriculum, CoRL 2017](https://proceedings.mlr.press/v78/florensa17a.html):**
  zorluğu mevcut başarımına göre genişletmek için yöntemsel dayanak. Dokuz
  sabit pozun dinamik olarak birbirine bağlı olduğunu varsaymak için dayanak
  değil; çalışma doğrudan X2 kalkış doğrulaması yapmıyor.
- **[FRASA](https://arxiv.org/abs/2410.08655):** Sigmaban ve CrossQ üzerinde bir
  yöntem. Oradaki frekans/ödül tercihi, X2/PhysX/31 aksiyon kombinasyonunun
  otomatik doğrulaması sayılamaz.

## Az iterasyon olasılığı

Mevcut ayarlarda 400 güncelleme = `3000 × 24 × 400 = 28,8 milyon` politika
geçişi. 3000 yerine 4096 ortam yaklaşık 1,37 kat örnek üretir; bunu yapmak
kontrolün sıfırlanmasını veya eksik başlangıç dağılımını düzeltmez. Paper'ın
5 milyarlık bütçesi daha uzun öğrenmenin gerekebileceğine dayanak; X2 için
mutlaka aynı bütçenin gerektiğini veya bunun başarı garantisi olduğunu söylemez.
Farklı frekans ve örnek sayımı tanımları da doğrudan duvar saati kıyasını sınırlar.

Stage-3 extension'da 21,6 milyon ek geçişle tepe yüksekliği 0,4210'dan 0,4145 m'ye
düştü; exact stance ödülü sıfır kaldı. Bu, mevcut ayarlarda ilerleme olmadığını
gösterir. “Daha çok eğitim hiçbir zaman işe yaramaz” veya “başka uygulama
hatası yok” sonucunu vermez.

## Eğitimden önce çözüm sırası

1. **Ölçümü düzelt:** yerel sensörde geçmiş reseti, güncel temasla strict kontrol,
   done'da dizi kesme, gerçek terminal state, trajektori bazında BC ayırma.
   Bu kontroller geçmeden yeni başarı oranı kullanma.
2. **Kontrolü doğrula:** sıfıra indiren freni kaldıran ayrı bir deney ayarı;
   seçilmiş tek aksiyon semantiğiyle hedef–ölçülen açı–gerçek tork kaydı.
   Limitler içinde ayakta yük taşıma ve küçük sapmaya tepki kontrolü.
   100/200/500 Hz fizik ve 20/50 Hz karar oranını aynı kısa hareketle karşılaştır.
   Sabit poza kısa süre dayanmak, aktif denge veya dinamik squat doğrulaması değildir.
3. **Tek bir yöntem seç ve sürümle:** bu aşamada yeni autoencoder/diffusion
   eklemek için kanıt yok. HoST'a yakın yardım/aksiyon curriculum'u veya
   HumanUP keşif→takip hattından birini kontrollü temel deney yap; farklı
   makalelerden giderek daha çok yüksek ağırlıklı ödül ekleme.
4. **Başlangıçları fiziksel bağla:** gerçek sırtüstü pozdan doğrulanmış temas
   geçişleri; başarıya göre küçük adımlar ve eski başarılı başlangıçları koruyan
   karışım. Referansların kök yönelimi, hızı ve temasları da gerekli.
5. **Ancak sonra kısa eğitim ablation'u:** aynı başlangıç/seed seti ve sabit
   başarı ölçütü; yükseklik, ayakta kalma, tork doygunluğu, termination nedeni,
   keşif std'si ve bölüm bazında sonuç. İlerleme varsa bütçeyi artır.

**Imitation zorunlu olduğu kanıtlanmış değil.** Başarılı bir öğretmen hareketi
varsa faydalı; mevcut kısmi/yeniden resetlenmiş kayıtları kopyalamak başarısızlığı
da kopyalayabilir. Kullanıcı talebi gereği bu planla yeni eğitim başlatılmadı.
