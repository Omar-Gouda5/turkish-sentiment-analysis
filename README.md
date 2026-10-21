# DistilBERT ile Türkçe Duygu Analizi

Hitit Üniversitesi Doğal Dil İşleme dersi için hazırlanmış Türkçe duygu analizi projesi. **Negative**, **Notr**, **Positive** olmak üzere üç sınıf vardır.

## Dosyalar
- [Google Colab notebook](Turkish_Sentiment_Analysis.ipynb) — veri hazırlama, modelin ince ayarı, değerlendirme ve örnek tahminler
- [Türkçe proje raporu](Turkce_Duygu_Analizi_Proje_Raporu.md) — öğrenci numarası çıkarılmış, GitHub'da doğrudan okunabilen sürüm
- [Eğitim kaybı grafiği](results/training_loss.svg) — rapordaki yaklaşık/yuvarlatılmış değerlerden yeniden çizilmiş görselleştirme
- [Karışıklık matrisi](results/confusion_matrix.svg) — orijinal test sayılarından yeniden çizilmiş görselleştirme
- [Deney özeti](results/experiment_summary.md) — orijinal Colab deneyinden alınan ölçümler

## Veri ve eğitim
Kaynak: [winvoker/turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset).
Her sınıftan 2.000 örnek seçilmiştir: 4.800 eğitim, 1.200 doğrulama. Ayrı test bölümünden sınıf oranları korunarak 3.000 örnek alınmıştır.
Model: [multilingual DistilBERT](https://huggingface.co/distilbert/distilbert-base-multilingual-cased); 3 epoch, learning rate `2e-5`, batch size `8`, maksimum uzunluk 128, seed 42. En iyi checkpoint doğrulama macro F1'e göre seçilmiştir.

## Sonuçlar
**Test Accuracy: %89,87** · **Macro F1: %85,05** · **Weighted F1: %90,51**.

Negatif sınıfın F1 skoru %65,89'dur. Üçüncü epochta doğrulama kaybının artması olası overfitting'e işaret eder.

## Çalıştırma
1. `Turkish_Sentiment_Analysis.ipynb` dosyasını Google Colab'de açın.
2. Mümkünse GPU seçin ve hücreleri sırayla çalıştırın.
3. Kurulum hücresindeki `transformers==4.46.3` sürümü, Colab ortamıyla uyumlu olduğu doğrulanan sürümdür. Paket kurulumundan sonra Colab yeniden başlatma isterse hücreleri tekrar sırayla çalıştırın.

**Not:** Eğitilmiş model ağırlıkları GitHub'a yüklenmemiştir. Notebook modeli yeniden eğitip `final_model/` klasörüne kaydeder. Raporun orijinal Word kopyası öğrenci numarası çıkarılmış olarak ayrıca saklanmıştır; GitHub'da görüntülenen sürüm Markdown biçimindedir.
