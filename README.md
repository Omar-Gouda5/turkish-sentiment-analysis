# DistilBERT ile Türkçe Duygu Analizi

Hitit Üniversitesi Doğal Dil İşleme dersi için hazırlanan üç sınıflı Türkçe duygu analizi projesi. Model: `distilbert/distilbert-base-multilingual-cased`. Sınıflar: **Negative**, **Notr**, **Positive**.

## Çalıştırma
`Turkish_Sentiment_Analysis.ipynb` dosyasını Google Colab ile açın, varsa T4 GPU seçin ve hücreleri sırayla çalıştırın. Kullanılan paket sürümü: `transformers==4.46.3`.

## Veri
[winvoker/turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset). Eğitim bölümünden her sınıftan 2.000 örnek alınarak 4.800 eğitim ve 1.200 doğrulama örneği hazırlanmıştır. Bağımsız test bölümünden tabakalı olarak 3.000 örnek değerlendirilmiştir. Random seed: 42.

## Eğitim
3 epoch, learning rate 2e-5, batch size 8, max length 128, FP16 (GPU varsa), doğrulama macro F1'e göre en iyi checkpoint.

## Sonuçlar
Orijinal Colab çalıştırmasında **test accuracy %89,87** ve **macro F1 %85,05** elde edilmiştir. Negatif sınıfın F1 skoru %65,89'dur. Eğitim kaybı azalırken üçüncü epochta doğrulama kaybı artmış, olası overfitting gözlenmiştir. Ayrıntılar: `results/experiment_summary.md`.

## Dosyalar
- `Turkish_Sentiment_Analysis.ipynb`: eğitim, test, grafikler ve tahminler
- `results/experiment_summary.md`: orijinal test ölçümleri
- `Turkce_Duygu_Analizi_Proje_Raporu.docx`: öğrenci numarası çıkarılmış rapor (ayrı dosya olarak eklenecek)
- `results/training_loss.png`, `results/confusion_matrix.png`: orijinal görseller (ayrı dosyalar olarak eklenecek)

**Not:** Model ağırlıkları boyut nedeniyle GitHub'a yüklenmez. Notebook tekrar çalıştırılarak model yeniden üretilebilir. Rapor ve görseller yüklenene kadar depodaki metrik özeti referans alınmalıdır.

## Kaynaklar
- https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset
- https://huggingface.co/distilbert/distilbert-base-multilingual-cased
- Sanh et al. (2019), DistilBERT, arXiv:1910.01108.
