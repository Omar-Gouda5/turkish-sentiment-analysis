# Gerçek Colab deneyi — sonuç özeti

- Test örneklem büyüklüğü: **3000**
- Accuracy: **0.8987**
- Macro F1: **0.8505**
- Weighted F1: **0.9051**

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.5553 | 0.8098 | 0.6589 | 347 |
| Notr | 0.9876 | 0.9914 | 0.9895 | 1047 |
| Positive | 0.9543 | 0.8574 | 0.9032 | 1606 |

Karışıklık matrisi (satırlar gerçek, sütunlar tahmin):

```text
             Negative  Notr  Positive
Negative          281     0        66
Notr                9  1038         0
Positive          216    13      1377
```

Örnekler:
- Bu ürünü çok beğendim, harika çalışıyor. → Positive (%98,61)
- Ürün çok kötü, hiç memnun kalmadım. → Negative (%99,09)
- Paket bugün teslim edildi. → Notr (%99,93)

Üçüncü epochta doğrulama kaybı artmaktadır; olası overfitting söz konusudur.
