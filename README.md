# Syllable based Neural Thai Word-Segmentation

By [Pattarawat Chormai](https://pat.chormai.org), [Ponrawee Prasertsom](https://ponraw.ee), [Jin Cheevaprawatdomrong](tbd), and [Attapol  T. Rutherford](https://attapol.github.io)

**Links**: [[Paper 📑]][paper] [[Poster 🖼]][poster] [[Citation  ⚓️️]](#Citation)

**Related Works**:
- [AttaCut](https://pythainlp.github.io/attacut/): this is the early version of this work.
- [CRF syllable segmenter for Thai (SSG)](https://github.com/ponrawee/ssg): our syllable segmenter.
- [DeepCut](https://github.com/rkcosmos/deepcut) via [Docker for Thai tokenizers](https://github.com/PyThaiNLP/docker-thai-tokenizers)
- [Segmentation Speed Benchmark](https://github.com/heytitle/segmentation-speed-benchmark): our code for orchestrating AWS environment for segmentation speed benchmarking.

🚧 for running the code, please consult [DEV.md](DEV.md). The code was developed using Python `3.8.7`.

## Highlights

### Syllable and Word Segmentation Performance
<div align="center">
    <img src="https://i.imgur.com/oKj7w5a.png"/>
</div>

<div align="center">
    <img src="https://i.imgur.com/Y7hV50I.png"/>
</div>

<div align="center">
    <img src="https://i.imgur.com/LiDvDPg.png"/>
</div>

### Expected Validation Performance

<div align="center">
    <img src="https://i.imgur.com/3CbEGUW.png"/>
</div>

### Explaining Word Segmentation

<div align="center">
    <img src="https://i.imgur.com/eSxZfy4.png"/>
</div>


## Model Statistics

<details>
    <summary>Statistic Files</summary>

| Model  | Statistics File  |  
|---|---|
| BiLSTM(CH)-BI  |  [seq_ch_lstm_bi.yaml-2020-06-04--09-17.20.csv](./stats/seq_ch_lstm_bi.yaml-2020-06-04--09-17.20.csv)  |
| BiLSTM(CH-SY)-BI  | [seq_sy_ch_lstm_bi.yaml-2020-06-03--20-26.20.csv](./stats/seq_sy_ch_lstm_bi.yaml-2020-06-03--20-26.20.csv)  |
| BiLSTM(SY)-SchemeBI  | [seq_sy_lstm_bi.yaml-2020-06-03--23-35.20.csv](./stats/seq_sy_lstm_bi.yaml-2020-06-03--23-35.20.csv)  |
| BiLSTM(SY)-SchemeA  | [seq_sy_lstm_scheme_a.yaml-2020-06-03--23-35.20.csv](./stats/seq_sy_lstm_scheme_a.yaml-2020-06-03--23-35.20.csv)  |
| BiLSTM(SY)-SchemeB |  [seq_sy_lstm_scheme_b.yaml-2020-06-03--23-35.20.csv](./stats/seq_sy_lstm_scheme_b.yaml-2020-06-03--23-35.20.csv) |
| BiLSTM-CRF(SY)-BI  | [seq_sy_lstm_bi_crf.yaml-2020-06-03--18-10.20.csv](./stats/seq_sy_lstm_bi_crf.yaml-2020-06-03--18-10.20.csv)  |
| BiLSTM-CRF(SY)-SchemeA  | [seq_sy_lstm_crf_scheme_a.yaml-2020-06-03--23-34.20.csv](./stats/seq_sy_lstm_crf_scheme_a.yaml-2020-06-03--23-34.20.csv)  |
| BiLSTM-CRF(SY)-SchemeB  | [seq_sy_lstm_crf_scheme_b.yaml-2020-06-03--23-35.20.csv](./stats/seq_sy_lstm_crf_scheme_b.yaml-2020-06-03--23-35.20.csv)  |
| ID-CNN(CH)-BI  | [seq_ch_conv_3lv.yaml-2020-06-03--12-11.20.csv](./stats/seq_ch_conv_3lv.yaml-2020-06-03--12-11.20.csv)  |
| ID-CNN(CH-SY)-BI  | [seq_sy_ch_conv_3lv.yaml-2020-06-02--23-23.20.csv](./stats/seq_sy_ch_conv_3lv.yaml-2020-06-02--23-23.20.csv)  |
| ID-CNN(SY)-BI  | [seq_sy_conv_3lv.yaml-2020-06-02--08-19.20.csv](./stats/seq_sy_conv_3lv.yaml-2020-06-02--08-19.20.csv)  |
| ID-CNN(SY)-SchemeA  | [seq_sy_conv_3lv_scheme_a.yaml-2020-06-02--10-49.20.csv](./stats/seq_sy_conv_3lv_scheme_a.yaml-2020-06-02--10-49.20.csv)  |
| ID-CNN(SY)-SchemeB  | [seq_sy_conv_3lv_scheme_b.yaml-2020-06-02--10-49.20.csv](./stats/seq_sy_conv_3lv_scheme_b.yaml-2020-06-02--10-49.20.csv)  |
| ID-CNN-CRF(SY)-BI  | [seq_sy_conv_3lv_crf_bi.yaml-2020-06-01--11-40.20.csv](stats/seq_sy_conv_3lv_crf_bi.yaml-2020-06-01--11-40.20.csv)  |
| ID-CNN-CRF(SY)-SchemeA  | [seq_sy_conv_3lv_crf_scheme_a.yaml-2020-06-01--11-39.20.csv](./stats/seq_sy_conv_3lv_crf_scheme_a.yaml-2020-06-01--11-39.20.csv)  |
| ID-CNN-CRF(SY)-SchemeB  | [seq_sy_conv_3lv_crf_scheme_b.yaml-2020-06-01--11-39.20.csv](./stats/seq_sy_conv_3lv_crf_scheme_b.yaml-2020-06-01--11-39.20.csv)  |

</details>


## Citation
```
@inproceedings{chormai-etal-2020-syllable,
    title = "Syllable-based Neural {T}hai Word Segmentation",
    author = "Chormai, Pattarawat  and
      Prasertsom, Ponrawee  and
      Cheevaprawatdomrong, Jin  and
      Rutherford, Attapol",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.407",
    pages = "4619--4637",
    abstract = "Word segmentation is a challenging pre-processing step for Thai Natural Language Processing due to the lack of explicit word boundaries.The previous systems rely on powerful neural network architecture alone and ignore linguistic substructures of Thai words. We utilize the linguistic observation that Thai strings can be segmented into syllables, which should narrow down the search space for the word boundaries and provide helpful features. Here, we propose a neural Thai Word Segmenter that uses syllable embeddings to capture linguistic constraints and uses dilated CNN filters to capture the environment of each character. Within this goal, we develop the first ML-based Thai orthographical syllable segmenter, which yields syllable embeddings to be used as features by the word segmenter. Our word segmentation system outperforms the previous state-of-the-art system in both speed and accuracy on both in-domain and out-domain datasets.",
}
```

[paper]: https://www.aclweb.org/anthology/2020.coling-main.407.pdf
[presentation_en]: tbd
[presentation_th]: th
[poster]: https://drive.google.com/file/d/1sV8xPgdj0tMaWhHcCs4vANYWBCygPTvO/view?usp=sharing
