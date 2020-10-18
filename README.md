# Syllable based Neural Thai Word-Segmentation

By [Pattarawat Chormai](https://pat.chormai.org), [Ponrawee Prasertsom](https://ponraw.ee), [Jin Cheevaprawatdomrong](tbd), and [Attapol  T. Rutherford](https://attapol.github.io)

**Links**: [[Paper 📑]](paper) [[Presentation 📹]](presentation_en) [[การนำเสนอ 📹]](tbd) [[Citation  ⚓️️]](#Citation)

**Related Works**:
- [AttaCut](https://pythainlp.github.io/attacut/): this is the early version of this work.
- [CRF syllable segmenter for Thai (SSG)](https://github.com/ponrawee/ssg): our syllable segmenter.
- [DeepCut](https://github.com/rkcosmos/deepcut) via [Docker for Thai tokenizers](https://github.com/PyThaiNLP/docker-thai-tokenizers)


## Highlights
Some Results

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
|   |   |
</details>

## Citation
At the moment, we am waiting for the proceeding of COLING2020 to be available. Please stay tuned!.
```
TBD
```

[paper]: tbd
[presentation_en]: tbd
[presentation_th]: th