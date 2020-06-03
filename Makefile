export pythonpath = .

tables: ./writing/tables/hyperopt-results-bilstm.tex ./writing/tables/hyperopt-results-ID-CNN.tex ./writing/tables/hyperopt-model-tables.tex

./writing/tables/hyperopt-results-bilstm.tex: ./hyperopt-results.yml
	PYTHONPATH=$(pythonpath) python ./scripts/writing/hyperopt-results-table.py BiLSTM

./writing/tables/hyperopt-results-ID-CNN.tex: ./hyperopt-results.yml
	PYTHONPATH=$(pythonpath) python ./scripts/writing/hyperopt-results-table.py ID-CNN

./writing/tables/hyperopt-model-tables.tex: ./hyperopt-results.yml
	PYTHONPATH=$(pythonpath) python ./scripts/writing/hyperopt-model-tables.py
