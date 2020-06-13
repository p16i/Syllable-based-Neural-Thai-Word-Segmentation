
export PYTHONPATH=.

.PHONY: main-table

ARCHS=BiLSTM BiLSTM-CRF ID-CNN ID-CNN-CRF


all: main-table appendix-table syllable-table

main-table: $(ARCHS)

$(ARCHS): ./hyperopt-results.yml
	echo $@ && python ./scripts/writing/hyperopt-results-table.py $@

appendix-table: ./hyperopt-results.yml
	python ./scripts/writing/hyperopt-model-tables.py

syllable-table: ./writing/syllable-segmentation-eval-results
	python ./scripts/writing/syllable-table.py

clean:
	rm -f ./writing/tables/*