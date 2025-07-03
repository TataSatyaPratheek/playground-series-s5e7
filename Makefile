install:
	python -m pip install -r requirements.txt

train:
	python -m src.train

submit:
	python -m src.predict && \
	kaggle competitions submit -c playground-series-s5e7 -f submission/submission.csv -m "CatBoost baseline"
