commit:
	git add .
	git commit -m "update"
	git push


install:
	pdm install


start:
	pdm run python train.py


test_model:
	pdm run python testcase/test_model.py






