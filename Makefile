PYTHON ?= python3

.PHONY: setup export-scripts validate-scripts

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

export-scripts:
	$(PYTHON) scripts/export_notebooks.py --input notebooks --output scripts

validate-scripts:
	$(PYTHON) -m compileall scripts