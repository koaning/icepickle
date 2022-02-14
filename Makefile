black:
	black icepickle tests setup.py --check

flake:
	flake8 icepickle tests setup.py

test:
	pytest

interrogate:
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 tests
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 icepickle

check: black flake interrogate test

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	pre-commit install
	python -m pip install wheel twine

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*
