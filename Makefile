ENV=$(CURDIR)/.env

all: $(ENV)

.PHONY: help
# target: help - Display callable targets
help:
	@egrep "^# target:" [Mm]akefile

.PHONY: clean
# target: clean - Display callable targets
clean:
	@rm -rf build dist docs/_build
	@rm -f *.py[co]
	@rm -f *.orig
	@rm -f */*.py[co]
	@rm -f */*.orig

.PHONY: register
# target: register - Register module on PyPi
register:
	@python setup.py register

.PHONY: upload
# target: upload - Upload module on PyPi
upload: docs
	@python setup.py sdist upload || echo 'Upload already'

.PHONY: test
# target: test - Runs tests
test: clean
	@python setup.py test

$(ENV): requirements.txt
	# virtualenv --no-site-packages .env
	viryualenv --system-site-packages .env  # system numpy scipy
	$(ENV)/bin/pip install -M -r requirements.txt
	touch $(ENV)
