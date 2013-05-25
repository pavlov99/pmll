ENV=$(CURDIR)/.env
PYTHON=$(ENV)/bin/python
PYVERSION=$(shell pyversions --default)
SITE_PACKAGES=numpy scipy

RED=\033[0;31m
GREEN=\033[0;32m
NC=\033[0m

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
	@rm -f *.prof
	@rm -f *.lprof
	@rm -f *.so
	@rm -f */*.py[co]
	@rm -f */*.orig
	@rm -f */*/*.py[co]

.PHONY: register
# target: register - Register module on PyPi
register:
	@python setup.py register

.PHONY: upload
# target: upload - Upload module on PyPi
upload:
	@python setup.py sdist upload || echo 'Upload already'

.PHONY: test
# target: test - Runs tests
test: clean
	@nose2

init_virtualenv: requirements.txt
	virtualenv --no-site-packages .env

.PHONY: site-packages
site-packages:
	for p in $(SITE_PACKAGES); do \
	    pp=/usr/lib/$(PYVERSION)/dist-packages/$$p; \
	    if test -d $$pp; then \
	    echo "$(GREEN)Package "$$p" exists in system$(NC)" ;\
	    pplocal=$(ENV)/lib/$(PYVERSION)/site-packages/$$p ;\
	    if test -d $$pplocal; then \
		echo "Package "$$p" already exists in virtualenv: "$$pplocal ;\
	    else \
		ln -s $$pp $$pplocal ;\
		echo "$(GREEN)Package "$$p" successfully imported to virtualenv: "$$pplocal"$(NC)" ;\
	    fi; \
	    else \
	    echo "$(RED)Package "$$p" does not exists in system$(NC)" ;\
	    fi; \
	done

$(ENV): init_virtualenv site-packages
	$(ENV)/bin/pip install -r requirements.txt --use-mirrors
	touch $(ENV)
