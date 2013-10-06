SHELL=/bin/bash
# ENV=$(CURDIR)/.env  # for manual creation

PYTHON=$(shell which python)  # current virtualenv for different versions test
BIN=$(shell dirname $(PYTHON))
ENV=$(shell dirname $(BIN))
PYVERSION=$(shell $(PYTHON) -c "import sys; print('python{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
PYTHON_PACKAGE_PATH="/usr/lib/$(shell [[ $(PYVERSION) == python3* ]] && echo "python3" || echo $(PYVERSION))/dist-packages/"

SITE_PACKAGES=numpy scipy
SPHINXBUILD=sphinx-build

RED=$(shell tput setaf 1)
GREEN=$(shell tput setaf 2)
NC=$(shell tput setaf 7)


.PHONY: tt
tt:
	$(PYTHON) --version

all: $(ENV)

.PHONY: help
# target: help - Display callable targets
help:
	@egrep "^# target:" [Mm]akefile

.PHONY: clean
# target: clean - clean project
clean:
	@find . -name \*.py[co] -delete
	@find . -name *\__pycache__ -delete
	@rm -f nosetests.xml pep8.pylama pylint.pylama
	@echo 'Finish cleaning'

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
	$(PYTHON) setup.py test

init_virtualenv: requirements.txt
	virtualenv --no-site-packages .env
	source .env/bin/activate

.PHONY: site-packages
# target: site-packages - link system packages to virtual environment
site-packages:
	@for p in $(SITE_PACKAGES); do \
	    pp=$(PYTHON_PACKAGE_PATH)/$$p; \
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

env: init_virtualenv site-packages
	$(BIN)/pip install -r requirements.txt --use-mirrors
