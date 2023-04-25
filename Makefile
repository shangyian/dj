pyenv: .python-version

.python-version: setup.cfg
	if [ -z "`pyenv virtualenvs | grep dj`" ]; then\
	    pyenv virtualenv dj;\
	fi
	if [ ! -f .python-version ]; then\
	    pyenv local dj;\
	fi
	pip install -r requirements/test.txt
	touch .python-version

docker-build:
	docker build .
	docker compose build

docker-run:
	docker compose up

test:
	poetry run pytest --cov=dj --cov-report=html -vv tests/ --doctest-modules dj --without-integration --without-slow-integration ${PYTEST_ARGS}

integration:
	poetry run pytest --cov=dj -vv tests/ --doctest-modules dj --with-integration --with-slow-integration

clean:
	pyenv virtualenv-delete dj

spellcheck:
	codespell -L froms -S "*.json" dj docs/*rst tests templates

check:
	poetry run pre-commit run --all-files

version:
	@poetry version $(v)
	@git add pyproject.toml
	@git commit -m "v$$(poetry version -s)"
	@git tag v$$(poetry version -s)
	@git push
	@git push --tags
	@poetry version

release:
	@poetry publish --build
