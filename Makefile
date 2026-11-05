.PHONY: test invariants check_env

define CHECK_RAG_ENV
	@if [ "$${CONDA_DEFAULT_ENV}" != "rag" ]; then \
		echo "ERROR: Conda environment 'rag' is not active (CONDA_DEFAULT_ENV='$${CONDA_DEFAULT_ENV}'). Please run: conda activate rag" >&2; \
		exit 1; \
	fi
endef

check_env:
	$(CHECK_RAG_ENV)
	python scripts/check_env.py

invariants:
	$(CHECK_RAG_ENV)
	python scripts/check_invariants.py

test:
	$(CHECK_RAG_ENV)
	pytest -v tests/test_invariants.py
