.PHONY: environment data clean lint

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = Squire_2022_correlation
ENV_NAME = squire_2022_correlation

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
ifeq (,$(shell $(CONDA_ACTIVATE) $(ENV_NAME) 2>&1))
HAS_ENV=True
else
HAS_ENV=False 
endif
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create the python environment or update it if it exists
environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (True,$(HAS_ENV)) # check if the directory is there
	@echo ">>> Project environment already exists, updating'"
	conda env update --name $(ENV_NAME) --file environment.yml --prune
else
	conda env create -f environment.yml
endif
else
 	@echo ">>> This project uses conda to install the environment. Please install conda"
endif
	$(CONDA_ACTIVATE) $(ENV_NAME) ; jupyter kernelspec uninstall -y $(ENV_NAME) ; python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))"

## Download/symlink raw data and prepare data from raw data sources
data:
	./data/raw/symlink_local_data.sh
	./data/raw/download_HadISST.sh
	./data/raw/download_HadSLP2.sh
	sed -i 's/NCI_PROJECT/$(PROJECT)/g' ./shell/prepare_data.sh
	qsub ./shell/prepare_data.sh

## Delete unneeded Python files, dask-worker files and PBS output files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.ipynb_checkpoints" -exec rm -rf {} +

## Lint using black and flake8
lint:
	($(CONDA_ACTIVATE) $(ENV_NAME) ; black src ; flake8 --exclude=.ipynb_checkpoints --ignore=E203,E266 src)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
#       * save line in hold space
#       * purge line
#       * Loop:
#               * append newline + line to hold space
#               * go to next line
#               * if line starts with doc comment, strip comment character off and loop
#       * remove target prerequisites
#       * append hold space (+ newline) to line
#       * replace newline plus comments by `---`
#       * print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
	        h; \
	        s/.*//; \
	        :doc" \
	        -e "H; \
	        n; \
	        s/^## //; \
	        t doc" \
	        -e "s/:.*//; \
	        G; \
	        s/\\n## /---/; \
	        s/\\n/ /g; \
	        p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
	        -v ncol=$$(tput cols) \
	        -v indent=19 \
	        -v col_on="$$(tput setaf 6)" \
	        -v col_off="$$(tput sgr0)" \
	'{ \
	        printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
	        n = split($$2, words, " "); \
	        line_length = ncol - indent; \
	        for (i = 1; i <= n; i++) { \
	                line_length -= length(words[i]) + 1; \
	                if (line_length <= 0) { \
	                        line_length = ncol - indent - length(words[i]) - 1; \
	                        printf "\n%*s ", -indent, " "; \
	                } \
	                printf "%s ", words[i]; \
	        } \
	        printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
