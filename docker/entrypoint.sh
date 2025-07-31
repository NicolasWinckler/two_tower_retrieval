#!/bin/bash

#tail -f /dev/null
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token= --NotebookApp.password= --notebook-dir=${WORK_DIR}
