#!/bin/bash

git submodule update --init
python -m venv venv
pip install -r requirements.txt