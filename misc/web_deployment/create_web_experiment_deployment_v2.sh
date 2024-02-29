#!/bin/bash

ln -s ../../core/aic_core aic_core
ln -s ../../domains/aic_domain aic_domain
ln -s ../../web_app_v2/web_experiment web_experiment
ln -s ../../web_app_v2/config.py config.py
ln -s ../../web_app_v2/run.py run.py
zip -r aws_web_app.zip aic_core aic_domain web_experiment run.py config.py requirements.txt Procfile
rm aic_core aic_domain web_experiment config.py run.py

# unzip command: unzip -d directory_name/ aws_web_app.zip
