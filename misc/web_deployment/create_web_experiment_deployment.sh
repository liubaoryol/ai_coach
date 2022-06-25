#!/bin/bash

ln -s ../../core/ai_coach_core ai_coach_core
ln -s ../../domains/ai_coach_domain ai_coach_domain
ln -s ../../web_app/web_experiment web_experiment
ln -s ../../web_app/config.py config.py
ln -s ../../web_app/run.py run.py
zip -r aws_web_app.zip ai_coach_core ai_coach_domain web_experiment run.py config.py requirements.txt Procfile
rm ai_coach_core ai_coach_domain web_experiment config.py run.py

# unzip command: unzip -d directory_name/ aws_web_app.zip
