## Web Interface for Human Experiment

To run web experiment, `ai_coach_core` and `ai_coach_domain` packages should be installed first.
```
cd web_app_v2/
pip install -e ../core
pip install -e ../domains
pip install -r "requirements.txt"
```

Then, run the following command:
```
python -m run
```

You can access the webpage through `http://localhost:5000/`. On the home page, enter `register1234` to create a new ID for the test.
