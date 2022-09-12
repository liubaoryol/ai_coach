Currently, gunicorn (version 20.1.0) is not compatible with newer version of eventlet. Please use eventlet version 0.30.2.

## Generate Zip File for the Deployment of AI Coach Web Experiment
```bash create_web_experiment_deployment.sh```
Copy the created zip file to the server:
```
scp aws_web_app.zip ACCOUNT@ADDRESS:aws_web_app.zip
```

## On Server PC (AWS EC2)
Ref: https://medium.com/techfront/step-by-step-visual-guide-on-deploying-a-flask-application-on-aws-ec2-8e3e8b82c4f7
### Install `AI Coach`
At your home folder,
```
unzip aws_web_app.zip -d ai_coach/
```

### Automatically run `AI Coach` whenever your server is restarted
Create a service file:
```sudo nano /etc/systemd/system/ai_coach.service```

Add the following to the file:
```
[Unit]
Description=Gunicorn instance for an AI Coach app
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/ai_coach
ExecStart=/home/ubuntu/ai_coach/venv/bin/gunicorn --worker-class eventlet -w 1 run:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable the service:
```
sudo systemctl daemon-reload
sudo systemctl start ai_coach.service
sudo systemctl enable ai_coach.service
```
Check if the app is running with: `curl localhost:8000`

### Run Nginx Webserver
* Install Nginx: `sudo apt-get install nginx`
* Start Nginx service:
  ```
  sudo systemctl start nginx
  sudo systemctl enable nginx
  ```
* Go to the Public IP address of your EC2 on the browser to see the default Nginx landing page

### Edit Nginx Default File
Open up the default file:
```sudo nano /etc/nginx/sites-available/default```

Add `upstream` and `location` to the file as follows:
```
upstream flaskaicoach {
        server 127.0.0.1:8000;
}

...

server {
        ...

        location / {
                # First attempt to serve request as file, then
                # as directory, then fall back to displaying a 404.
                proxy_pass http://flaskaicoach;
        }

        ...
}
```

### Restart the services
```
sudo systemctl restart ai_coach.service
sudo systemctl restart nginx
```
