# set the base image 
FROM python:3.6

#add project files to the usr/src/app folder
ADD . /usr/src/app

#set directoty where CMD will execute 
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app


# Get pip to download and install requirements:

RUN pip install -r /usr/src/app/requirements.txt

# Expose ports

EXPOSE 80

# default command to execute    


CMD python /usr/src/app/manage.py runserver 0.0.0.0:80
