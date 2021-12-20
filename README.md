# Me_Bot
A simple tool to make a bot that speaks like you, simply learning from your WhatsApp Chats.

Instructions:-

1. From WhatsApp on your phone, go to any chat and export it by going into the settings. Move the txt file that you receive inside the Me_Bot folder.

2. Run the clean_whatsapp_chats.py script using the command. Before running, change the names of the people by changing YOUR_NAME and OTHER_NAME in the scripts according to the txt file you have for your chats.

`python clean_whatsapp_chats.py whatsapp_chat.txt`

3. Run the prepare_files.py script to create embeddings from cleaned chat.

4. Run the predict.py file and and edit to play with the bot at the bottom!

To use django app for a rest api experience:

Note: Python 3.6 and Linux based machine are core requirements.

1. Create a virtual environment and use:

`pip install -r requirements.txt`

2. Run a django server using:

`python manage.py runserver`

Note: A pretrained chat model has been included for testing. Use your model after cleaning and preparing files (instructions stated above) using the same directory structure in "res" folder.

