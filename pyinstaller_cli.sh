pyinstaller main.py\
    --noconsole\
    --add-data ./videos:videos --add-data ./images:images --add-data ./known-faces:known-faces --add-data ./name-cards:name-cards\
    --icon="app-icon.png"\
    --name AR-app