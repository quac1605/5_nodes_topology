#!/bin/bash

CONTAINER_NAME="clab-century-serf1"
FILE_NAME="coordinate_2D.exe"
TARGET_PATH="/tmp/$FILE_NAME"

echo "Copying $FILE_NAME to Docker container $CONTAINER_NAME:$TARGET_PATH ..."
docker cp "$FILE_NAME" "$CONTAINER_NAME":"$TARGET_PATH"

if [ $? -eq 0 ]; then
    echo "✅ File copied successfully to $CONTAINER_NAME!"
else
    echo "❌ Failed to copy the file. Make sure the container is running and the file exists in this folder."
fi
