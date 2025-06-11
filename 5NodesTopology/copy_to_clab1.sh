#!/bin/bash

FILES=("coordinate_ping_RTT_printcode" "checking_ping_rtt_with_cmd.sh" "RTT_code.exe" "coordinate_go_code" "code_go_jte")

# Loop through container names clab-century-serf1 to clab-century-serf5
for i in {1..26}; do
  CONTAINER_NAME="clab-century-serf$i"

  for FILE_NAME in "${FILES[@]}"; do
    TARGET_PATH="/opt/serfapp/$FILE_NAME"

    echo "Copying $FILE_NAME to Docker container $CONTAINER_NAME:$TARGET_PATH ..."
    docker cp "$FILE_NAME" "$CONTAINER_NAME":"$TARGET_PATH"

    if [ $? -eq 0 ]; then
      echo "✅ $FILE_NAME copied successfully to $CONTAINER_NAME!"
  else
      echo "❌ Failed to copy $FILE_NAME. Make sure the container is running and the file exists in this folder."
  fi
  done
done
