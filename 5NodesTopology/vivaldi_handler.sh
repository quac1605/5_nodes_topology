#!/bin/bash

# Check if this is a query and matches "vivaldi_coordinates"
if [ "$SERF_QUERY_NAME" == "vivaldi_coordinates" ]; then
  # Read coordinates from environment variables or hardcoded values
  VIVALDI_X=${VIVALDI_X:-1.23}
  VIVALDI_Y=${VIVALDI_Y:-4.56}

  # Respond with JSON containing the coordinates
  echo "{\"vivaldi_x\": \"$VIVALDI_X\", \"vivaldi_y\": \"$VIVALDI_Y\"}"
fi
