#!/bin/bash

SESSION="robots"
ROBOTS=(30 34 26 25 28 35 27 37 12 21)
#ROBOTS=(07 10 12 13 14 15 19 20)

# -----------------------------n
# Function 1: Connect to robots
# -----------------------------
connect_robots() {

  tmux kill-session -t $SESSION 2>/dev/null

  # Create session AND attach immediately
  tmux new-session -d -s $SESSION

  # Create remaining robot windows
  for ROBOT in "${ROBOTS[@]}"; do
    tmux new-window -t $SESSION -n $ROBOT \
      "sirius -w clem -r $ROBOT"

    echo "Next robot: ${ROBOT}"
  done

  tmux new-window -t $SESSION -n extra-functions \
      "sirius -w clem"

  tmux attach -t $SESSION
  
}

# -----------------------------------------
# Function 2: Run script on connected robots
# -----------------------------------------

# Declare associative array of exclusions
declare -A EXCLUDE
# Format: EXCLUDE["r1,r2"]=1 (r1 excludes r2)
# EXCLUDE["r21,r28"]=1
# EXCLUDE["r21,r27"]=1
# EXCLUDE["r25,r27"]=1
# EXCLUDE["r25,r26"]=1
# EXCLUDE["r24,r37"]=1
# EXCLUDE["r24,r28"]=1
# EXCLUDE["r26,r37"]=1
# EXCLUDE["r21,r28"]=1
# # Automatically symmetric:
# EXCLUDE["r3,r1"]=1
# EXCLUDE["r4,r2"]=1

run_on_robots() {

  for i in "${!ROBOTS[@]}"; do
    ROBOT="${ROBOTS[$i]}"
    # Ring topology: each robot's neighbour is the next in the list (wrapping around)
    NEXT_I=$(( (i + 1) % ${#ROBOTS[@]} ))
    NEIGHBOURS=("r${ROBOTS[$NEXT_I]}")

    # Full mesh with exclusions (previous behaviour):
    # NEIGHBOURS=()
    # for j in "${!ROBOTS[@]}"; do
    #   if [ "$i" -ne "$j" ]; then
    #     CANDIDATE="r${ROBOTS[$j]}"
    #     # Check if this pair is excluded
    #     if [[ -z "${EXCLUDE["r$ROBOT,$CANDIDATE"]}" && -z "${EXCLUDE["$CANDIDATE,r$ROBOT"]}" ]]; then
    #       NEIGHBOURS+=("$CANDIDATE")
    #     fi
    #   fi
    # done

    NEIGHBOUR_STRING=$(IFS=,; echo "${NEIGHBOURS[*]}")

    # Send commands to tmux window
    tmux send-keys -t "$SESSION:$ROBOT" "cd dots_templates_arm/" C-m
    tmux send-keys -t "$SESSION:$ROBOT" "colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release" C-m
    tmux send-keys -t "$SESSION:$ROBOT" "source install/setup.bash" C-m
    tmux send-keys -t "$SESSION:$ROBOT" "ros2 launch dots_example_controllers run_unicycles.launch.py name:=r$ROBOT neighbours_names:=$NEIGHBOUR_STRING" C-m

    sleep 0.5
  done

  NEIGHBOURS=()
  for r in "${ROBOTS[@]}"; do
      NEIGHBOURS+=("r$r")
  done
  NEIGHBOUR_STRING=$(IFS=,; echo "${NEIGHBOURS[*]}")

  tmux send-keys -t "$SESSION:extra-functions" "cd dots_templates_arm/" C-m
  tmux send-keys -t "$SESSION:extra-functions" "colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release" C-m
  tmux send-keys -t "$SESSION:extra-functions" "source install/setup.bash" C-m
  tmux send-keys -t "$SESSION:extra-functions" "ros2 launch dots_example_controllers run_unicycles_extras.launch.py neighbours_names:=$NEIGHBOUR_STRING" C-m

  echo "Launch script executed on all robots"
}

# -----------------------------------------
# Function 3: Kill all robots
# -----------------------------------------
kill_robots() {
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "No tmux session '$SESSION' found."
    return
  fi

    # Also handle the extra window you created in connect_robots
  echo "Stopping extra tmux window"
  tmux send-keys -t "$SESSION:extra-functions" C-c
  sleep 0.5

  for WIN in "${ROBOTS[@]}"; do
    echo "Stopping robot window: $WIN"
    # Send Ctrl+C to each tmux window
    tmux send-keys -t "$SESSION:$WIN" C-c
    sleep 0.5  # small pause to avoid flooding
  done

  echo "All robots signaled to stop."
}



# -----------------------------
# Command line interface
# -----------------------------
case "$1" in
  connect)
    connect_robots
    ;;
  run)
    run_on_robots
    ;;
  kill)
    kill_robots
    ;;
  restart)
    connect_robots
    sleep 5
    run_on_robots
    ;;
  attach)
    tmux attach -t $SESSION
    ;;
  *)
    echo "Usage: $0 {connect|run|restart|attach}"
    exit 1
    ;;
esac