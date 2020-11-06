#!/bin/bash

session="ml-sandbox"

# New tmux session:
tmux new-session -d -s $session

# --- Window [1] --- #
tmux rename-window Dashboard

# Select Pane 1:
tmux selectp -t 1
tmux send-keys "cd ~/src/" C-m
tmux splitw -h -p 48

# Select Pane 2
tmux selectp -t 2
tmux send-keys "htop" C-m

tmux splitw -v -p 39
tmux send-keys "watch -n 1.0 nvidia-smi" C-m

tmux selectp -t 1

# Window [2]: Jupyter and TensorBoard
tmux new-window -t $session:1 -a -n Jupyter-TensorBoard
tmux selectp -t 1
tmux send-keys "cd ~" C-m

# Window [3]: Editing
tmux new-window -t $session:2 -a -n Editing
tmux selectp -t 1
tmux send-keys "cd ~/src/" C-m

tmux splitw -h -p  60
tmux splitw -v

tmux selectp -t 1

# --- Finish setup and return to Dashboard --- #
tmux select-window -t $session:Dashboard
tmux attach-session -t $session

# --- Done! ---
