#!/bin/bash
session="dl-playground"
tmux new-session -d -s $session
tmux rename-window -t 0 'frontend'
tmux send-keys -t 'frontend' 'yarn run startf' C-m
tmux new-window -t $session:1 -n 'backend'
tmux send-keys -t $session:1 'conda activate dlplayground && pythom -m backend.driver' C-m
tmux attach-session -t $session:0
