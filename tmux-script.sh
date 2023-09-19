#!/bin/bash
session="dl-playground"
tmux new-session -d -s $session
tmux rename-window -t 0 'backend'
tmux send-keys -t 'backend' 'yarn run startt' C-m
tmux new-window -t $session:1 -n 'frontend'
tmux send-keys -t $session:1 'yarn run startf' C-m
tmux attach-session -t $session:0
