#!/bin/bash

branch_name=$1

# checks if branch name is provided
if [ -z "$branch_name" ]; then
  echo "Error: Branch name not provided."
  exit 1
fi

# Stash any local changes
git stash

# Switch to the 'nextjs' branch
git checkout nextjs

# Pull the latest changes from the remote 'nextjs' branch
git pull origin nextjs

# Create the new branch with the provided name
git checkout -b $branch_name

# Provide feedback to the user
echo "Branch '$branch_name' created successfully."
