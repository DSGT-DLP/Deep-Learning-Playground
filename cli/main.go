/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package main

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/start"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/start/backend"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/start/frontend"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/start/serverless"
)

func main() {
	cmd.Execute()
}
