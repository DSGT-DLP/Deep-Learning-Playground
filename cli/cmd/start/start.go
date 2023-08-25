/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package start

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd"
	"github.com/spf13/cobra"
)

// startCmd represents the start command
var StartCmd = &cobra.Command{
	Use:   "start",
	Short: "Starts an instance in the terminal (e.g. frontend, backend, etc)",
	Long:  `Starts an instance of a DLP-created app in the terminal`,
	Args:  cobra.ExactArgs(0),
}

func init() {
	cmd.RootCmd.AddCommand(StartCmd)
}
