/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package functions

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless"
	"github.com/spf13/cobra"
)

const FunctionsDir string = serverless.ServerlessDir + "/packages/functions"

// FunctionsCmd represents the serverless functions command
var FunctionsCmd = &cobra.Command{
	Use:   "functions",
	Short: "All serverless functions related subcommands",
	Long:  `Contains all serverless /serverless/functions directory related subcommands`,
	Args:  cobra.ExactArgs(0),
}

func init() {
	serverless.ServerlessCmd.AddCommand(FunctionsCmd)
}
