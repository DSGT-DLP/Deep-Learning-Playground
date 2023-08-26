/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package core

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless"
	"github.com/spf13/cobra"
)

const CoreDir string = serverless.ServerlessDir + "/packages/core"

// FunctionsCmd represents the serverless core command
var CoreCmd = &cobra.Command{
	Use:   "core",
	Short: "All serverless core related subcommands",
	Long:  `Contains all serverless /serverless/core directory related subcommands`,
	Args:  cobra.ExactArgs(0),
}

func init() {
	serverless.ServerlessCmd.AddCommand(CoreCmd)
}
