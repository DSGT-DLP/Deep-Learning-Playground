/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package serverless

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd"
	"github.com/spf13/cobra"
)

// ServerlessCmd represents the serverless command
var ServerlessCmd = &cobra.Command{
	Use:   "serverless",
	Short: "All serverless related subcommands",
	Long:  `Contains all serverless /serverless directory related subcommands`,
	Args:  cobra.ExactArgs(0),
}

func init() {
	cmd.RootCmd.AddCommand(ServerlessCmd)
}
