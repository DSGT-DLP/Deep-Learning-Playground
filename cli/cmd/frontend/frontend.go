/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package frontend

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd"
	"github.com/spf13/cobra"
)

// FrontendCmd represents the frontend command
var FrontendCmd = &cobra.Command{
	Use:   "frontend",
	Short: "All frontend related subcommands",
	Long:  `Contains all frontend /frontend directory related subcommands`,
	Args:  cobra.ExactArgs(0),
}

func init() {
	cmd.RootCmd.AddCommand(FrontendCmd)
}
