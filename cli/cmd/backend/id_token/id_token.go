package id_token

/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend"
	"github.com/spf13/cobra"
)

// IdTokenCmd represents the IdToken command
var IdTokenCmd = &cobra.Command{
	Use:   "id-token",
	Short: "gets the id token",
	Long:  `gets the id token from the backend`,
	Args:  cobra.ExactArgs(0),
	Run:   func(cmd *cobra.Command, args []string) {},
}

func init() {
	backend.BackendCmd.AddCommand(IdTokenCmd)
}
