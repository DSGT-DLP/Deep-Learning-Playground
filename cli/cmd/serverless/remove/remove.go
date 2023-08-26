/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package remove

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// RemoveCmd represents the serverless remove command
var RemoveCmd = &cobra.Command{
	Use:   "remove {package}",
	Short: "Remove package from pyproject.toml",
	Long:  `Remove package from pyproject.toml from /serverless`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		bash_args := []string{"remove", args[0]}
		if cmd.Flag("dev").Value.String() == "true" {
			bash_args = append(bash_args, "--dev")
		}
		pkg.ExecBashCmd(serverless.ServerlessDir, "yarn", bash_args...)
	},
}

func init() {
	serverless.ServerlessCmd.AddCommand(RemoveCmd)
	RemoveCmd.Flags().BoolP("dev", "d", false, "Remove package from dev dependencies")
}
