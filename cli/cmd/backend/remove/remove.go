/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package remove

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// AddCmd represents the backend add command
var RemoveCmd = &cobra.Command{
	Use:   "remove {package}",
	Short: "Remove package from pyproject.toml",
	Long:  `Remove package from pyproject.toml from /training`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		bash_args := []string{"remove", args[0]}
		if cmd.Flag("dev").Value.String() == "true" {
			bash_args = append(bash_args, "--group", "dev")
		}
		pkg.ExecBashCmd(backend.BackendDir, "poetry", bash_args...)
	},
}

func init() {
	backend.BackendCmd.AddCommand(RemoveCmd)
	RemoveCmd.Flags().BoolP("dev", "d", false, "Remove package from dev dependencies")
}
