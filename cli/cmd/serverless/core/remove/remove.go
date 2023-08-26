/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package remove

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/core"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// RemoveCmd represents the serverless core remove command
var RemoveCmd = &cobra.Command{
	Use:   "remove {package}",
	Short: "Remove package from package.json",
	Long:  `Remove package from package.json from /serverless/packages/core`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		bash_args := []string{"remove", args[0]}
		if cmd.Flag("dev").Value.String() == "true" {
			bash_args = append(bash_args, "--dev")
		}
		pkg.ExecBashCmd(core.CoreDir, "yarn", bash_args...)
	},
}

func init() {
	core.CoreCmd.AddCommand(RemoveCmd)
	RemoveCmd.Flags().BoolP("dev", "d", false, "Remove package from dev dependencies")
}
