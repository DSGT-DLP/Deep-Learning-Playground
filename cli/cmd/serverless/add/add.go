/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package add

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// AddCmd represents the serverless add command
var AddCmd = &cobra.Command{
	Use:   "add {package}",
	Short: "Add package to pyproject.toml",
	Long:  `Add package to pyproject.toml from /serverless`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		bash_args := []string{"add", args[0]}
		if cmd.Flag("dev").Value.String() == "true" {
			bash_args = append(bash_args, "--dev")
		}
		pkg.ExecBashCmd(serverless.ServerlessDir, "yarn", bash_args...)
	},
}

func init() {
	serverless.ServerlessCmd.AddCommand(AddCmd)
	AddCmd.Flags().BoolP("dev", "d", false, "Add package as dev dependency")
}
