/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package install

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// InstallCmd represents the backend install command
var InstallCmd = &cobra.Command{
	Use:   "install",
	Short: "Starts the training backend",
	Long:  `Starts an instance of the training backend Django app in /training in the terminal`,
	Args:  cobra.ExactArgs(0),
	Run: func(cmd *cobra.Command, args []string) {
		if cmd.Flag("force").Value.String() == "true" {
			pkg.ExecBashCmd(backend.BackendDir, "poetry", "env", "remove", "--all")
		}
		pkg.ExecBashCmd(backend.BackendDir, "pyenv", "local", "3.9")
		pkg.ExecBashCmd(backend.BackendDir, "poetry", "install")
		pkg.ExecBashCmd(backend.BackendDir, "poetry", "run", "ggshield", "auth", "login")
		pkg.ExecBashCmd(backend.BackendDir, "poetry", "run", "pre-commit", "install")
	},
}

func init() {
	backend.BackendCmd.AddCommand(InstallCmd)
	InstallCmd.Flags().BoolP("force", "f", false, "Force a reinstall of the backend")
}
