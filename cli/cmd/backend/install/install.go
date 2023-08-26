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
	Short: "Installs training backend packages from pyproject.toml",
	Long:  `Installs training backend packages from pyproject.toml from /training in .venv`,
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
