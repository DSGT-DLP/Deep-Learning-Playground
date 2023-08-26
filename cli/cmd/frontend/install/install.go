/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package install

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/frontend"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// InstallCmd represents the frontend install command
var InstallCmd = &cobra.Command{
	Use:   "install",
	Short: "Installs packages from package.json",
	Long:  `Installs frontend packages from package.json from /frontend in node_modules`,
	Args:  cobra.ExactArgs(0),
	Run: func(cmd *cobra.Command, args []string) {
		bash_args := []string{"install"}
		if cmd.Flag("force").Value.String() == "true" {
			bash_args = append(bash_args, "--force")
		}
		pkg.ExecBashCmd(frontend.FrontendDir, "yarn", bash_args...)
	},
}

func init() {
	frontend.FrontendCmd.AddCommand(InstallCmd)
	InstallCmd.Flags().BoolP("force", "f", false, "Force a reinstall of the frontend")
}
