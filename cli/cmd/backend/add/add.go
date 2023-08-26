/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package add

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// AddCmd represents the backend add command
var AddCmd = &cobra.Command{
	Use:   "add {package}",
	Short: "Add package to pyproject.toml",
	Long:  `Add package to pyproject.toml from /training`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		bash_args := []string{"add", args[0]}
		if cmd.Flag("dev").Value.String() == "true" {
			bash_args = append(bash_args, "--group", "dev")
		}
		/*
			bash_cmd := exec.Command("poetry", bash_args...)
			bash_cmd.Dir = backend.BackendDir
			err := bash_cmd.Run()
			cmd.Println(err)*/
		//pkg.ExecBashCmd(backend.BackendDir, "ls")
		//pkg.ExecBashCmd(backend.BackendDir, "poetry", "show")
		pkg.ExecBashCmd(backend.BackendDir, "poetry", bash_args...)
	},
}

func init() {
	backend.BackendCmd.AddCommand(AddCmd)
	AddCmd.Flags().BoolP("dev", "d", false, "Add package as dev dependency")
}
