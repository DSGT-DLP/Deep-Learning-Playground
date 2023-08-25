package id_token

/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/

import (
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// IdTokenCmd represents the IdToken command
var IdTokenCmd = &cobra.Command{
	Use:   "id-token [email]",
	Short: "gets a user's id token by email",
	Long:  `gets a user's id token by email from the backend`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		bash_cmd := exec.Command("poetry", "run", "python", "cli.py", "get-id-token", args[0])
		bash_cmd.Dir = backend.BackendDir
		cmd.Println(strings.Join(bash_cmd.Args, " "))
		pkg.ExecBashCmd(bash_cmd)
	},
}

func init() {
	backend.BackendCmd.AddCommand(IdTokenCmd)
	//IdTokenCmd.Flags().StringP("email", "")
}
