package uid

/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// UidCmd represents the Uid command
var UidCmd = &cobra.Command{
	Use:   "uid [email]",
	Short: "gets a user's uid by email",
	Long:  `gets a user's uid by email from the backend`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		bash_cmd := exec.Command("poetry", "run", "python", "cli.py", "get-uid", args[0])
		bash_cmd.Dir = backend.BackendDir
		fmt.Println(strings.Join(bash_cmd.Args, " "))
		pkg.ExecBashCmd(bash_cmd)
	},
}

func init() {
	backend.BackendCmd.AddCommand(UidCmd)
	//IdTokenCmd.Flags().StringP("email", "")
}
