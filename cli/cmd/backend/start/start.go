/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package start

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// StartCmd represents the backend start command
var StartCmd = &cobra.Command{
	Use:   "start",
	Short: "Starts the training backend",
	Long:  `Starts an instance of the training backend Django app in /training in the terminal`,
	Args:  cobra.ExactArgs(0),
	Run: func(cmd *cobra.Command, args []string) {
		bash_cmd := exec.Command("poetry", "run", "python", "manage.py", "runserver", fmt.Sprintf("%v", cmd.Flag("port").Value))
		bash_cmd.Dir = backend.BackendDir
		cmd.Println(strings.Join(bash_cmd.Args, " "))
		pkg.ExecBashCmd(bash_cmd)

	},
}

func init() {
	backend.BackendCmd.AddCommand(StartCmd)
	StartCmd.PersistentFlags().IntP("port", "p", 8000, "A port to run the backend on")
}
