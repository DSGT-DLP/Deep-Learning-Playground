/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package backend

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/start"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// BackendCmd represents the backend command
var BackendCmd = &cobra.Command{
	Use:   "backend",
	Short: "Starts the training backend",
	Long:  `Starts an instance of the training backend Django app in /training in the terminal`,
	Args:  cobra.ExactArgs(0),
	Run: func(cmd *cobra.Command, args []string) {
		bash_cmd := exec.Command("poetry", "run", "python", "manage.py", "runserver", fmt.Sprintf("%v", cmd.Flag("port").Value))
		bash_cmd.Dir = "./training"
		fmt.Println(strings.Join(bash_cmd.Args, " "))
		pkg.ExecBashCmd(bash_cmd)
	},
}

func init() {
	start.StartCmd.AddCommand(BackendCmd)
	BackendCmd.PersistentFlags().Int("port", 8000, "A port to run the backend on")
}
