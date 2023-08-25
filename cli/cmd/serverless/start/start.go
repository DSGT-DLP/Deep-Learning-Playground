/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package serverless

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// StartCmd represents the serverless start command
var StartCmd = &cobra.Command{
	Use:   "start",
	Short: "Starts the serverless environment",
	Long:  `Starts SST's Live Lambda Development environment in the terminal`,
	Args:  cobra.ExactArgs(0),
	Run: func(cmd *cobra.Command, args []string) {
		bash_cmd := exec.Command("yarn", "sst", "dev")
		bash_cmd.Dir = serverless.ServerlessDir
		fmt.Println(strings.Join(bash_cmd.Args, " "))
		pkg.ExecBashCmd(bash_cmd)
	},
}

func init() {
	serverless.ServerlessCmd.AddCommand(StartCmd)
}
