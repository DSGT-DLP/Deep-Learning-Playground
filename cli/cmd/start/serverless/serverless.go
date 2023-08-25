/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package serverless

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/start"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// ServerlessCmd represents the serverless command
var ServerlessCmd = &cobra.Command{
	Use:   "serverless",
	Short: "Starts the serverless environment",
	Long:  `Starts SST's Live Lambda Development environment in the terminal`,
	Args:  cobra.ExactArgs(0),
	Run: func(cmd *cobra.Command, args []string) {
		bash_cmd := exec.Command("yarn", "sst", "dev")
		bash_cmd.Dir = "./serverless"
		fmt.Println(strings.Join(bash_cmd.Args, " "))
		pkg.ExecBashCmd(bash_cmd)
	},
}

func init() {
	start.StartCmd.AddCommand(ServerlessCmd)
}
