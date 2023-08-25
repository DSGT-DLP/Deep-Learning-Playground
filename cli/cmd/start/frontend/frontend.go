/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package frontend

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/start"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// FrontendCmd represents the frontend command
var FrontendCmd = &cobra.Command{
	Use:   "frontend",
	Short: "Starts the frontend",
	Long:  `Starts an instance of the training backend Django app in /training in the terminal`,
	Args:  cobra.ExactArgs(0),
	Run: func(cmd *cobra.Command, args []string) {
		bash_cmd := exec.Command("yarn", "next", "dev", "-p", fmt.Sprintf("%v", cmd.Flag("port").Value))
		bash_cmd.Dir = "./frontend"
		fmt.Println(strings.Join(bash_cmd.Args, " "))
		pkg.ExecBashCmd(bash_cmd)
	},
}

func init() {
	start.StartCmd.AddCommand(FrontendCmd)
	FrontendCmd.PersistentFlags().Int("port", 3000, "A port to run the frontend on")
}
