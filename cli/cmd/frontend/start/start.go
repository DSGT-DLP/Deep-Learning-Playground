/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package frontend

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/frontend"
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/pkg"
	"github.com/spf13/cobra"
)

// StartCmd represents the frontend start command
var StartCmd = &cobra.Command{
	Use:   "start",
	Short: "Starts the frontend",
	Long:  `Starts an instance of the nextjs frontend in the terminal`,
	Args:  cobra.ExactArgs(0),
	Run: func(cmd *cobra.Command, args []string) {
		bash_cmd := exec.Command("yarn", "next", "dev", "-p", fmt.Sprintf("%v", cmd.Flag("port").Value))
		bash_cmd.Dir = "./frontend"
		fmt.Println(strings.Join(bash_cmd.Args, " "))
		pkg.ExecBashCmd(bash_cmd)
	},
}

func init() {
	frontend.FrontendCmd.AddCommand(StartCmd)
	StartCmd.PersistentFlags().IntP("port", "p", 3000, "A port to run the frontend on")
}
