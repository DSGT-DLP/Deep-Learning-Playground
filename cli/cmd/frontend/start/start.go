/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package frontend

import (
	"fmt"

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
		pkg.ExecBashCmd(frontend.FrontendDir, "yarn", "next", "dev", "-p", fmt.Sprintf("%v", cmd.Flag("port").Value))
	},
}

func init() {
	frontend.FrontendCmd.AddCommand(StartCmd)
	StartCmd.PersistentFlags().IntP("port", "p", 3000, "A port to run the frontend on")
}
