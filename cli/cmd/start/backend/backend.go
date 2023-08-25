/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package backend

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/start"
	"github.com/creack/pty"
	"github.com/spf13/cobra"
	"golang.org/x/term"
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
		fmt.Println(bash_cmd.Dir)
		fmt.Println(strings.Join(bash_cmd.Args, " "))
		// Code below found in pty examples: https://github.com/creack/pty
		ptmx, err := pty.Start(bash_cmd)
		if err != nil {
			panic(err)
		}
		// Make sure to close the pty at the end.
		defer func() { _ = ptmx.Close() }() // Best effort.
		// Set stdin in raw mode.
		oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
		if err != nil {
			panic(err)
		}
		defer func() { _ = term.Restore(int(os.Stdin.Fd()), oldState) }() // Best effort.

		// Copy stdin to the pty and the pty to stdout.
		// NOTE: The goroutine will keep reading until the next keystroke before returning.
		go func() { io.Copy(ptmx, os.Stdin) }()
		io.Copy(os.Stdout, ptmx)
	},
}

func init() {
	start.StartCmd.AddCommand(BackendCmd)
	BackendCmd.PersistentFlags().Int("port", 8000, "A port to run the backend on")
}
