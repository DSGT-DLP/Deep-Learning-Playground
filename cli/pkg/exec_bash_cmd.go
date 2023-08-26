package pkg

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/creack/pty"
)

func ExecBashCmd(dir string, name string, arg ...string) {
	// Code below found in pty examples: https://github.com/creack/pty
	bash_cmd := exec.Command(name, arg...)
	bash_cmd.Dir = dir
	fmt.Println(strings.Join(bash_cmd.Args, " "))
	ptmx, err := pty.Start(bash_cmd)
	if err != nil {
		panic(err)
	}
	// Make sure to close the pty at the end.
	defer func() { _ = ptmx.Close() }() // Best effort.
	// Set stdin in raw mode.
	/*
		oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
		if err != nil {
			panic(err)
		}
		defer func() { _ = term.Restore(int(os.Stdin.Fd()), oldState) }() // Best effort.*/

	// Copy stdin to the pty and the pty to stdout.
	// NOTE: The goroutine will keep reading until the next keystroke before returning.
	go func() { io.Copy(ptmx, os.Stdin) }()
	io.Copy(os.Stdout, ptmx)
}
