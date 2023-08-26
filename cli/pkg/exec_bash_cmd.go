package pkg

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"

	"github.com/creack/pty"
	"golang.org/x/term"
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

	// Handle pty size.
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGWINCH)
	go func() {
		for range ch {
			if err := pty.InheritSize(os.Stdin, ptmx); err != nil {
				log.Printf("error resizing pty: %s", err)
			}
		}
	}()
	ch <- syscall.SIGWINCH                        // Initial resize.
	defer func() { signal.Stop(ch); close(ch) }() // Cleanup signals when done.

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
}

/*
func ExecBashCmd2(dir string, name string, arg ...string) {
	// Use this if the pty one doesn't work
	bash_cmd := exec.Command(name, arg...)
	bash_cmd.Dir = dir
	fmt.Println(strings.Join(bash_cmd.Args, " "))

	stdoutPipe, _ := bash_cmd.StdoutPipe()
	stderrPipe, _ := bash_cmd.StderrPipe()
	err := bash_cmd.Start()
	if err != nil {
		fmt.Println("Error starting cmd: ", err)
		return
	}
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		io.Copy(os.Stdout, stdoutPipe)
	}()

	go func() {
		defer wg.Done()
		io.Copy(os.Stderr, stderrPipe)
	}()

	wg.Wait()
	err = bash_cmd.Wait()
	if err != nil {
		fmt.Println("Error waiting for cmd: ", err)
		return
	}
}*/
