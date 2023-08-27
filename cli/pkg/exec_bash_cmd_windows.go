//go:build windows

package pkg

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
)

func ExecBashCmd(dir string, name string, arg ...string) {
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
}
