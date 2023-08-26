/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package main

import (
	"github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend/add"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend/id_token"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend/install"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend/remove"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend/start"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/backend/uid"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/frontend"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/frontend/add"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/frontend/install"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/frontend/remove"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/frontend/start"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/core"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/core/add"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/core/remove"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/functions"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/functions/add"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/functions/remove"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/install"
	_ "github.com/DSGT-DLP/Deep-Learning-Playground/cli/cmd/serverless/start"
)

func main() {
	cmd.Execute()
}
