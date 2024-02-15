import { it } from "vitest";
import { initProject } from "sst/project";
import { App, getStack } from "sst/constructs";
import { AppStack } from "./AppStack";
import { Template } from "aws-cdk-lib/assertions";

it("Check Appstack for User Endpoints", async () => {
  await initProject({});
  const app = new App({ mode: "deploy" });
  // WHEN
  app.stack(AppStack);
  // THEN
  const template = Template.fromStack(getStack(AppStack));
  template.hasOutput("CreateUserFunctionName", Object);
  template.hasOutput("GetUserFunctionName", Object);
  template.hasOutput("DeleteUserFunctionName", Object);
});