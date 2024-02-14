import { it } from "vitest";
import { initProject } from "sst/project";
import { App, getStack } from "sst/constructs";
import { AppStack } from "./AppStack";
import { Template } from "aws-cdk-lib/assertions/index";
//import { Template } from "../node_modules/.pnpm/aws-cdk-lib@2.101.1_constructs@10.2.69/node_modules/aws-cdk-lib/assertions/index";

it("Check Appstack for User Endpoints", async () => {
  await initProject({});
  const app = new App({ mode: "deploy" });
  // WHEN
  app.stack(AppStack);
  // THEN
  const template = Template.fromStack(getStack(AppStack));
  //console.log(template)
  template.hasOutput("CreateUserFunctionName", Object);
  template.hasOutput("GetUserFunctionName", Object);
  template.hasOutput("DeleteUserFunctionName", Object);
});