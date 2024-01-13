# SST Endpoint Runbook

This doc outlines the process to setup and use SST (Serverless Stack Toolkit). SST in short is a tool that allows us to easily add AWS to full stack applications. 

## Prerequisties
1. Having access to AWS using SSO (if you don't have this, ask the project leads)
1. Configuring your AWS credentials via this [section](https://github.com/DSGT-DLP/Deep-Learning-Playground?tab=readme-ov-file#5-aws-setup) in the project `README.md`. **Please ensure you follow these steps carefully!!!**

## Installation
First, install the SST related  in your local terminal:
### With `dlp-cli`
`dlp-cli serverless install`

### Without `dlp-cli`
Assuming you're at the `~/Deep-Learning-Playground` directory, run the following commands (if you don't have `pnpm`, install it via `npm install -g pnpm`):
```
cd serverless
pnpm install
```

## Dev Testing
If you are making changes to `~/serverless` and want to dev test your changes, you can do so by the following (assuming you are in `~/Deep-Learning-Playground/serverless`):

```
pnpm run dev
```

**NOTE: When you run `pnpm run dev`, you are spinning up a development environment using Live Lambda Dev. Instead of making requests to production endpoint, the request is proxied to your development environment**

## Viewing your Resources from Dev Testing
You can then login to AWS using your SSO credentials and navigate to the console to see your resources 

## Support/Documentation
1. [Documentation](https://docs.sst.dev/)
1. [Discord Group](https://sst.dev/discord)