#!/usr/bin/env node
import "source-map-support/register";

import * as cdk from "aws-cdk-lib";
import { InferenceStack } from "../lib/stack/inference";

const app = new cdk.App();

new InferenceStack(app, "InferenceStack", {});
