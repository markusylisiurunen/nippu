import * as cdk from "aws-cdk-lib";
import * as apigateway from "aws-cdk-lib/aws-apigateway";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as sfn from "aws-cdk-lib/aws-stepfunctions";
import * as tasks from "aws-cdk-lib/aws-stepfunctions-tasks";
import { Construct } from "constructs";
import * as path from "path";

export type InferenceStackProps = cdk.StackProps & {};

/**
 * The inference flow is roughly the following:
 *
 * 1. An endpoint is invoked with a base64 encoded image of a receipt.
 * 2. OCR is performed on the image and text blocks are extracted.
 * 3. Given the image and the text blocks, they are passed through the trained model.
 * 4. The model output if transformed into the response.
 */
export class InferenceStack extends cdk.Stack {
  private readonly props: InferenceStackProps;

  constructor(scope: Construct, id: string, props: InferenceStackProps) {
    super(scope, id, props);
    this.props = props;

    const [ocr, inference, transform] = [
      this.createOCRLambda(),
      this.createInferenceLambda(),
      this.createTransformLambda(),
    ];

    this.createStateMachine(ocr, inference, transform);
  }

  private createOCRLambda(): lambda.Function {
    const ocrLambda = new lambda.Function(this, "OCRLambda", {
      code: lambda.Code.fromAsset(path.join(__dirname, "..", "lambda")),
      handler: "ocr.handler",
      runtime: lambda.Runtime.PYTHON_3_9,
    });

    // FIXME: this is obviously not good, I'm just so fucking tired of this IAM bullshit
    ocrLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        resources: ["*"],
        actions: ["*"],
      })
    );

    return ocrLambda;
  }

  private createInferenceLambda(): lambda.Function {
    return new lambda.Function(this, "InferenceLambda", {
      code: lambda.Code.fromAsset(path.join(__dirname, "..", "lambda")),
      handler: "inference.handler",
      runtime: lambda.Runtime.PYTHON_3_9,
    });
  }

  private createTransformLambda(): lambda.Function {
    return new lambda.Function(this, "TransformLambda", {
      code: lambda.Code.fromAsset(path.join(__dirname, "..", "lambda")),
      handler: "transform.handler",
      runtime: lambda.Runtime.PYTHON_3_9,
    });
  }

  private createStateMachine(
    ocrLambda: lambda.Function,
    inferenceLambda: lambda.Function,
    transformLambda: lambda.Function
  ): void {
    const machineDefinition = new sfn.Parallel(this, "PerformOCR")
      .branch(
        new sfn.Pass(this, "PassImage", {
          outputPath: "$.body",
        })
      )
      .branch(
        new tasks.LambdaInvoke(this, "OCRLambdaInvoke", {
          inputPath: "$.body",
          lambdaFunction: ocrLambda,
          outputPath: "$.Payload.body",
        })
      )
      .next(
        new sfn.Pass(this, "FuseImageAndOCR", {
          parameters: {
            "image.$": "$[0]",
            "ocr.$": "$[1]",
          },
        })
      )
      .next(
        new tasks.LambdaInvoke(this, "InferenceLambdaInvoke", {
          lambdaFunction: inferenceLambda,
          outputPath: "$.Payload.body",
        })
      )
      .next(
        new tasks.LambdaInvoke(this, "TransformLambdaInvoke", {
          lambdaFunction: transformLambda,
          outputPath: "$.Payload.body",
        })
      );

    const stateMachine = new sfn.StateMachine(this, "InferenceStateMachine", {
      definition: machineDefinition,
      stateMachineType: sfn.StateMachineType.EXPRESS,
      logs: {
        destination: new logs.LogGroup(this, "InferenceStateMachineLogs"),
        level: sfn.LogLevel.ERROR,
      },
    });

    this.createAPIGateway(stateMachine);
  }

  private createAPIGateway(stateMachine: sfn.StateMachine): void {
    new apigateway.StepFunctionsRestApi(this, "InferenceRestAPI", {
      stateMachine: stateMachine,
    });
  }
}
