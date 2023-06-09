def test_all_imports():
    import backend.aws_helpers.aws_rekognition_utils.rekognition_client
    import backend.aws_helpers.aws_secrets_utils.build_env
    import backend.aws_helpers.dynamo_db_utils.execution_db
    import backend.aws_helpers.dynamo_db_utils.trainspace_db
    import backend.aws_helpers.dynamo_db_utils.userprogress_db
    import backend.aws_helpers.lambda_utils.lambda_client
    import backend.aws_helpers.s3_utils.s3_client
    import backend.aws_helpers.sqs_utils.sqs_client
    import backend.common.constants
    import backend.common.dataset
    import backend.common.default_datasets
    import backend.common.email_notifier
    import backend.common.loss_functions
    import backend.common.optimizer
    import backend.common.utils
    import backend.dl.dl_eval
    import backend.dl.dl_model
    import backend.dl.dl_model_parser
    import backend.dl.dl_trainer
    import backend.ml.ml_trainer
