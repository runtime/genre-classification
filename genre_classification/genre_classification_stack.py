from aws_cdk import Stack, RemovalPolicy, CfnOutput
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_iam as iam
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_rds as rds
from constructs import Construct

class GenreClassificationStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create the VPC once and reuse it
        vpc = self.create_vpc()

        # Security group for RDS
        rds_sg = ec2.SecurityGroup(self, "RDSecurityGroup", vpc=vpc)
        rds_sg.add_ingress_rule(ec2.Peer.ipv4(vpc.vpc_cidr_block), ec2.Port.tcp(5432), "Allow Postgres access within VPC")

        # First RDS instance for original data
        rds_instance = rds.DatabaseInstance(self, "GenreClassificationDB",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_15),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MICRO),
            vpc=vpc,
            security_groups=[rds_sg],
            multi_az=False,
            allocated_storage=20,
            max_allocated_storage=100,
            removal_policy=RemovalPolicy.DESTROY,
            credentials=rds.Credentials.from_generated_secret("postgres"),
            publicly_accessible=False,
            database_name="media_db"
        )

        # Second RDS instance for processed data
        rds_instance_processed = rds.DatabaseInstance(self, "ProcessedMediaDB",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_15),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MICRO),
            vpc=vpc,
            security_groups=[rds_sg],
            allocated_storage=20,
            max_allocated_storage=100,
            removal_policy=RemovalPolicy.DESTROY,
            credentials=rds.Credentials.from_generated_secret("postgres"),
            publicly_accessible=False,
            database_name="processed_media_db"
        )

        # Security group for EC2
        ec2_sg = ec2.SecurityGroup(self, "EC2SecurityGroup", vpc=vpc)
        ec2_sg.add_ingress_rule(rds_sg, ec2.Port.tcp(5432), "Allow EC2 to connect to RDS")

        # IAM role for EC2 with SSM
        ec2_role = iam.Role(self, "EC2SSMRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonRDSFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess")
            ]
        )

        # EC2 instance for accessing RDS
        ec2_instance = ec2.Instance(self, "GenreClassificationEC2",
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            vpc=vpc,
            security_group=ec2_sg,
            role=ec2_role,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
        )

        # Lambda for ingestion
        ingestion_lambda = _lambda.Function(self, "GenreIngestionLambda",
            runtime=_lambda.Runtime.PYTHON_3_9,
            handler="app.handler",
            code=_lambda.Code.from_asset("lambdas/ingest_data"),
            environment={
                "DB_HOST": rds_instance.db_instance_endpoint_address,
                "DB_NAME": "media_db",
                "DB_USER": "postgres",
                "DB_PASSWORD": rds_instance.secret.secret_value_from_json("password").unsafe_unwrap(),
            }
        )

        # Grant Lambda access to RDS secrets
        rds_instance.secret.grant_read(ingestion_lambda)

        # Output EC2 Instance ID and RDS Endpoint
        self.output_ec2_and_rds_info(ec2_instance, rds_instance)

    def create_vpc(self):
        return ec2.Vpc(self, "GenreClassificationVPC",
            cidr="10.1.0.0/16",  # Unique CIDR block
            max_azs=2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24  # Smaller subnets
                ),
                ec2.SubnetConfiguration(
                    name="PrivateWithNat",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24
                )
            ],
            nat_gateways=1
        )

    def output_ec2_and_rds_info(self, ec2_instance, rds_instance):
        CfnOutput(self, "EC2InstanceID", value=ec2_instance.instance_id, description="EC2 Instance ID")
        CfnOutput(self, "RDSEndpoint", value=rds_instance.db_instance_endpoint_address, description="RDS Endpoint")
